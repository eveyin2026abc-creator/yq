"""Orchestration for query-driven shape-grid generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

from tensor_cast.performance_model.profiling_database.query_demand import (
    KernelQueryDemand,
    load_query_demand_traces,
)

try:
    from ..signature_utils import normalize_op_name
except ImportError:
    from signature_utils import normalize_op_name

from .query_coverage import build_query_generated_rows
from .query_model import resolve_query_model_architecture
from .query_workloads import (
    CommandRunner,
    QueryWorkloadRunResult,
    QUERY_WORKLOAD_POLICY_VERSION,
    build_workload_scenarios,
    run_query_workloads,
)
from .config import load_op_mapping_metadata, load_shape_grid_config
from .theory_fallback import build_theory_fallback_rows, theory_generation_is_skipped
from .utils import load_csv_template_rows, replace_csv_with_generated_rows


@dataclass(frozen=True)
class OperatorGenerationResult:
    csv_path: Path
    appended_rows: int
    demand_count: int
    projected_exact: int
    rejected: int
    duplicates: int
    reason: str = "generated"


@dataclass(frozen=True)
class QueryGenerationResult:
    workloads: QueryWorkloadRunResult
    operators: tuple[OperatorGenerationResult, ...]
    captured_demands: int

    @property
    def total_appended_rows(self) -> int:
        return sum(item.appended_rows for item in self.operators)

    @property
    def generated_files(self) -> tuple[Path, ...]:
        return tuple(item.csv_path for item in self.operators if item.appended_rows)

    @property
    def skipped_files(self) -> tuple[Path, ...]:
        return tuple(item.csv_path for item in self.operators if not item.appended_rows)


@dataclass(frozen=True)
class _OperatorWritePlan:
    csv_path: Path
    headers: list[str]
    source_rows: list[dict[str, str]]
    generated_rows: list[dict[str, str]]


def discover_replay_supported_ops(op_replay_dir: Path) -> tuple[str, ...]:
    """Discover the support boundary from the real replay entry points."""
    return tuple(
        sorted(
            normalize_op_name(path.name)
            for path in op_replay_dir.glob("*_run.py")
            if path.name != "run_all_op.py"
        )
    )


def iter_csv_files(data_dir: Path) -> Iterable[Path]:
    return sorted(path for path in data_dir.rglob("*.csv") if f".tmp{path.suffix}" not in path.name)


def load_csv_files(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise ValueError(f"Database directory does not exist: {data_dir}")
    csv_files = list(iter_csv_files(data_dir))
    if not csv_files:
        raise ValueError(f"No CSV files found under database directory: {data_dir}")
    return csv_files


def _update_digest_from_file(digest, path: Path, *, label: str) -> None:
    digest.update(label.encode("utf-8"))
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)


def _update_digest_from_stat(digest, path: Path, *, label: str) -> None:
    """Update *digest* using O(1) stat fingerprint (size + mtime_ns).

    Avoids reading full file contents for large databases (35+ CSVs) on every
    run. Content changes still trigger cache invalidation via mtime_ns/size.
    """
    digest.update(label.encode("utf-8"))
    stat = path.stat()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))


def _query_cache_directory(data_dir: Path, model_ids: list[str], repo_root: Path) -> Path:
    """Return an automatic cache isolated by model, database, and query semantics."""
    digest = hashlib.sha256()
    digest.update(QUERY_WORKLOAD_POLICY_VERSION.encode("utf-8"))
    digest.update("\0".join(sorted(model_ids)).encode("utf-8"))
    for path in [data_dir / "op_mapping.yaml", *iter_csv_files(data_dir)]:
        _update_digest_from_file(digest, path, label=path.relative_to(data_dir).as_posix())
    semantic_sources = (
        "tools/perf_data_collection/grid_generator/query_workloads.py",
        "tools/perf_data_collection/grid_generator/query_model.py",
        "tensor_cast/performance_model/profiling_database/query_demand.py",
        "tensor_cast/performance_model/profiling_database/backend_projector.py",
        "tensor_cast/performance_model/profiling_database/profiling_data_source.py",
        "tensor_cast/performance_model/profiling_database/interpolating_data_source.py",
    )
    for relative_path in semantic_sources:
        source_path = repo_root / relative_path
        if source_path.is_file():
            _update_digest_from_file(digest, source_path, label=relative_path)
    return Path(tempfile.gettempdir()) / "msmodeling-shape-query-cache" / digest.hexdigest()[:32]


def _load_completed_query_demands(workload_result: QueryWorkloadRunResult, trace_dir: Path) -> list[KernelQueryDemand]:
    demands: list[KernelQueryDemand] = []
    seen: set[str] = set()
    completed_dirs = workload_result.trace_directories or (trace_dir,)
    for completed_dir in completed_dirs:
        for demand in load_query_demand_traces(completed_dir):
            if demand.signature in seen:
                continue
            seen.add(demand.signature)
            demands.append(demand)
    return demands


def normalize_target_models(raw_models: str | Iterable[str]) -> list[str]:
    values = [raw_models] if isinstance(raw_models, str) else list(raw_models)
    models = [part.strip() for value in values for part in str(value).split(",") if part.strip()]
    if not models:
        raise ValueError("--target-models requires at least one HuggingFace model ID")
    return list(dict.fromkeys(models))


def normalize_selected_ops(raw_ops: Iterable[str] | None, supported_ops: Iterable[str]) -> set[str] | None:
    if raw_ops is None:
        return None
    supported = set(supported_ops)
    selected = {normalize_op_name(value) for value in raw_ops}
    unknown = sorted(selected - supported)
    if unknown:
        raise ValueError(
            "--ops contains operators without an op_replay entry point: "
            f"{', '.join(unknown)}. Supported operators: {', '.join(sorted(supported))}"
        )
    return selected


def _collect_query_demands(
    *,
    model_ids: list[str],
    database_path: Path,
    trace_dir: Path,
    repo_root: Path,
    command_runner: CommandRunner,
    allow_empty: bool = False,
) -> tuple[list[KernelQueryDemand], QueryWorkloadRunResult]:
    scenarios = []
    for model_id in model_ids:
        config = resolve_query_model_architecture(model_id)
        scenarios.extend(build_workload_scenarios(model_id, config, database_path))
    unique = {scenario.workload_id: scenario for scenario in scenarios}
    ordered_scenarios = [unique[key] for key in sorted(unique)]
    workload_result = run_query_workloads(
        ordered_scenarios,
        database_path=database_path,
        trace_dir=trace_dir,
        repo_root=repo_root,
        command_runner=command_runner,
    )
    demands = _load_completed_query_demands(workload_result, trace_dir)
    if workload_result.succeeded == 0:
        raise RuntimeError(
            "All internal throughput_optimizer workloads failed; no trustworthy shape demand was produced. "
            "Inspect the [QUERY] failure output above."
        )
    if not demands and not allow_empty:
        raise RuntimeError(
            "Optimizer workloads completed but captured no profiling-database query. "
            "Verify that the profiling performance model is active for this database."
        )
    return demands, workload_result


def run_query_mode(
    args: argparse.Namespace,
    *,
    data_dir: Path,
    op_replay_dir: Path,
    repo_root: Path,
    command_runner: CommandRunner = subprocess.run,
) -> QueryGenerationResult:
    """Capture optimizer queries, plan local coverage, and append replay rows."""
    if args.rows <= 0:
        raise ValueError("--rows must be greater than 0")
    model_ids = normalize_target_models(args.target_models)
    mapping_path = data_dir / "op_mapping.yaml"
    if not mapping_path.is_file():
        raise ValueError(f"op_mapping.yaml does not exist under database path: {data_dir}")
    supported_ops = discover_replay_supported_ops(op_replay_dir)
    selected_ops = normalize_selected_ops(args.ops, supported_ops)
    csv_files = load_csv_files(data_dir)
    csv_by_op: dict[str, Path] = {}
    duplicate_stems: set[str] = set()
    for path in csv_files:
        if path.stem in csv_by_op:
            duplicate_stems.add(path.stem)
        csv_by_op[path.stem] = path
    if duplicate_stems:
        raise ValueError(
            "Database path contains duplicate operator CSV names; pass one versioned database directory: "
            + ", ".join(sorted(duplicate_stems))
        )
    available_ops = set(csv_by_op) & set(supported_ops)
    if selected_ops is not None:
        missing_csv = sorted(selected_ops - set(csv_by_op))
        if missing_csv:
            raise ValueError(
                "Selected replay-supported operators have no CSV in the target database: "
                + ", ".join(missing_csv)
            )
    elif not available_ops:
        raise ValueError("Target database contains no CSV with a matching op_replay entry point")

    print(
        "Query-driven shape generation: "
        f"models={model_ids}, requested_ops={sorted(selected_ops) if selected_ops else 'model queries'}, "
        f"rows/csv={args.rows}, seed={args.seed}"
    )
    trace_dir = _query_cache_directory(data_dir, model_ids, repo_root)
    print(f"[QUERY] automatic checkpoint cache: {trace_dir}")
    demands, workload_result = _collect_query_demands(
        model_ids=model_ids,
        database_path=data_dir,
        trace_dir=trace_dir,
        repo_root=repo_root,
        command_runner=command_runner,
        allow_empty=selected_ops is not None,
    )

    demands_by_kernel: dict[str, list[KernelQueryDemand]] = {}
    for demand in demands:
        if demand.kernel_type in available_ops:
            demands_by_kernel.setdefault(demand.kernel_type, []).append(demand)

    target_ops = selected_ops if selected_ops is not None else set(demands_by_kernel)
    if not target_ops:
        raise ValueError(
            "Target model workloads queried no operator with both a database CSV and op_replay entry point"
        )

    operator_results: list[OperatorGenerationResult] = []
    write_plans: list[_OperatorWritePlan] = []
    theory_config: dict | None = None
    op_meta: dict[str, dict] | None = None
    for kernel_type in sorted(target_ops):
        csv_path = csv_by_op[kernel_type]
        kernel_demands = demands_by_kernel.get(kernel_type, [])
        loaded = load_csv_template_rows(csv_path, require_rows=True)
        if loaded is None:
            raise ValueError(f"{csv_path} is missing a usable Shape schema")
        headers, source_rows = loaded
        if kernel_demands:
            generated_rows, summary = build_query_generated_rows(
                csv_path=csv_path,
                headers=headers,
                source_rows=source_rows,
                demands=kernel_demands,
                row_limit=args.rows,
                seed=args.seed,
            )
            result = OperatorGenerationResult(
                csv_path=csv_path,
                appended_rows=len(generated_rows),
                demand_count=summary["demands"],
                projected_exact=summary["projected_exact"],
                rejected=summary["rejected"],
                duplicates=summary["duplicates"],
                reason="generated" if generated_rows else "no_new_candidate",
            )
        else:
            if theory_config is None:
                theory_config = load_shape_grid_config(Path(__file__).with_name("config.yaml"))
                op_meta = load_op_mapping_metadata(data_dir)
            assert op_meta is not None
            if theory_generation_is_skipped(kernel_type, theory_config, op_meta):
                operator_results.append(
                    OperatorGenerationResult(
                        csv_path,
                        0,
                        0,
                        0,
                        0,
                        0,
                        reason="theory_skipped",
                    )
                )
                print(f"[GRID] {kernel_type}: skipped by generic Shape policy")
                continue
            fallback = build_theory_fallback_rows(
                kernel_type=kernel_type,
                model_names=model_ids,
                config=theory_config,
                op_meta=op_meta,
                csv_path=csv_path,
                headers=headers,
                source_rows=source_rows,
                row_limit=args.rows,
            )
            if fallback is None:
                operator_results.append(
                    OperatorGenerationResult(
                        csv_path,
                        0,
                        0,
                        0,
                        0,
                        0,
                        reason="theory_skipped",
                    )
                )
                print(f"[GRID] {kernel_type}: skipped because no generic Shape generator exists")
                continue
            generated_rows, summary = fallback
            result = OperatorGenerationResult(
                csv_path=csv_path,
                appended_rows=len(generated_rows),
                demand_count=0,
                projected_exact=0,
                rejected=0,
                duplicates=summary["duplicates"],
                reason="generated" if generated_rows else "no_new_candidate",
            )
        operator_results.append(result)
        if generated_rows:
            write_plans.append(
                _OperatorWritePlan(
                    csv_path=csv_path,
                    headers=headers,
                    source_rows=source_rows,
                    generated_rows=generated_rows,
                )
            )
        print(
            f"[GRID] {kernel_type}: demand={result.demand_count}, exact={result.projected_exact}, "
            f"rejected={result.rejected}, duplicate={result.duplicates}, appended={result.appended_rows}",
            end="",
        )
        fallback_appended = summary.get("fallback_appended", 0)
        if fallback_appended:
            fallback_attempted = summary.get("fallback_attempted", 0)
            fallback_duplicates = summary.get("fallback_duplicates", 0)
            fallback_rejected_safety = summary.get("fallback_rejected_safety", 0)
            print(
                f" | constraint_fallback: attempted={fallback_attempted}, "
                f"safety_rejected={fallback_rejected_safety}, "
                f"duplicate={fallback_duplicates}, appended={fallback_appended}",
                end="",
            )
        print()

    for plan in write_plans:
        replace_csv_with_generated_rows(
            plan.csv_path,
            plan.headers,
            plan.source_rows,
            plan.generated_rows,
        )
    return QueryGenerationResult(
        workloads=workload_result,
        operators=tuple(operator_results),
        captured_demands=len(demands),
    )
