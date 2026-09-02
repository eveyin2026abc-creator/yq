# tests/ — Test layout, markers, and authoring

Entry scripts, environment variables, CI triggers → [scripts/README.md](../scripts/README.md).

## Layout

```bash
tests/
├── conftest.py              # Hub offline, cache paths, session-end weight cleanup
├── .ci/
│   ├── gate_policy.yaml     # CI gate roots / tests / configs / exemptions
│   └── approvers.yaml       # Required approvers for gate_policy changes
├── smoke/                   # Cross-layer cheap guards (not L1/L2/L3)
├── regression/              # Functional / integration
│   ├── tensor_cast/         # model/ (L1 model-level) + module/ (L2 no-model integration) + unit/ (L3)
│   ├── serving_cast/        # Out of current L1/L2/L3 wave
│   ├── cli/                 # model/ (L1 real runner) + module/ (L2 no-model orchestration) + unit/ (L3)
│   ├── optix/               # Out of current L1/L2/L3 wave
│   ├── model_diagnostics/   # e2e = L1; synthetic application integration = L2; pure contracts = L3
│   ├── scripts/             # ci_gate/nightly toolchain UT (mirrors scripts/helpers/)
│   └── web_ui/              # Out of current L1/L2/L3 wave
├── assets/                  # Model config fixtures
├── helpers/                 # Shared builders/assertions
└── benchmark/
    ├── models/              # L1 accuracy net (JSON total-time / Top-N; independent schedule)
    └── ops/                 # Operator DB guards (not L1/L2)
```

CI layering is by directory (`smoke`/`regression`/`benchmark`). Markers: `nightly` (long-running), `npu` (hardware), `network` (live Hub).

Responsibility layers L1 / L2 / L3 sit **on top of** those directories. Design and deletion gate: [docs/design/ut_refactor.md](../docs/design/ut_refactor.md). File-level migration list: [docs/design/ut_refactor_migration_inventory.md](../docs/design/ut_refactor_migration_inventory.md).
Seven-artifact case inventory: [docs/design/l2_seven_artifact_inventory.xlsx](../docs/design/l2_seven_artifact_inventory.xlsx).

## L1 / L2 / L3 placement

| Intent | Layer | Directory |
|--------|-------|-----------|
| Model-level total-time / Top-N precision baseline | L1 precision | `tests/benchmark/models/` |
| Model-level Theory↔Runtime diagnostics | L1 structure | `tests/regression/model_diagnostics/e2e/` |
| Any real `model_id`, `ModelRunner`, model build/forward/compile, or multimodal pipeline | L1 capability | `tests/regression/<component>/model/` |
| No-model component integration with one canonical synthetic build artifact | L2 | `tests/regression/<component>/module/` |
| Shipped `op_mapping.yaml` integrity, real-DB numeric anchors | operator DB guard | `tests/benchmark/ops/perf_database/` |
| Pure function, Pass, parser, exception, synthetic CSV | L3 | `tests/regression/<component>/unit/` |
| Cheap PR guard for a nightly L2 path | smoke | `tests/smoke/` |

Do **not** put schema / parse / decompose / `tmp_path` unit tests under `tests/benchmark/ops/`. L2 must not import or invoke `model_runner` / `model_builder`, construct `TransformerModel` from a real `model_id`, capture a real model, or repeat an L1 assertion. A synthetic layer or minimal `torch.nn.Module` is allowed; a registered model tree is not.

### L1 / L2 / L3 routing tree

For every new case:

1. Does it use a real `model_id`, model config tree, `ModelRunner`, `build_model`, model forward/compile, or real diagnostics capture? → **L1 model-level**.
2. Otherwise, what single artifact does its canonical fixture build? Route to exactly one L2 module below.
3. If it only calls a pure function/class, parses or validates arguments, checks an exception/exit code, or needs no shared build artifact → **L3**.

Model identity belongs to L1; mechanism belongs to L2. One L1 family build should carry numeric, structural, and capability assertions together where their scenario and branches match. L2 asserts only mechanism invariants and never a result already asserted by L1.

| L2 module | Unique fixture artifact | Canonical helper target |
|---|---|---|
| Quantization components | `QuantLinear` / single-layer Attention + synthetic weights | `tests/helpers/quant_component_builder.py` |
| Parallel / communication components | single-layer `ParallelLinear` / Fake MoE / comm group | `tests/helpers/parallel_component_builder.py` |
| Graph transform / compile components | minimal synthetic FX graph | `tests/helpers/fx_graph_builder.py` |
| Runtime / performance components | synthetic event stream + breakdown inputs | `tests/helpers/runtime_event_builder.py` |
| Diagnostics application components | synthetic Artifact | `tests/helpers/diagnostics_artifact_builder.py` |
| Replay orchestration | replay request + fake adapter/subprocess | `tests/helpers/replay_case_builder.py` |
| Microbench generation | normalized op case + expected command/script | `tests/helpers/microbench_case_builder.py` |

“CLI” is not an L2 module. Route a CLI case by the artifact it tests: replay orchestration, microbench generation, or another mechanism above. Pure parser/default/validation/exit-code cases are L3.

The current tree is in migration: legacy `module/` files still contain real-model and pure-unit cases. The L2 static import/literal guard becomes blocking only after those cases move to L1/L3; enabling it before migration would make the existing suite fail by construction. `build_test_map.detect_redundant_cases` already implements the Jaccard calculation and has unit tests, so no new detector is needed. Its test-map-sync/nightly/Feishu wiring is still pending; do not describe redundancy warnings as currently reported.

**This wave does not relocate** `optix/`, `serving_cast/`, or `web_ui/`. `model_diagnostics/` keeps its domain tree. `scripts/` stays a mirror of `scripts/helpers/`. Do not add nested `conftest.py` under `model/`, `unit/`, or `module/` (a conftest change triggers a full CI suite). Import sibling helpers from `tests.regression.tensor_cast.test_common` / `conftest`, not package-relative `.` after a move.

### Merge / delete gate

`over_covered_symbol` and Jaccard alerts are review candidates only. Delete or merge a case only when:

1. The kept case covers the same product symbol **and** the same branches.
2. Every assertion class of the removed case (success, failure, exception, boundary, shape, dtype, timing) is preserved.
3. Fixture / model / parallel / quant differences have no independent value.
4. No product symbol loses its sole `test_map` watcher.
5. Affected L1/L2/L3 tests and smoke pass.

File size, similar names, or Jaccard = 1 over a single entry function is **not** enough to delete.

## Marker Semantics

| Marker | Where runs |
|--------|-----------|
| `nightly` | Full smoke/regression; excluded from mapped/guard wave, but **changed-test wave runs them** unless exempt. |
| `npu` | Excluded everywhere. |
| `network` | Only nightly Phase 2c. |

## Model Configs: Offline by Default

- Local fixtures under `tests/assets/model_config/<name>/` (`config.json`, optional `configuration_*.py`/`modeling_*.py`, and for VL models `preprocessor_config.json`). Tests load offline, no marker.
- Remote loading (live Hub) under `@pytest.mark.network` → nightly Phase 2c only.
- Vendoring: run `scripts/prefetch_model_configs.py` (uses `tensor_cast.model_hub` config-only snapshot, including `preprocessor_config.json` when present); copy needed files to `tests/assets/model_config/<name>/`, register Hub ids in `tests/helpers/model_assets.py` for offline preprocessor lookup.

## CI Gate Policy (`tests/.ci/gate_policy.yaml`)

### `roots`

Authoritative product source prefixes (each ends with `/`) — SSOT for:

- Diff classification (product vs test vs config)
- Coverage `--cov` package names
- `test_map` key validation
- Nightly `collect_from_coverage` scope

To add a product tree, append a `roots` entry — no Python constant duplication.

### Sections

| Section | Purpose |
|---------|---------|
| `tests` | Include/exclude patterns for gate test paths under `tests/` |
| `configs` | Patterns that trigger full-suite pytest: root `pyproject.toml`, `requirements.txt`, `uv.lock`, `tests/**/conftest.py`. Changes to `gate_policy.yaml` itself do **not** trigger full suite, only validation. |
| `exemptions.sources` | Waive coverage mapping for product symbols (`path::symbol`). Symbol is the **full canonical AST name** (e.g. `fn`, `_`, `_@torch.ops.foo.bar`, `Bar::method`, `Bar::method@classmethod`, `Foo::%`). Requires `reason`, `applicant`, `approver`, `deadline`. |
| `exemptions.tests` | Waive pytest nodes (`tests/...::test_func`). Must be concrete function/method id — no class-only, no parametrized brackets. If all nodes in a changed test file are exempt, file skipped. Selected-test failure outputs YAML hint; full-suite failure does not. |

### Canonical symbols

- `def _` without decorator → symbol `_`
- **Functions and methods** with decorators → `{name}@{decorator_suffix}` (multi-decorator: `@` joined; string literals double-quoted via stable unparse). Examples: `foo@deco`, `_@deco("a")`, `Foo::run@staticmethod`
- **Classes** use bare class name in spans; class-level decorators (e.g. `@dataclass`) are **not** mangled into a separate `Foo@{suffix}` gate or `test_map` key — class names rarely collide in one scope
- Class decorator lines and class-body non-method code → `Foo::%` in line mapping, `test_map`, and gate
- Class methods do **not** inherit class decorator mangling (`Foo::run@staticmethod`, not `Foo@dataclass::run`)
- Duplicate **function** names in one scope: **identical mangled symbol** repeats → last-wins for coverage mapping plus non-blocking CI PR shadow comment; **different mangled** symbols (e.g. two `def _()` with distinct decorators) coexist
- Exemption keys use the first `::` to split path from symbol; the symbol part must match the canonical name **exactly**

### Coverage omit

SSOT: `pyproject.toml` `[tool.coverage.run] omit`. Gate and nightly `test_map` skip matched product sources. Paths matched by omit **cannot** appear in `exemptions.sources`.

### Exemption drift (blocking)

If a PR deletes or renames a product/test file referenced by `exemptions`, ci_gate **blocks** until `gate_policy.yaml` is updated. Rename pairs must use the new path in exemption keys.

Example:

```yaml
exemptions:
  tests:
    - symbols:
        - tests/regression/cli/test_run.py::test_run
      reason: "Fixture unavailable on PR runners"
      applicant: alice
      approver: fangkai
      deadline: 2026-12-31
      ticket: "issue-123"
```

### Coverage fallback (ci_gate post-run)

Named symbols (new or modified) pass when `test_map` has an entry or strict same-PR coverage applies (`require_test_context=True`: only `tests/...::test_xxx` contexts count). Body symbols (`%` or `*::%`, suffix rule) pass when `test_map` has an entry or relaxed coverage applies (`require_test_context=False`: import-time/conftest hits count).

Coverage.py records multi-line statements on the **statement start line** only. Gate fallback therefore queries the original diff lines **union** their Coverage-measurable starts (innermost AST stmt / `ExceptHandler` / `match_case` / decorator expr `lineno`). Continuation diffs in parenthesized imports, multi-line list/dict/tuple literals, backslash continuations, and multi-line decorators still trigger the gate, but match coverage on the statement head. Same-statement short-circuit subexpressions share Coverage's statement granularity (no finer model). Keyword-only `else:` / `finally:` lines have no AST node and remap to the enclosing compound statement (intentional AST limit).

Modified definitions use three independent branches (all applicable branches must pass; body strict wins over proxy):

| diff touches | import (`%` / `Class::%`, relaxed) | body coverage |
|--------------|-------------------------------------|---------------|
| decorator only (function/method) | yes | body proxy on mangled symbol (`foo@deco`, `Foo::run@deco`) |
| decorator only (class) | yes (`Class::%`) | body proxy on `Class::%` |
| def header only | no | body proxy |
| body only | no | strict on changed body lines |
| decorator + body | yes | strict (no proxy) |
| def + body | no | strict (no proxy) |
| all three | yes | strict (no proxy) |

Decorator lines on functions attribute to `%` / `Class::%` for line mapping. Function and method mangled symbols (`foo@deco`, `Foo::run@staticmethod`) drive `test_map` and touched-definition checks. Class-level decorator changes gate via `Class::%` only.

Modified body symbols without a body watcher fall back to `source_watchers` on the same file when scheduling tests. Nightly phase1 and sync maintain external map; ci_gate reads only.

### Approval

`gate_policy.yaml` changes require approver from `tests/.ci/approvers.yaml`.

## Shared Test Helpers (`tests/helpers/`)

| Module | Public API | Role |
|--------|-----------|------|
| `model_cache.py` | `get_hf_config(model_id)` (deepcopy), `get_built_model(user_config)` (session cache), `user_config_build_cache_key` | Session-scoped cache for HF configs & built models |
| `model_assets.py` | `vendored_preprocessor_config_path(model_id)` | Offline `preprocessor_config.json` under `tests/assets/model_config/` |
| `model_builder.py` | `make_user_input_config`, `build_or_get_cached_model` | Build `UserInputConfig`; build-once-per-key |
| `config_factory.py` | `build_case_matrix`, `build_latency_thresholds` | Cartesian parametrize; shared latency thresholds |
| `op_registry.py` | `build_op_registry(cfg_registry)` | Lightweight per-model op registry |
| `assert_utils.py` | `assert_tensor_close`, `assert_latency_within` | Tensor closeness & latency assertions |
| `cli_runner.py` | `run_module_main`, `run_cli_main`, `CliResult` | Run CLI main **in-process** — coverage/`test_map` sees real path |
| `fake_subprocess.py` | `FakeCompleted(returncode, stdout, stderr)` | Minimal `subprocess.CompletedProcess` stand-in |

Self-tests: `tests/helpers/tests/`.

## `conftest.py` Rules

| Rule | Reason |
|------|--------|
| **Never** assign `sys.modules["tensor_cast"]` in any conftest | Replaces real modules → pickle failures across workers |
| Use fixture-scoped `monkeypatch` / `@patch` for test isolation | Scope stays local |
| `pytest_plugins = (...)` only in root `tests/conftest.py` | Subdirectory registration invalid; root shares fixtures across smoke/regression |
| Subdirectory conftests: directory-local fixtures only | No global import hacks |
| `gate_policy.yaml` `configs` matches `tests/**/conftest.py` → CI full suite | Ensures conftest changes are tested |
| Guard: `tests/smoke/test_conftest_hygiene.py` validates `tensor_cast.__spec__` intact | Prevention |

## Adding Test Cases

### 1. Choose directory

| Intent | Directory |
|--------|-----------|
| Quick validation, PR guard | `tests/smoke/` |
| Any real model build/forward/compile or model-backed CLI | `tests/regression/<component>/model/` (L1 capability) |
| Model-level Theory↔Runtime comparison | `tests/regression/model_diagnostics/e2e/` (L1 structure) |
| Model-level precision / performance baseline | `tests/benchmark/models/` (L1 precision) |
| No-model integration that builds one canonical synthetic artifact | `tests/regression/<component>/module/` (L2) |
| Operator DB integrity / real numeric anchors | `tests/benchmark/ops/perf_database/` |
| Pure function / parser / validation / exit-code tests | `tests/regression/<component>/unit/` (L3) |

No layer markers (`smoke`, `regression`, `benchmark`). Only `@pytest.mark.nightly` or `@pytest.mark.npu` when applicable.

### 2. Reuse helpers

| Need | Module | Key API |
|------|--------|---------|
| Build `UserInputConfig` | `tests/helpers/model_builder.py` | `make_user_input_config(model_id=..., ...)` |
| Build/cache model | `tests/helpers/model_cache.py` | `get_built_model(user_config)` or `build_or_get_cached_model(user_config, cache)` |
| Get HF config | `tests/helpers/model_cache.py` | `get_hf_config(model_id)` |
| Assert tensor/latency | `tests/helpers/assert_utils.py` | `assert_tensor_close(...)`, `assert_latency_within(...)` |
| Build op registry | `tests/helpers/op_registry.py` | `build_op_registry(cfg_registry)` |
| Run CLI in-process | `tests/helpers/cli_runner.py` | `run_module_main(module_name, argv)` |
| Stub subprocess | `tests/helpers/fake_subprocess.py` | `FakeCompleted(returncode, stdout, stderr)` |

### 3. Use session fixtures for L1 model-level regression

```python
from tests.helpers.model_builder import make_user_input_config
from tests.regression.tensor_cast.conftest import get_session_model

def test_my_feature():
    user_config = make_user_input_config(model_id="my-model-id")
    model = get_session_model(user_config)  # cached across session
```

`get_session_model` / `get_session_hf_config` delegate to `tests.helpers.model_cache`, shared across fixtures and `unittest.TestCase`. Do not use these model fixtures in L2.

### 4. Add benchmark case

1. Create JSON under `tests/benchmark/models/cases/` for a model-level precision scenario. Operator **unit** tests do not belong in `ops/perf_database/`.
2. Set `baseline_time_s` (0 → auto-baseline on first run) and `tolerance`.
3. `TestModelRegression` loads all JSON cases automatically.

### 5. Verify locally

```bash
python build.py test --suite smoke
python build.py test --suite regression
python build.py test --suite benchmark

# default local full suite
python build.py test

# CI gate (auto-downloads test_map when unset)
python build.py test --suite ci_gate

# check collection scope
PYTHONPATH=. python -m pytest tests/smoke/ tests/regression/ \
  -m "not npu and not nightly and not network" --collect-only -q
```

### Checklist

- [ ] Correct directory and L1/L2/L3 intent; no layer markers except `nightly`/`npu`
- [ ] Real `model_id` / model build / `ModelRunner` appears only in L1 model-level tests
- [ ] L2 builds exactly one of the seven canonical synthetic artifacts; CLI cases are routed by tested artifact
- [ ] Pure parser / validation / exception / exit-code cases are L3
- [ ] Shared canonical builder used (no copy-paste fixture construction)
- [ ] Session model fixtures are used only in L1 model-level regression
- [ ] If `@pytest.mark.nightly`, smoke guard exists under `tests/smoke/`
- [ ] No `sys.modules` / global mocks in conftest; `pytest_plugins` only in root
- [ ] New product symbols covered or in `exemptions.sources` / `exemptions.tests`
- [ ] Exemption keys use existing `path::symbol` pairs; omit-covered paths are not exempted
- [ ] Local smoke + regression pass before push

## Merge Checklist

- [ ] Test in correct directory; only necessary markers
- [ ] New product symbols covered or exempted in `gate_policy.yaml` with valid canonical symbols
- [ ] Rename/delete PRs update stale `exemptions` entries
- [ ] Local smoke + regression pass
- [ ] Nightly impact considered
