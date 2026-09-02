---
name: msmodeling-test-case-generator
description: Generate test cases for msmodeling under tests/smoke/, tests/regression/, or tests/benchmark/ from a feature description, model ID, or source symbol.
version: 1.0.0
---

# msmodeling Test Case Generator

Generate structurally correct test cases conforming to the three-layer test framework. For framework details, execution model, env vars → [scripts/README.md](../scripts/README.md). For test layout, markers, helpers, gate policy, conftest rules → [tests/README.md](README.md).

## When to Use

- Add new test case (feature, model, or source symbol)
- Extend existing test module
- Create smoke guard for nightly-marked test
- Add benchmark precision/performance baseline

## When NOT to Use

- Modify production code (`cli/`, `serving_cast/`, `tensor_cast/`, `web_ui/`, `scripts/helpers/`)
- Change `pyproject.toml`, `conftest.py`, or shell scripts
- Debug failing tests
- Generate mock data, model weights, or asset files
- Generate tests for third-party libraries or external dependencies

## Required Inputs

Collect from user. Ask if missing.

| Input | Required | If missing |
|-------|----------|------------|
| **What to test** — feature name, model ID, or source symbol path | Yes | Ask |
| **Test intent** — quick path check, functional verification, or precision/perf baseline | Yes | Infer from description, confirm |
| **Model ID** (e.g. `Qwen/Qwen3-30B-A13B`) | Conditional (compilation/inference) | Ask; suggest `tests/assets/model_config/` |
| **Directory preference** | No | Infer from intent (see Decision Tree) |
| **Existing test file to extend** | No | Create new file unless user specifies |

## Output

Exactly one of:

1. **Python test file** (smoke/regression) — path under `tests/smoke/` or `tests/regression/`; docstring; imports from `tests/helpers/` only; `unittest.TestCase` or standalone `test_*`; `@pytest.mark.nightly` only if `do_compile=True` and >300s; `@pytest.mark.npu` only if NPU required.
2. **Benchmark JSON** — path under `tests/benchmark/models/cases/` or `tests/benchmark/ops/perf_database/`; fields: `name`, `description`, `user_input`, `baseline_time_s` (0 = no baseline yet), `tolerance`.
3. **Smoke guard companion** — when generating `@pytest.mark.nightly`, also generate a smoke test exercising the same path with `num_hidden_layers_override=1` + `do_compile=False`.

## Framework Conventions

### Layering by Directory (NOT markers)

| Directory | When |
|-----------|------|
| `tests/smoke/` | Quick path validation, PR guard, <10s (not L1/L2/L3) |
| `tests/benchmark/models/cases/` | L1 model-level precision: JSON total-time / Top-N |
| `tests/regression/model_diagnostics/e2e/` | L1 model-level Theory↔Runtime |
| `tests/regression/<component>/model/` | L1 model-level capability: any real model build/forward/compile |
| `tests/regression/<component>/module/` | L2 no-model component integration; exactly one canonical synthetic artifact |
| `tests/regression/<component>/unit/` | L3 unit invariant |
| `tests/benchmark/ops/perf_database/` | Operator DB integrity / real numeric anchors only — not L1/L2 |

Only two markers: `@pytest.mark.nightly` (>300s compile), `@pytest.mark.npu` (NPU hardware). Never add layer markers.

### Decision Tree

1. Real `model_id`, model config tree, `ModelRunner`, model build/forward/compile, or real diagnostics capture? → L1 under the matching model-level directory. Combine numeric, structural, and capability assertions in one family build when the scenario and branches match.
2. Otherwise, does the case build one shared synthetic artifact? → L2 and route by artifact: quant component, parallel/comm component, minimal FX graph, synthetic Runtime event stream, synthetic diagnostics Artifact, replay request, or microbench case.
3. Pure function/class, parser, validation, exception, default, or exit code? → L3.
4. Quick path check? → `tests/smoke/`. Smoke is not a responsibility layer.

“CLI” is not a module. Replay orchestration and microbench generation are separate L2 modules; CLI parameter surfaces are normally L3. L2 must not import/invoke model runner or model builder, construct a registered `TransformerModel`, or use a real model-id literal.

### Shared Helpers

Always import from `tests/helpers/` (no copy-paste). Full API → [tests/README.md](./README.md).

Core: `config_factory.py` (build config), `model_builder.py` (L1 model only), `assert_utils.py` (assertions), `op_registry.py` (op registry), `fake_subprocess.py` (subprocess stubs).

Migration target: each L2 module owns one canonical builder under `tests/helpers/`: `quant_component_builder.py`, `parallel_component_builder.py`, `fx_graph_builder.py`, `runtime_event_builder.py`, `diagnostics_artifact_builder.py`, `replay_case_builder.py`, or `microbench_case_builder.py`. Do not create a second builder for the same artifact.

### Session Fixtures (L1 Model Regression Only)

```python
from tests.regression.tensor_cast.conftest import get_session_model, get_session_hf_config
```

Use these only in L1 model-level tests — never in L2. L2 uses one of the seven synthetic builders above.

### conftest Hygiene

| Do | Do Not |
|----|--------|
| Fixture-scoped `monkeypatch` in test file | `sys.modules["tensor_cast"] = MagicMock()` at conftest import |
| Real `torch`/`tensor_cast` (project deps) | Mock product packages globally in conftest |
| Directory-local fixtures only in subdirectory conftest | `pytest_plugins` in subdirectory (root only) |

Full rules → [tests/README.md](README.md#conftestpy-rules).

## Templates

### Smoke

```python
"""Smoke test for <feature>."""

from tests.helpers.config_factory import create_user_config
from tests.helpers.model_builder import build_transformer_model


def test_<feature>_smoke():
    user_config = create_user_config("<model_id>", num_hidden_layers_override=1, do_compile=False)
    model = build_transformer_model(user_config)
    assert model is not None
```

### L1 Model Capability Regression

```python
"""Regression test for <feature>."""

import unittest

from tests.helpers.config_factory import create_user_config
from tests.helpers.assert_utils import assert_model_metrics_valid
from tests.regression.tensor_cast.conftest import get_session_model


class Test<Feature>(unittest.TestCase):
    def test_<scenario>(self):
        user_config = create_user_config("<model_id>", do_compile=False)
        model = get_session_model(user_config)
        result = model.run_inference(...)
        assert_model_metrics_valid(result, "test_<scenario>")
```

### L2 No-Model Component Integration

```python
"""No-model integration test for <mechanism>."""

from tests.helpers.<canonical_builder> import build_<artifact>


def test_<mechanism>_<invariant>():
    artifact = build_<artifact>()
    result = artifact.run_synthetic_case()
    assert result.<mechanism_invariant>
```

The builder must not accept a real `model_id` or call `ModelRunner`, `build_model`, `get_built_model`, `get_cached_build_model`, or a registered `TransformerModel`.

### Nightly Regression

```python
"""Nightly regression test for <feature> (full compile)."""

import unittest

import pytest

from tests.helpers.config_factory import create_user_config
from tests.helpers.assert_utils import assert_model_metrics_valid
from tests.regression.tensor_cast.conftest import get_session_model


@pytest.mark.nightly
class Test<Feature>Nightly(unittest.TestCase):
    def test_<scenario>_nightly(self):
        user_config = create_user_config("<model_id>", do_compile=True)
        model = get_session_model(user_config)
        result = model.run_inference(...)
        assert_model_metrics_valid(result, "test_<scenario>_nightly")
```

### Benchmark JSON

```json
{
  "name": "<case_name>",
  "description": "<what this case validates>",
  "user_input": {
    "model_id": "<model_id>",
    "do_compile": true
  },
  "baseline_time_s": 0,
  "tolerance": 0.20
}
```

## Boundaries

- Do not modify production source code
- Do not add layer markers (`smoke`, `regression`, `benchmark`)
- Do not copy-paste builder or assertion logic — import from `tests/helpers/`
- Do not call `build_model()` directly in L1 regression — use session fixtures
- Do not use model builders, model runners, registered model trees, or real model-id literals in L2
- Do not classify “CLI” as a module; route by replay, microbench, Runtime, or L3 parameter surface
- Do not add `@pytest.mark.nightly` for tests <300s
- Do not add `@pytest.mark.npu` unless NPU is truly required
- Do not create new helper modules without checking existing ones
- Do not hardcode model weights or file paths
- Do not test third-party library internals
- Do not skip the smoke guard when generating a nightly test
- Do not generate conftest code that mutates `sys.modules` for product packages
- Do not suggest `pytest_plugins` in subdirectory conftest files

## CI Gate Policy (Summary)

Full details → [tests/README.md](./README.md).

- **Prefer real coverage** — regression tests should execute changed symbols.
- **ci_gate** (`run_ci_gate.sh`): PR-only, read-only `test_map`. Hard block → changed-test wave (no `-m`) + mapped/guard wave (`-m "not npu and not nightly and not network"`). Config change → full suite. Product/test file change → `--cov`.
- **`gate_policy.yaml`**: `roots` (SSOT for product prefixes), `tests` (include/exclude), `configs` (full-suite triggers: `pyproject.toml`, `requirements.txt`, `uv.lock`, `tests/**/conftest.py`), `exemptions.sources` (product symbol waivers: `path::symbol`), `exemptions.tests` (pytest node waivers: `tests/...::test_func`). Both exemptions require `reason`, `applicant`, `approver`, `deadline`. Changing `gate_policy.yaml` does **not** trigger full suite.
- **Coverage omit**: `pyproject.toml` `[tool.coverage.run] omit` (SSOT, not gate_policy).
- **Coverage fallback**: post-run `.coverage` test-node hits can clear unmapped symbols. Decorator lines map to `%` / `Class::%`. **Functions/methods** use mangled symbols in `test_map` (`foo@deco`, `Foo::run@staticmethod`); **classes** gate class-decorator edits via `Class::%` only (no `Class@dataclass` key). Modified defs use three branches via `gate_modified_source` only. See [tests/README.md](./README.md) § Coverage fallback.

Example test exemption:

```yaml
exemptions:
  tests:
    - symbols:
        - tests/regression/cli/test_run.py::test_run
      reason: "Blocked on upstream fixture; tracked in issue-123"
      applicant: alice
      approver: fangkai
      deadline: 2026-12-31
      ticket: "issue-123"
```

## Checklist

- [ ] Correct directory; no layer markers (only `nightly`/`npu` when applicable)
- [ ] Shared helpers used (no copy-paste)
- [ ] Real model identity/build/forward/compile appears only in L1
- [ ] L2 uses exactly one canonical synthetic artifact builder
- [ ] Pure parser/validation/exception/exit-code tests are L3
- [ ] Session fixtures for model construction are used only in L1
- [ ] If `@pytest.mark.nightly`, smoke guard co-generated
- [ ] No `sys.modules` mocks in conftest
- [ ] New product symbols covered or in `exemptions.sources`/`exemptions.tests`
- [ ] Local smoke + regression pass before push
