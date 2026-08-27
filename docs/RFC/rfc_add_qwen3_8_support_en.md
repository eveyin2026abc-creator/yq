# RFC: Qwen3.8 Model Adaptation Support

## Metadata

| Item | Content              |
| :--- |:---------------------|
| **Status** | Implemented          |
| **Author** | msmodeling Contributor |
| **Created** | 2026-08-14          |
| **Related Links** | Qwen3.8-2.4T-A95B open weights (released 2026-08-12) |

---

## 1. Overview

This proposal adds support for the Qwen3.8 model family to the tensor_cast framework, enabling performance simulation and bottleneck localization without physical hardware.

Qwen3.8 was open-sourced by Alibaba on 2026-08-12, currently shipping `Qwen/Qwen3.8-2.4T-A95B` (2.4T total params, 95B activated). The model inherits the Qwen3.5 architecture: a 92-layer structure mixing Gated DeltaNet linear attention with Gated Attention at a 3:1 ratio, a 512-expert MoE (10 routed + 1 shared expert per token), and multi-token prediction (MTP). Because its `config.json` keeps `architectures = ["Qwen3_5MoeForCausalLM"]` and `model_type = qwen3_5_moe_text`, the model loads directly via the existing Qwen3.5 modeling code in transformers — no upstream code vendoring is required.

## 2. Detailed Design

### 2.1 Implementation

#### 2.1.1 Strategy: Reuse Qwen3.5 Code

Key fields from the Qwen3.8-Max `config.json`:

| Field | Value | Adaptation Implication |
|:---|:---|:---|
| `architectures` | `["Qwen3_5MoeForCausalLM"]` | transformers loads classes from `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`; no new modeling code needed |
| `model_type` | `qwen3_5_moe_text` | New profile key (existing registry only has `qwen3_5` / `qwen3_5_moe`) |
| `num_experts` | 512 | Flat top-level config; read via a str path |
| `num_experts_per_tok` | 10 | Standard MoE routing |
| `shared_expert_intermediate_size` | 2048 | Shared expert present |
| `num_hidden_layers` | 92 | 3:1 linear:full attention mix |
| `mtp_num_hidden_layers` | 1 | MTP present |
| `vision_config` | absent | Text-only model; VL config must not be populated |

Since `architectures` reuses the Qwen3.5 class name, `patch_method` (linear-attention routing, meta-mask fix, VL placeholder relaxation, TP sharding) is reused as-is from `patch_method_for_qwen3_5`.

#### 2.1.2 New Profile Registration

A new profile is registered in `tensor_cast/transformers/builtin_model/qwen3_8.py`; `builtin_model/__init__.py` auto-loads it via `os.listdir` + `importlib.import_module`.

Registered fields:

```python
ModelProfile(
    model_type="qwen3_5_moe_text",
    moe_module_name="Qwen3_5MoeSparseMoeBlock",
    moe_gate_returns_raw_logits=False,
    moe_num_experts_key="num_experts",  # flat config: str, not list
    moe_field_names_override=MoEFieldNames(
        shared_experts="shared_expert",        # singular naming
        shared_experts_gate="shared_expert_gate",
    ),
    mtp_block_module_name="Qwen3_5MoeDecoderLayer",
    model_family="qwen3_8",
    patch_method=patch_method_for_qwen3_5,  # imported reference; original untouched
)
```

Differences vs. the Qwen3.5-MoE profile:

| Field | Qwen3.5-MoE | Qwen3.8 | Reason |
|:---|:---|:---|:---|
| `model_type` | `qwen3_5_moe` | `qwen3_5_moe_text` | Actual value in Qwen3.8 config |
| `moe_num_experts_key` | `["text_config", "num_experts"]` | `"num_experts"` | Qwen3.5 is VL with nested config; Qwen3.8 is flat text-only config |
| visual config | `resolve_visual_config({})` | not passed | Qwen3.8 has no `vision_config`; passing default VL paths raises `AttributeError` |
| `model_family` | `qwen3_5` | `qwen3_5` | Reuses the Qwen3.5 TP plan gate in `transformations.py` (`model_family == "qwen3_5"`) so Gated DeltaNet linear attention sharding stays compatible with the patch when TP>1 |

#### 2.1.3 Two Issues Fixed During Adaptation

**Issue 1: Patch not applied → torch.compile error**

Without the profile registered, `get_model_profile("qwen3_5_moe_text")` returns None and `patch_model` silently skips (`if profile and profile.patch_method` is False). The model falls back to the original `_update_linear_attn_mask`, which hits the data-dependent branch `if cache_position[0] > 0`, causing torch.compile to raise `Unsupported: Data-dependent branching`. Once the profile is registered, the patch replaces `_update_linear_attn_mask` with a compile-friendly version and the error is resolved.

**Issue 2: Spurious VL defaults → `AttributeError`**

Initially, following the Qwen3.5-VL pattern, `**resolve_visual_config({})` was passed. That helper fills `visual_layers_module_path="visual.blocks"` from `COMMON_VISUAL_CONFIG`. `get_visual_layers` then calls `operator.attrgetter("visual.blocks")(model)`, but Qwen3.8 is text-only (`ForCausalLM`, no `visual` attribute), raising `AttributeError`. Fix: omit the visual config argument entirely, leaving `visual_layers_module_path` as the default None — consistent with other text-only models like `qwen3_moe.py` and `deepseek_v3.py`.

### 2.2 Alternatives

**Alternative A: Vendor the official modeling code**

Following the `bailing_moe_hf/` and `mimo_v2_flash_hf/` pattern, place Qwen3.8's modeling code under `builtin_model/qwen3_8_hf/`. This is used when transformers does not support the target model. Since Qwen3.8 reuses the Qwen3.5 class name and transformers already loads it, vendoring is unnecessary and was not adopted.

**Alternative B: Split dense and MoE into two files**

Mirroring the `qwen3_5.py` + `qwen3_5_moe.py` two-file structure. Only the MoE variant of Qwen3.8 is currently open-sourced (the 27B dense version is not yet released), so a single file suffices. A dense profile can be appended to the same file once released.

### 2.3 Analysis

**Why this approach was chosen:**

1. **Zero code vendoring**: Qwen3.8 reuses Qwen3.5 modeling classes, avoiding a maintained upstream copy and any sync cost on transformers upgrades.
2. **Minimal change surface**: A single 20-line file is added; no existing code is modified. Controlled experiments confirm no impact on other models (see Section 4).
3. **Full patch reuse**: Linear-attention routing, meta-mask fix, etc. are identical to Qwen3.5. `patch_method_for_qwen3_5` is reused by reference without any modification.

## 3. Implementation Plan

### Completed

- [x] Performance simulation modeling for Qwen3.8-Max (2.4T-A95B)
- [x] Mixed linear + full attention (92 layers, 3:1) simulation
- [x] 512-expert MoE + shared expert simulation
- [x] MTP (multi-token prediction) simulation
- [x] W8A8 static quantization simulation
- [x] TP + EP parallel simulation (validated on 16 devices, TP2+EP8)

### Simulation Results

| Config | TPS/Device | Memory/Device | Bottleneck Breakdown |
|:---|:---|:---|:---|
| 2 devices TP2+EP1 (smoke) | 24.74 token/s | 1147 GB (over limit) | memory 98% |
| 16 devices TP2+EP8 | 62.39 token/s | 181 GB (still over) | compute 36.8% / memory 33% / comm 29.1% |

Note: 181 GB/device still exceeds the 64 GB budget. Real deployment requires larger EP (e.g., 64 devices EP=32) or stronger quantization (MXFP4).

### Future Work

- [ ] Adapt Qwen3.8-27B dense variant (append profile after official release; ~5 lines)
- [ ] Validate memory-feasible configs (64 devices EP=32 / MXFP4)
- [ ] On-device profiling comparison (provide `reports/qwen3_8/raw_insight.txt` and run the model-adaptation doctor flow)
- [ ] Add performance regression cases (`tests/benchmark/models/cases/qwen3_8-*.json`; operator baselines need refresh first)

## 4. Impact Assessment

### 4.1 Change Surface

Only `tensor_cast/transformers/builtin_model/qwen3_8.py` is added (untracked). `git diff` is empty; no tracked files are modified.

### 4.2 No Profile Conflicts

19 profiles register normally, including all 7 qwen-family profiles. Existing `qwen3_5` and `qwen3_5_moe` are unaffected.

### 4.3 Regression Test Control

A controlled experiment was run on 6 text-model cases (deepseek-v3.1, GLM-4.7, qwen3-30b-a3b ×2, qwen3-8B ×2) with and without `qwen3_8.py`. Results are identical: operator timings, failure reasons, and percentage deviations match item by item. All failures are pre-existing operator-baseline drift, unrelated to this change.

### 4.4 Adapter Automation Tests

`tests/regression/tensor_cast/test_adapter_automation.py` — 40 passed.
