# RFC: Qwen3.8 模型适配支持

## 元数据

| 项目 | 内容                 |
| :--- |:---------------------|
| **状态** | 已实现                |
| **作者** | msmodeling Contributor |
| **创建日期** | 2026-08-14          |
| **相关链接** | Qwen3.8-2.4T-A95B 开源权重（2026-08-12 发布） |

---

## 1. 概述

本提案旨在在 tensor_cast 框架中增加 Qwen3.8 系列模型的支持，使其能够在无需物理硬件的情况下完成性能仿真与瓶颈定位。

Qwen3.8 由阿里巴巴于 2026-08-12 开源，当前发布 `Qwen/Qwen3.8-2.4T-A95B`（总参 2.4T，激活 95B）。该模型在架构上继承 Qwen3.5，采用 Gated DeltaNet 线性注意力与 Gated Attention 按 3:1 混合的 92 层结构，并使用 512 专家 MoE（每 token 路由 10 + 1 共享专家）与多 token 预测（MTP）。因其 `config.json` 的 `architectures` 字段仍为 `Qwen3_5MoeForCausalLM`、`model_type` 为 `qwen3_5_moe_text`，可直接复用 Qwen3.5 的 modeling 代码，无需 vendor 上游实现。

## 2. 详细设计

### 2.1 实现方案

#### 2.1.1 适配策略：复用 Qwen3.5 代码

Qwen3.8-Max 的 `config.json` 关键字段如下：

| 字段 | 值 | 适配含义 |
|:---|:---|:---|
| `architectures` | `["Qwen3_5MoeForCausalLM"]` | transformers 加载 `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe` 的类，无需新增 modeling 代码 |
| `model_type` | `qwen3_5_moe_text` | 新增 profile 注册键（现有仅 `qwen3_5`/`qwen3_5_moe`，无此键） |
| `num_experts` | 512 | 顶层扁平 config，用 str 路径读取 |
| `num_experts_per_tok` | 10 | 标准 MoE 路由 |
| `shared_expert_intermediate_size` | 2048 | 含共享专家 |
| `num_hidden_layers` | 92 | 3:1 线性:全注意力混合 |
| `mtp_num_hidden_layers` | 1 | 含 MTP |
| `vision_config` | 无 | 纯文本模型，不应填充 VL 配置 |

因 `architectures` 复用 Qwen3.5 的类名，`patch_method`（线性注意力路由、meta mask 修复、VL placeholder 放宽、TP 切分）可零改动复用 `patch_method_for_qwen3_5`。

#### 2.1.2 新增 Profile 注册

在 `tensor_cast/transformers/builtin_model/qwen3_8.py` 新增 profile 注册，`builtin_model/__init__.py` 通过 `os.listdir` + `importlib.import_module` 自动加载。

注册字段：

```python
ModelProfile(
    model_type="qwen3_5_moe_text",
    moe_module_name="Qwen3_5MoeSparseMoeBlock",
    moe_gate_returns_raw_logits=False,
    moe_num_experts_key="num_experts",  # 扁平 config，用 str 而非 list
    moe_field_names_override=MoEFieldNames(
        shared_experts="shared_expert",        # 单数命名
        shared_experts_gate="shared_expert_gate",
    ),
    mtp_block_module_name="Qwen3_5MoeDecoderLayer",
    # 设为 "qwen3_5"（非 "qwen3_8"）以复用 transformations.py 中已有的 Qwen3.5
    # TP plan 门闩（model_family == "qwen3_5"），确保 TP>1 时 Gated DeltaNet
    # 线性注意力切分与 patch 行为一致。
    model_family="qwen3_5",
    patch_method=patch_method_for_qwen3_5,  # 引用复用，未修改原函数
)
```

与 Qwen3.5-MoE profile 的差异点：

| 字段 | Qwen3.5-MoE | Qwen3.8 | 差异原因 |
|:---|:---|:---|:---|
| `model_type` | `qwen3_5_moe` | `qwen3_5_moe_text` | Qwen3.8 config 的实际值 |
| `moe_num_experts_key` | `["text_config", "num_experts"]` | `"num_experts"` | Qwen3.5 为 VL 嵌套 config，Qwen3.8 为扁平纯文本 config |
| visual config | `resolve_visual_config({})` | 不传 | Qwen3.8 无 `vision_config`，传默认 VL 路径会导致 `AttributeError` |
| `model_family` | `qwen3_5` | `qwen3_5` | 复用 `transformations.py` 中 Qwen3.5 的 TP plan 门闩（`model_family == "qwen3_5"`），确保 TP>1 时 Gated DeltaNet 线性注意力切分与 patch 兼容 |

#### 2.1.3 适配过程修复的两个问题

**问题一：patch 未生效导致 torch.compile 报错**

未注册 profile 时，`get_model_profile("qwen3_5_moe_text")` 返回 None，`patch_model` 静默跳过（`if profile and profile.patch_method` 为 False）。模型走原始 `_update_linear_attn_mask`，命中 `if cache_position[0] > 0` 数据依赖分支，torch.compile 抛 `Unsupported: Data-dependent branching`。注册 profile 后 patch 生效，`_update_linear_attn_mask` 被替换为 compile-friendly 版本，问题解决。

**问题二：误传 VL 默认配置导致 `AttributeError`**

初期参照 Qwen3.5-VL 传 `**resolve_visual_config({})`，该函数会用 `COMMON_VISUAL_CONFIG` 填充 `visual_layers_module_path="visual.blocks"`。`get_visual_layers` 通过 `operator.attrgetter("visual.blocks")(model)` 访问，但 Qwen3.8 是纯文本模型（`ForCausalLM`，无 `visual` 属性），抛 `AttributeError`。修复方式：删除 visual config 传参，保持 `visual_layers_module_path` 为默认 None（与 `qwen3_moe.py`、`deepseek_v3.py` 等纯文本模型一致）。

### 2.2 替代方案

**方案 A：vendor 官方 modeling 代码**

仿照 `bailing_moe_hf/`、`mimo_v2_flash_hf/` 的做法，将 Qwen3.8 的 modeling 代码放入 `builtin_model/qwen3_8_hf/`。该方案在 transformers 未支持目标模型时使用。Qwen3.8 因 `architectures` 复用 Qwen3.5 类名，transformers 已可加载，无需 vendor，故未采用。

**方案 B：拆分 dense 与 MoE 为两个文件**

参照 `qwen3_5.py` + `qwen3_5_moe.py` 的双文件结构。当前 Qwen3.8 仅开源 MoE 版本（27B dense 尚未发布），单文件已足够；待 27B 发布后可在同文件追加 dense profile，无需拆分。

### 2.3 方案分析

**选择当前方案的原因：**

1. **零代码 vendor**：Qwen3.8 复用 Qwen3.5 的 modeling 类，无需维护上游代码副本，后续 transformers 升级时无同步成本。
2. **最小改动面**：仅新增一个 20 行文件，不修改任何现有代码，经对照验证不影响其他模型（见第 4 节）。
3. **patch 完全复用**：线性注意力路由、meta mask 修复等逻辑与 Qwen3.5 一致，`patch_method_for_qwen3_5` 以引用方式复用，未做任何改动。

## 3. 实施计划

### 已完成功能开发

- [x] Qwen3.8-Max（2.4T-A95B）性能仿真建模
- [x] 线性注意力 + 全注意力混合结构（92 层 3:1）仿真
- [x] 512 专家 MoE + 共享专家仿真
- [x] MTP（多 token 预测）仿真支持
- [x] W8A8 静态量化仿真
- [x] TP + EP 并行仿真（16 卡 TP2+EP8 验证通过）

### 仿真验证结果

| 配置 | TPS/Device | 显存/卡 | 瓶颈分布 |
|:---|:---|:---|:---|
| 2 卡 TP2+EP1（冒烟） | 24.74 token/s | 1147 GB（超限） | memory 98% |
| 16 卡 TP2+EP8 | 62.39 token/s | 181 GB（仍超限） | compute 36.8% / memory 33% / comm 29.1% |

注：181 GB/卡仍超 64 GB 显存，真实部署需更大 EP（如 64 卡 EP=32）或更高量化（MXFP4）。

### 后续优化

- [ ] Qwen3.8-27B dense 版本适配（待官方开源后追加 profile，预计 5 行代码）
- [ ] 显存优化配置验证（64 卡 EP=32 / MXFP4 量化）
- [ ] 真机 profiling 比对（补 `reports/qwen3_8/raw_insight.txt` 走 model-adaptation doctor 流程）
- [ ] 性能回归用例入库（`tests/benchmark/models/cases/qwen3_8-*.json`，需先刷新算子基线）

## 4. 影响评估

### 4.1 改动范围

仅新增 `tensor_cast/transformers/builtin_model/qwen3_8.py`（untracked），`git diff` 为空，零修改现有文件。

### 4.2 Profile 注册无冲突

共 19 个 profile 正常注册，7 个 qwen 系列 profile 全在，现有 `qwen3_5`、`qwen3_5_moe` 不受影响。

### 4.3 回归测试对照

对 6 个文本模型用例（deepseek-v3.1、GLM-4.7、qwen3-30b-a3b ×2、qwen3-8B ×2）做有无 `qwen3_8.py` 的对照实验，结果完全一致：算子耗时、失败原因、百分比偏差逐项相同。失败均为既有算子基线漂移（与本次改动无关）。

### 4.4 Adapter 自动化测试

`tests/regression/tensor_cast/test_adapter_automation.py` 40 项全部通过。
