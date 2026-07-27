# MindStudio Modeling 26.1.0 版本发布说明

## 1. 版本概述

MindStudio Modeling 26.1.0 是面向昇腾 AI 处理器的神经网络推理性能仿真与服务化部署寻优版本，主要服务于模型适配、性能预研、容量规划和大模型服务化调优场景。核心亮点如下：

- 提供模型推理性能仿真能力，可在无需真实硬件的情况下，基于设备画像预测模型在昇腾 AI 处理器上的理论性能。
- 支持 Prefill/Decode 分阶段仿真、Prefix Cache、MTP 投机解码、量化仿真、并行策略与 MoE 扩展等典型大模型推理场景。
- 提供服务化性能仿真与吞吐寻优能力，支持在 TTFT、TPOT 等 SLO 约束下搜索较优部署参数。
- 支持 optix 服务化实测寻优，可结合 vLLM、MindIE 等推理引擎在真实部署环境中自动调参。
- 支持 Web UI 可视化配置、结果展示与导出，便于完成仿真预测、实测验证和可视化分析闭环。

## 2. 配套关系

| 软件/硬件 | 版本要求 | 说明 |
| --- | --- | --- |
| 产品型号 | Atlas 350 加速卡；Atlas A3 训练系列产品/Atlas A3 推理系列产品；Atlas A2 训练系列产品/Atlas A2 推理系列产品；Atlas 200I/500 A2 推理产品；Atlas 推理系列产品；Atlas 训练系列产品 | 根据仓库文档中的产品支持情况整理 |
| 操作系统 | Linux；Windows | 仓库文档提及 Linux 与 Windows 使用场景，具体依赖需按安装指南配置 |
| 驱动版本 | 不依赖固定驱动版本 | 模型仿真能力可在无真实硬件驱动环境下运行 |
| 固件版本 | 不依赖固定固件版本 | 模型仿真能力可在无真实硬件固件环境下运行 |
| CANN 版本 | 不依赖固定 CANN 版本 | 模型仿真能力可在未安装 CANN 的环境下运行 |
| Python 版本 | 3.10 及以上 | 根据仓库 README 整理 |
| PyTorch 版本 | 建议使用 2.8 及更早版本 | 仓库 README 提示 Windows 环境下 PyTorch 2.10 可能运行异常 |
| transformers 版本 | 5.6.0 及以上，低于 5.8.0 | GLM5 系列模型依赖 5.6.0 及以上版本的返回值和 `indexer_types` 契约 |
| 推理引擎 | vLLM、MindIE 等 | 主要用于 optix 服务化实测寻优场景，具体版本需结合推理引擎官方配套关系确认 |
| 许可证 | MulanPSL2-style License | 根据仓库 README 与 LICENSE 信息整理 |

## 3. 新增特性

本版本新增特性围绕 Q2 Roadmap 的模型适配效率、推理特性建模、算子建模底座、易用性与结果分析、精度治理与工程稳定性五条主线展开。

| 序号 | 特性名称 | 特性描述 | 关联 Issue/PR |
| --- | --- | --- | --- |
| 1 | Qwen3.5 Dense/MoE 模型适配 | 新增 Qwen3.5 Dense、Qwen3.5 MoE 模型文本输入仿真支持，扩展 Qwen 系列重点模型覆盖范围，支撑 Q2 新模型适配目标。 | [!349](https://gitcode.com/Ascend/msmodeling/pull/349) |
| 2 | GLM-4 MoE 模型适配 | 新增 GLM-4 MoE 模型仿真支持，补齐 GLM 系列 MoE 推理性能评估能力，提升重点模型接入效率。 | [!217](https://gitcode.com/Ascend/msmodeling/pull/217) |
| 3 | DeepSeek-V4 模型适配 | msModeling 新增 DeepSeek-V4 模型支持，覆盖 Flash/Pro 端到端仿真建模，支持 sparse/compressed attention、KV cache 压缩与 MTP 等推理特性，补齐 DeepSeek 系列重点模型性能评估能力。 | [!166](https://gitcode.com/Ascend/msmodeling/merge_requests/166) |
| 4 | GLM-5.2 IndexShare 推理特性适配 | 新增 GLM5 专用 IndexShare 辅助逻辑，支持全量层执行 indexer、共享层复用上一全量层 top-k indices，并扩展 MTP `indexer_types` 支持。 | [!535](https://gitcode.com/Ascend/msmodeling/merge_requests/535)、[!509](https://gitcode.com/Ascend/msmodeling/merge_requests/509) |
| 5 | GLM-5/GLM-5.2 MTP 兼容性增强 | 增强 repetition、MTP、torch.compile 组合场景下的模型兼容性与进程池序列化能力，恢复 `throughput_optimizer` 并行搜索能力。 | [!563](https://gitcode.com/Ascend/msmodeling/merge_requests/563)、[!538](https://gitcode.com/Ascend/msmodeling/merge_requests/538) |
| 6 | throughput_optimizer 结果分析与 token budget 自适应 | `--max-batched-tokens` 支持自动模式，按 4 倍、2 倍、1 倍 input_length 顺序尝试 token budget，并在 Prefill OOM 时自动降级，降低 case 构造与调参成本。 | [!538](https://gitcode.com/Ascend/msmodeling/merge_requests/538)、[!563](https://gitcode.com/Ascend/msmodeling/merge_requests/563) |
| 7 | EvalScope benchmark 插件 | 新增 EvalScope 作为服务化寻优 benchmark 补充选项，增强模型类型覆盖、评估功能与生态集成能力，支撑服务化部署寻优和结果验证。 | [!515](https://gitcode.com/Ascend/msmodeling/merge_requests/515) |
| 8 | optix 优化器工程稳定性增强 | 新增结构化 loguru 日志、领域异常体系、Benchmark 启动前 fail-fast 校验和安全清理守卫，提升服务化实测寻优稳定性和故障可排查性。 | [!518](https://gitcode.com/Ascend/msmodeling/merge_requests/518) |
| 9 | 构建入口与依赖自举流程优化 | 新增根目录 `build.py` 统一构建/测试入口，去除 import-time 阶段对 pydantic 的强依赖，并在缺少 uv 时进行非交互自举，降低安装和测试门槛。 | [!485](https://gitcode.com/Ascend/msmodeling/merge_requests/485)、[!546](https://gitcode.com/Ascend/msmodeling/merge_requests/546) |
| 10 | CI Gate 与测试入口稳定性增强 | `build.py test` 在未设置 `MSMODELING_TEST_MAP_PATH` 时默认执行全量 `pytest tests`，设置后继续走 CI Gate 增量测试模式，提升回归防护能力。 | [!557](https://gitcode.com/Ascend/msmodeling/merge_requests/557) |
| 11 | pre-commit 密钥扫描流程增强 | 新增 `gitleaks-offline-scan` 本地离线 hook，在提交前对暂存文件进行敏感信息扫描，降低密钥误提交风险。 | [!541](https://gitcode.com/Ascend/msmodeling/merge_requests/541) |
| 12 | 文档体系与模型支持说明完善 | 修复文档失效链接、Markdown 格式问题和部分模型支持说明缺失问题，提升安装、运行、结果解读等高频使用场景的文档可用性。 | [!558](https://gitcode.com/Ascend/msmodeling/merge_requests/558)、[!559](https://gitcode.com/Ascend/msmodeling/merge_requests/559) |

## 4. 修复缺陷

| 序号 | 问题描述 | 影响范围 | 关联 Issue/PR |
| --- | --- | --- | --- |
| 1 | 修复 GLM-5/GLM-5.2 在同时启用 repetition、MTP、torch.compile 时的模型兼容与进程池序列化问题，恢复并行搜索能力。 | GLM-5/GLM-5.2 服务化寻优场景 | [!563](https://gitcode.com/Ascend/msmodeling/merge_requests/563) |
| 2 | 修复 GLM5 DSA 契约不匹配导致的运行时错误，通过提升 transformers 最低版本要求避免低版本返回值不兼容。 | GLM5 系列模型仿真场景 | [!509](https://gitcode.com/Ascend/msmodeling/merge_requests/509) |
| 3 | 修复文档失效链接、Markdown 格式问题和部分模型支持说明缺失问题，提升文档可读性与可维护性。 | 文档阅读与维护场景 | [!558](https://gitcode.com/Ascend/msmodeling/merge_requests/558)、[!559](https://gitcode.com/Ascend/msmodeling/merge_requests/559) |
| 4 | 重构 optix 优化器可靠性，新增结构化日志、领域异常、Benchmark 启动前 fail-fast 校验和安全清理守卫。 | optix 服务化实测寻优场景 | [!518](https://gitcode.com/Ascend/msmodeling/merge_requests/518) |

### 5. 致谢
感谢以下贡献者对本版本的贡献:
| 序号 | 贡献者 | 贡献内容 | 关联 PR |
| --- | --- | --- | --- |
| 1 | jia_ya_nan | GLM-5 MTP 兼容性修复 | [!563](https://gitcode.com/Ascend/msmodeling/merge_requests/563) |
| 2 | minghang_c | GLM-5.2 IndexShare 适配 | [!535](https://gitcode.com/Ascend/msmodeling/merge_requests/535) |
| 3 | liujiawang / AvadaKedavrua | optix 优化器重构 | [!518](https://gitcode.com/Ascend/msmodeling/merge_requests/518) |
| 4 | jhon-117 | sig-review AI 代码检视 skill | [!553](https://gitcode.com/Ascend/msmodeling/merge_requests/553) |
| 5 | liu977803265 | evalscope 插件合入 | [!515](https://gitcode.com/Ascend/msmodeling/merge_requests/515) |
| 6 | genius52 | MoE 字段配置重构 | [!128](https://gitcode.com/Ascend/msmodeling/merge_requests/128) |
| 7 | Secluded_Ocean | GLM5 A3 实测算子样本 | [!568](https://gitcode.com/Ascend/msmodeling/merge_requests/568) |
| 8 | Horacehxw | profiling 插值 phase1 实现 | [!262](https://gitcode.com/Ascend/msmodeling/merge_requests/262) |
| 9 | zhenyu_zhang | profiling wrapper 修复 | [!540](https://gitcode.com/Ascend/msmodeling/merge_requests/540) |
| 10 | elrond-g | kimi-K2.6 trust_remote_code 修复 | [!260](https://gitcode.com/Ascend/msmodeling/merge_requests/260) |
| 11 | Hudingyi | mla_sparse_attention DSA 算子 | [!363](https://gitcode.com/Ascend/msmodeling/merge_requests/363) |
| 12 | stormchasingg | mixed-batch 可变输入仿真 | [!438](https://gitcode.com/Ascend/msmodeling/merge_requests/438) |
| 13 | jgong5 | PyTorch 2.10 版本支持扩展 | [!64](https://gitcode.com/Ascend/msmodeling/merge_requests/64) |
| 14 | Abstrey | real-split pipeline parallel 模型构建 | [!367](https://gitcode.com/Ascend/msmodeling/merge_requests/367) |
| 15 | ChenHuiwen | DeepSeek V4 模型适配 | [!166](https://gitcode.com/Ascend/msmodeling/merge_requests/166) |
| 16 | cmh1056291129 | serving_cast 搜索方法效率优化 | [!199](https://gitcode.com/Ascend/msmodeling/merge_requests/199) |
| 17 | gcw_hasgjVbP | run_throughput_optimizer_cases 组合测试脚本 | [!247](https://gitcode.com/Ascend/msmodeling/merge_requests/247) |
| 18 | liu_jiaxu | Qwen3-VL resize 参数配置读取 | [!126](https://gitcode.com/Ascend/msmodeling/merge_requests/126) |
| 19 | sunguozhong | msmodeling Web UI 可视化界面 | [!161](https://gitcode.com/Ascend/msmodeling/merge_requests/161) |
| 20 | weixin_43368449 | Qwen3-VL ViT TP 并行与 MoE | [!46](https://gitcode.com/Ascend/msmodeling/merge_requests/46) |
| 21 | yikangLin | vLLM 服务化参数自动寻优双机插件 | [!403](https://gitcode.com/Ascend/msmodeling/merge_requests/403) |