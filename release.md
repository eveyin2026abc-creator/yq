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

| 序号 | 特性名称 | 特性描述 | 关联 Issue/PR |
| --- | --- | --- | --- |
| 1 | Qwen3.5 Dense/MoE 文本输入支持 | 新增 Qwen3.5 Dense、Qwen3.5 MoE 模型的文本输入仿真支持，扩展 Qwen 系列模型覆盖范围。 | [!349](https://gitcode.com/Ascend/msmodeling/pull/349) |
| 2 | GLM-4 MoE 模型支持 | 新增 GLM-4 MoE 模型仿真支持，补齐 GLM 系列 MoE 推理性能评估能力。 | [!217](https://gitcode.com/Ascend/msmodeling/pull/217) |
| 3 | GLM-5.2 IndexShare 适配 | 新增 GLM5 专用 IndexShare 辅助逻辑，支持全量层执行 indexer、共享层复用上一全量层 top-k indices，并扩展 MTP `indexer_types` 支持。 | [!535](https://gitcode.com/Ascend/msmodeling/merge_requests/535) |
| 4 | GLM-5/GLM-5.2 MTP 兼容性增强 | 修复 repetition、MTP、torch.compile 同时开启时的模型兼容与进程池序列化问题，恢复 `throughput_optimizer` 并行搜索能力。 | [!563](https://gitcode.com/Ascend/msmodeling/merge_requests/563) |
| 5 | throughput_optimizer 自适应 max-batched-tokens | `--max-batched-tokens` 支持自动模式，按 4 倍、2 倍、1 倍 input_length 顺序尝试 token budget，并在 Prefill OOM 时自动降级。 | [!538](https://gitcode.com/Ascend/msmodeling/merge_requests/538) |
| 6 | EvalScope benchmark 插件 | 新增 EvalScope 作为服务化寻优 benchmark 补充选项，增强模型类型覆盖、评估功能与生态集成能力。 | [!515](https://gitcode.com/Ascend/msmodeling/merge_requests/515) |
| 7 | optix 优化器可靠性增强 | 新增结构化 loguru 日志、领域异常体系、Benchmark 启动前 fail-fast 校验和安全清理守卫，提升服务化实测寻优稳定性。 | [!518](https://gitcode.com/Ascend/msmodeling/merge_requests/518) |

## 4. 变更说明

| 序号 | 变更内容 | 变更影响 | 关联 Issue/PR |
| --- | --- | --- | --- |
| 1 | `throughput_optimizer` 的 `--max-batched-tokens` 默认值调整 | 不兼容变更：默认值由固定 `8192` 调整为 `None` 自动模式。依赖旧默认值的脚本建议显式传参以保持原行为。 | [!538](https://gitcode.com/Ascend/msmodeling/merge_requests/538) |
| 2 | transformers 最低版本要求提升 | 不兼容变更：最低版本由 `>=5.3.0` 调整为 `>=5.6.0`。低于 5.6.0 的环境需升级后才能稳定运行 GLM5 系列模型。 | [!509](https://gitcode.com/Ascend/msmodeling/merge_requests/509) |
| 3 | `build.py test` 默认行为调整 | 未设置 `MSMODELING_TEST_MAP_PATH` 时，默认执行全量 `pytest tests`；设置后继续走 CI Gate 增量测试模式。依赖旧报错行为的 CI 脚本需同步适配。 | [!557](https://gitcode.com/Ascend/msmodeling/merge_requests/557) |
| 4 | 构建入口与依赖自举流程调整 | 新增根目录 `build.py` 统一构建/测试入口，去除 import-time 阶段对 pydantic 的强依赖，并在缺少 uv 时进行非交互自举。 | [!485](https://gitcode.com/Ascend/msmodeling/merge_requests/485)、[!546](https://gitcode.com/Ascend/msmodeling/merge_requests/546) |
| 5 | pre-commit 密钥扫描流程增强 | 新增 `gitleaks-offline-scan` 本地离线 hook，在提交前对暂存文件进行敏感信息扫描，降低密钥误提交风险。 | [!541](https://gitcode.com/Ascend/msmodeling/merge_requests/541) |

## 5. 修复缺陷

| 序号 | 问题描述 | 影响范围 | 关联 Issue/PR |
| --- | --- | --- | --- |
| 1 | 修复 GLM-5/GLM-5.2 在同时启用 repetition、MTP、torch.compile 时的模型兼容与进程池序列化问题，恢复并行搜索能力。 | GLM-5/GLM-5.2 服务化寻优场景 | [!563](https://gitcode.com/Ascend/msmodeling/merge_requests/563) |
| 2 | 修复 GLM5 DSA 契约不匹配导致的运行时错误，通过提升 transformers 最低版本要求避免低版本返回值不兼容。 | GLM5 系列模型仿真场景 | [!509](https://gitcode.com/Ascend/msmodeling/merge_requests/509) |
| 3 | 修复文档失效链接、Markdown 格式问题和部分模型支持说明缺失问题，提升文档可读性与可维护性。 | 文档阅读与维护场景 | [!558](https://gitcode.com/Ascend/msmodeling/merge_requests/558)、[!559](https://gitcode.com/Ascend/msmodeling/merge_requests/559) |
| 4 | 重构 optix 优化器可靠性，新增结构化日志、领域异常、Benchmark 启动前 fail-fast 校验和安全清理守卫。 | optix 服务化实测寻优场景 | [!518](https://gitcode.com/Ascend/msmodeling/merge_requests/518) |

## 6. 已知问题

| 序号 | 问题描述 | 影响范围 | 规避方案 |
| --- | --- | --- | --- |
| 1 | Windows 环境下 PyTorch 2.10 可能运行异常。 | Windows 本地仿真场景 | 建议使用 PyTorch 2.8 及更早版本。 |
| 2 | optix 实测寻优依赖真实硬件、推理引擎、CANN、驱动和固件的配套关系，仓库未单独声明固定组合版本。 | optix 服务化实测寻优场景 | 按 vLLM、MindIE 及昇腾软件栈官方配套表准备环境。 |

## 7. 致谢

感谢以下贡献者对本版本的贡献：

| 序号 | 贡献者 | 贡献内容 | 关联 PR |
| --- | --- | --- | --- |
| 1 | @jia_ya_nan | 新增 throughput_optimizer 自适应 max-batched-tokens，并修复 GLM-5/GLM-5.2 MTP 兼容性问题。 | [!538](https://gitcode.com/Ascend/msmodeling/merge_requests/538)、[!563](https://gitcode.com/Ascend/msmodeling/merge_requests/563) |
| 2 | @minghang_c | 完成 GLM-5.2 IndexShare 适配，并调整 transformers 版本要求。 | [!535](https://gitcode.com/Ascend/msmodeling/merge_requests/535)、[!509](https://gitcode.com/Ascend/msmodeling/merge_requests/509) |
| 3 | @liujiawang / @AvadaKedavrua | 重构 optix 优化器可靠性，并完善 build.py 构建测试入口。 | [!518](https://gitcode.com/Ascend/msmodeling/merge_requests/518)、[!485](https://gitcode.com/Ascend/msmodeling/merge_requests/485)、[!546](https://gitcode.com/Ascend/msmodeling/merge_requests/546)、[!557](https://gitcode.com/Ascend/msmodeling/merge_requests/557) |