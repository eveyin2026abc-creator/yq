# MindStudio Modeling 26.0.0 版本发布说明

## 1. 版本概述

MindStudio Modeling 26.0.0 是面向昇腾 AI 处理器的神经网络推理性能仿真与服务化部署寻优版本，主要服务于模型适配、性能预研、容量规划和大模型服务化调优场景。核心亮点如下：

- 提供模型推理性能仿真能力，可在无需真实硬件的情况下，基于设备画像预测模型在昇腾 AI 处理器上的理论性能。
- 支持 Prefill/Decode 分阶段仿真、Prefix Cache、MTP 投机解码、量化仿真、并行策略与 MoE 扩展等典型大模型推理场景。
- 提供服务化性能仿真与吞吐寻优能力，支持在 TTFT、TPOT 等 SLO 约束下搜索较优部署参数。
- 支持 optix 服务化实测寻优，可结合 vLLM、MindIE 等推理引擎在真实部署环境中自动调参。
- 支持 Web UI 可视化配置、结果展示与导出，便于完成仿真预测、实测验证和可视化分析闭环。

## 2. 配套关系

| 软件/硬件 | 版本要求 | 说明 |
| --- | --- | --- |
| 产品型号 | Atlas 800 A2/A3、Atlas 350 等昇腾设备画像；支持自定义设备画像 | 模型仿真能力本身不要求持有对应真机，optix 实测寻优场景需准备真实硬件环境 |
| 操作系统 | Linux；Windows | 仓库文档提及 Linux 与 Windows 使用场景，具体依赖需按安装指南配置 |
| 驱动版本 | 随真实推理环境配套 | 仅 optix 实测寻优场景涉及真实硬件驱动，仓库未单独声明固定驱动版本 |
| 固件版本 | 随真实推理环境配套 | 仅 optix 实测寻优场景涉及真实硬件固件，仓库未单独声明固定固件版本 |
| CANN 版本 | 随真实推理环境配套 | 仿真功能不强依赖 CANN；optix 场景需结合所选推理引擎确认 CANN 配套版本 |
| Python 版本 | 3.10 及以上 | 根据仓库 README 整理 |
| PyTorch 版本 | 建议使用 2.8 及更早版本 | 仓库 README 提示 Windows 环境下 PyTorch 2.10 可能运行异常 |
| transformers 版本 | 5.6.0 及以上，低于 5.8.0 | GLM5 系列模型依赖 5.6.0 及以上版本的返回值和 `indexer_types` 契约 |
| 推理引擎 | vLLM、MindIE 等 | 主要用于 optix 服务化实测寻优场景，具体版本需结合推理引擎官方配套关系确认 |
| 许可证 | MulanPSL2-style License | 根据仓库 README 与 LICENSE 信息整理 |

## 3. 新增特性

| 序号 | 特性名称 | 特性描述 | 关联 Issue/PR |
| --- | --- | --- | --- |
| 1 | Qwen3.5 Dense/MoE 文本输入支持 | 新增 Qwen3.5 Dense、Qwen3.5 MoE 模型的文本输入仿真支持，扩展 Qwen 系列模型覆盖范围。 | README 版本动态 |
| 2 | GLM-4 MoE 模型支持 | 新增 GLM-4 MoE 模型仿真支持，补齐 GLM 系列 MoE 推理性能评估能力。 | README 版本动态 |
| 3 | GLM-5.2 IndexShare 适配 | 新增 GLM5 专用 IndexShare 辅助逻辑，支持全量层执行 indexer、共享层复用上一全量层 top-k indices，并扩展 MTP `indexer_types` 支持。 | !535 |
| 4 | GLM-5/GLM-5.2 MTP 兼容性增强 | 修复 repetition、MTP、torch.compile 同时开启时的模型兼容与进程池序列化问题，恢复 `throughput_optimizer` 并行搜索能力。 | !563 |
| 5 | transformers 版本下限提升 | 根据 GLM5 DSA 返回值与 `indexer_types` 契约变化，将 transformers 最低版本要求提升至 5.6.0，避免低版本环境不兼容。 | !509 |
| 6 | throughput_optimizer 自适应 max-batched-tokens | `--max-batched-tokens` 支持自动模式，按 4 倍、2 倍、1 倍 input_length 顺序尝试 token budget，并在 Prefill OOM 时自动降级。 | !538 |
| 7 | EvalScope benchmark 插件 | 新增 EvalScope 作为服务化寻优 benchmark 补充选项，增强模型类型覆盖、评估功能与生态集成能力。 | !515 |
| 8 | optix 优化器可靠性增强 | 新增结构化 loguru 日志、领域异常体系、Benchmark 启动前 fail-fast 校验和安全清理守卫，提升服务化实测寻优稳定性。 | !518 |
| 9 | 根目录 build.py 统一构建入口 | 新增 `build.py` 作为统一构建/测试委托入口，支持 wheel 构建、增量门禁测试委托和 `-v/--version` 指定制品版本。 | !485 |
| 10 | build.py test 默认全量回归 | `python build.py test` 在未设置 `MSMODELING_TEST_MAP_PATH` 时默认执行全量 `pytest tests`，改善本地回归体验。 | !557 |
| 11 | build.py 去除 pydantic 强依赖并支持 uv 自举 | 切断 `build.py` 对 pydantic 的 import-time 依赖，并在缺少 uv 时支持非交互自举，降低构建入口启动门槛。 | !546 |
| 12 | pre-commit 集成 Gitleaks 本地离线密钥扫描 | 新增 `gitleaks-offline-scan` local hook，在提交前对暂存文件做本地离线密钥扫描，降低敏感信息进入 Git 历史的风险。 | !541 |
| 13 | sig-review AI 代码检视 skill | 新增面向 PR 检视流程的 AI 代码检视 skill，支持按目录自动路由到子 SIG 并辅助完成检视移交。 | !553 |

## 4. 变更说明

### 2.1 模型支持新增

| 特性 | 说明 | 来源 |
| --- | --- | --- |
| GLM-5.2 IndexShare 适配 | 在 GLM-5/GLM-5.1 基础上新增 GLM-5.2 支持：新增 GLM5 专用 IndexShare 辅助逻辑，全量层执行 indexer、共享层复用上一全量层的 top-k indices；扩展 MTP `indexer_types` 支持 IndexShare 模式 | PR !535 |
| Qwen3.5 Dense / MoE 文本输入支持 | 新增 Qwen3.5 Dense、Qwen3.5 MoE 模型的文本输入仿真支持 | README 版本动态（2026-03-31） |
| GLM-4 MoE 模型支持 | 新增 GLM-4 MoE 模型仿真支持 | README 版本动态（2026-03-31） |

| 序号 | Issue 链接 | 问题描述 | 影响范围 |
| --- | --- | --- | --- |
| 1 | !563 | 修复 GLM-5/GLM-5.2 在同时启用 repetition、MTP、torch.compile 时的模型兼容与进程池序列化问题，恢复并行搜索能力。 | GLM-5/GLM-5.2 服务化寻优场景 |
| 2 | !509 | 修复 GLM5 DSA 契约不匹配导致的运行时错误，通过提升 transformers 最低版本要求避免低版本返回值不兼容。 | GLM5 系列模型仿真场景 |
| 3 | !558、!559 | 修复文档失效链接、Markdown 格式问题和部分模型支持说明缺失问题，提升文档可读性与可维护性。 | 文档阅读与维护场景 |
| 4 | !518 | 重构 optix 优化器可靠性，新增结构化日志、领域异常、Benchmark 启动前 fail-fast 校验和安全清理守卫。 | optix 服务化实测寻优场景 |

## 6. 已知问题

| 序号 | 问题描述 | 影响范围 | 规避方案 |
| --- | --- | --- | --- |
| 1 | Windows 环境下 PyTorch 2.10 可能运行异常。 | Windows 本地仿真场景 | 建议使用 PyTorch 2.8 及更早版本。 |
| 2 | optix 实测寻优依赖真实硬件、推理引擎、CANN、驱动和固件的配套关系，仓库未单独声明固定组合版本。 | optix 服务化实测寻优场景 | 按 vLLM、MindIE 及昇腾软件栈官方配套表准备环境。 |

## 7. 致谢

感谢以下贡献者对本版本的贡献：

| 序号 | 贡献者 | 贡献内容 | 关联 PR |
| --- | --- | --- | --- |
| 1 | @jia_ya_nan | 新增 throughput_optimizer 自适应 max-batched-tokens，并修复 GLM-5/GLM-5.2 MTP 兼容性问题。 | !538、!563 |
| 2 | @minghang_c | 完成 GLM-5.2 IndexShare 适配，并调整 transformers 版本要求。 | !535、!509 |
| 3 | @liujiawang / @AvadaKedavrua | 重构 optix 优化器可靠性，并完善 build.py 构建测试入口。 | !518、!485、!546、!557 |
| 4 | @eveyin1 | 集成 pre-commit Gitleaks 本地离线密钥扫描，并修复文档格式问题。 | !541、!559 |
| 5 | @tt0cool | 修复文档失效链接。 | !558 |
| 6 | @jhon-117 | 新增 sig-review AI 代码检视 skill。 | !553 |
| 7 | @liu977803265 | 新增 EvalScope benchmark 插件。 | !515 |

## 参考链接

- 仓库首页（master）：<https://gitcode.com/Ascend/msmodeling>
- develop 开发分支：<https://gitcode.com/Ascend/msmodeling/tree/develop>
- Issues：<https://gitcode.com/Ascend/msmodeling/issues>
- Releases：<https://gitcode.com/Ascend/msmodeling/releases>
- 模型支持与特性支持矩阵：<https://gitcode.com/Ascend/msmodeling/blob/master/docs/zh/user_guide/support_matrix/support_matrix_user_guide.md>