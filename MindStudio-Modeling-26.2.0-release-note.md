# MindStudio Modeling 26.2.0 版本发布说明

*发布日期：2026-10-09（以分支 `26.2.0` 最新构建标签 `tag_MindStudio_26.2.0.B100_001` 为准；正式发布件日期以版本发布流程为准）*

## 1. 版本概述

MindStudio Modeling（msModeling）26.2.0 是面向昇腾 AI 处理器的推理性能仿真与服务寻优版本，主要服务模型适配、部署评估和实测调优人员。本说明依据 [2026 Q3 Roadmap](https://gitcode.com/Ascend/msmodeling/issues/286)（里程碑 MindStudio 26.2.0），整理 `26.2.0` 相对 `26.1.0` 在 2026-06-23 至 2026-10-09 合入的用户可见能力。核心亮点如下：

- 扩展重点模型与生成式模型仿真，覆盖 GLM-5.2 / GLM-5.3 Flash、Kimi K3、MiniMax M3、Qwen3.8，以及 FLUX.1-dev、Qwen-Image-Edit 图像生成。
- 把流水线并行、DSA 上下文并行、DFlash/DSpark 投机解码和 Chunked Prefill 接到服务化吞吐评估，便于评估长序列和复杂并行部署。
- 用 GLM-5.2 实测算子库、Profiling 插值和解析模型校准补强性能估算，并提供模型适配阶段的理论与运行时诊断。
- 重构 Web 控制台，并让 OptiX 支持 Agent 寻优、多机混布 / PD 分离和 benchmark 早停。
- 新增 Atlas A5 系列设备画像，多硬件批量仿真可复用与硬件无关的 workload 分析。

## 2. 配套关系

仿真本身不绑定真实硬件栈。驱动、固件和 CANN 只在源码构建镜像或 OptiX 实测部署时按目标环境配套。

| 软件/硬件 | 版本要求 | 说明 |
| --- | --- | --- |
| 产品型号 | Atlas 800 A2 训练/推理系列；Atlas 800 A3 训练/推理系列；Atlas 350 加速卡；Atlas 850/850E；Atlas 950（A5） | 按支持矩阵与内置设备画像整理。Atlas 950 为本版本新增，同时支持自定义设备画像 |
| 操作系统 | Linux；macOS；Windows | 在线安装与离线安装覆盖这三类系统。源码构建使用 MindStudio 统一构建镜像，基础系统为 openEuler |
| 驱动版本 | 不依赖固定驱动版本 | 模型仿真可在无真实硬件驱动的环境下运行。OptiX 实测跟随目标部署环境的驱动 |
| 固件版本 | 不依赖固定固件版本 | 模型仿真可在无真实硬件固件的环境下运行。OptiX 实测跟随目标部署环境的固件 |
| CANN 版本 | 不依赖固定 CANN 版本 | 模型仿真可在未安装 CANN 的环境下运行。源码构建以统一构建镜像中的 CANN 为准。GLM-5.2 实测算子样例库对齐 CANN 9.0.1 |
| Python 版本 | 3.10 及以上 | 安装与运行要求。开发容器示例为 Python 3.11 |
| PyTorch 版本 | 2.8 至 2.10 | 同时需要 `torchvision>=0.23.0`。Windows 上 PyTorch 2.10 可能运行异常，遇到问题时改用 2.8 |
| transformers 版本 | 5.16.0 及以上，低于 5.17.0 | GLM5 等模型依赖该版本区间的配置与返回值契约 |
| diffusers 版本 | 0.38.0 及以上 | 图像生成与视频生成仿真依赖 |
| 推理引擎 | vLLM、MindIE 等 | 仅用于 OptiX 服务化实测寻优，安装在系统部署环境，不装进 msModeling 虚拟环境。具体版本遵循推理引擎官方配套。GLM-5.2 实测算子样例对应 vLLM-Ascend 0.23.0rc1 |
| 许可证 | 木兰宽松许可证第 2 版（MulanPSL2） | 见仓库许可证声明 |

## 3. 新增特性

| 序号 | 特性名称 | 特性描述 | 关联 Issue/PR |
| --- | --- | --- | --- |
| 1 | GLM-5.2 / GLM-5.3 Flash 仿真 | 支持 GLM-5.2 的 IndexShare 稀疏注意力建模，避免把共享层重复算成完整 indexer；新增 GLM-5.3 Flash 仿真，并提供对齐 CANN 9.0.1 软件栈的 GLM-5.2 实测算子库。 | [!535](https://gitcode.com/Ascend/msmodeling/merge_requests/535)、[!592](https://gitcode.com/Ascend/msmodeling/merge_requests/592)、[!786](https://gitcode.com/Ascend/msmodeling/merge_requests/786)、[!805](https://gitcode.com/Ascend/msmodeling/merge_requests/805) |
| 2 | Kimi K3、MiniMax M3 与 Qwen3.8 | 支持 Kimi K3 文本及视觉语言仿真，支持 MiniMax M3，并新增 Qwen3.8 文本仿真（复用 Qwen3.5 MoE 路径）。 | [!663](https://gitcode.com/Ascend/msmodeling/merge_requests/663)、[!566](https://gitcode.com/Ascend/msmodeling/merge_requests/566)、[!747](https://gitcode.com/Ascend/msmodeling/merge_requests/747) |
| 3 | 图像生成仿真 | 新增图像生成仿真主路径，支持 FLUX.1-dev 与 Qwen-Image-Edit 的多步去噪 workload，可评估尺寸、文本条件和采样步数。 | [!700](https://gitcode.com/Ascend/msmodeling/merge_requests/700)、[!701](https://gitcode.com/Ascend/msmodeling/merge_requests/701)、[!708](https://gitcode.com/Ascend/msmodeling/merge_requests/708) |
| 4 | 流水线并行与投机解码协同 | `throughput_optimizer` 支持 PP 搜索；PP 可与 MTP、DFlash、DSpark 同时评估，输出调度感知的吞吐和气泡。 | [!711](https://gitcode.com/Ascend/msmodeling/merge_requests/711)、[!773](https://gitcode.com/Ascend/msmodeling/merge_requests/773) |
| 5 | DSA 上下文并行 | 支持 DSA 路径的上下文并行，用于长序列场景下的计算、通信和显存评估。 | [!555](https://gitcode.com/Ascend/msmodeling/merge_requests/555) |
| 6 | DFlash / DSpark 投机推理 | 在 `text_generate` 和 `throughput_optimizer` 中支持 DFlash 与 DSpark，并与既有 MTP 使用统一的投机推理参数边界。 | [!675](https://gitcode.com/Ascend/msmodeling/merge_requests/675)、[!742](https://gitcode.com/Ascend/msmodeling/merge_requests/742)、[!779](https://gitcode.com/Ascend/msmodeling/merge_requests/779) |
| 7 | Chunked Prefill 与变长输入 | 长 prompt 超过单步 prefill 预算时按 chunk 建模，并支持变长输入的混部评估，便于观察 TTFT、P 阶段吞吐和显存。 | [!653](https://gitcode.com/Ascend/msmodeling/merge_requests/653) |
| 8 | 实测性能建模与校准 | 增强 Profiling 插值；新增 analytic calibration，可用实测 profile 校准解析时延，未命中或不匹配设备/软件栈时回退原始解析模型。 | [!262](https://gitcode.com/Ascend/msmodeling/merge_requests/262)、[!388](https://gitcode.com/Ascend/msmodeling/merge_requests/388)、[!389](https://gitcode.com/Ascend/msmodeling/merge_requests/389)、[!816](https://gitcode.com/Ascend/msmodeling/merge_requests/816) |
| 9 | 模型适配诊断 | 提供理论与运行时成对诊断，覆盖混合注意力以及 Qwen3 MoE、DeepSeek、GLM、Kimi 等文本模型的兼容检查。 | [!588](https://gitcode.com/Ascend/msmodeling/merge_requests/588)、[!698](https://gitcode.com/Ascend/msmodeling/merge_requests/698)、[!724](https://gitcode.com/Ascend/msmodeling/merge_requests/724) |
| 10 | DeepSeek V4 主 KV 混合精度存储 | 开启 attention FP8 量化时，按混合行估算 V4 主 KV 存储和反量化开销；关闭量化时主 KV 按 BF16 估算。 | [!840](https://gitcode.com/Ascend/msmodeling/merge_requests/840) |
| 11 | Vue 3 Web 控制台 | Web 控制台迁移为 Vue 3 + FastAPI，覆盖前向仿真、视频生成和吞吐寻优，并支持模型选择、结果可视化和历史任务。 | [!632](https://gitcode.com/Ascend/msmodeling/merge_requests/632) |
| 12 | 多硬件 workload 复用与 Atlas A5 | 多硬件搜索复用与硬件无关的计算量、访存量和通信量；新增 Atlas A5 系列设备画像。 | [!655](https://gitcode.com/Ascend/msmodeling/merge_requests/655)、[!731](https://gitcode.com/Ascend/msmodeling/merge_requests/731) |
| 13 | OptiX Agent、多机与早停 | OptiX 增加 Agent 寻优模式；支持多机混合部署和多机 PD 分离；vLLM benchmark 可在明显差于基线时提前结束。 | [!778](https://gitcode.com/Ascend/msmodeling/merge_requests/778)、[!686](https://gitcode.com/Ascend/msmodeling/merge_requests/686)、[!542](https://gitcode.com/Ascend/msmodeling/merge_requests/542) |

## 4. 变更说明

| 序号 | 变更内容 | 变更影响 | 关联 Issue/PR |
| --- | --- | --- | --- |
| 1 | OptiX 与仿真环境分离 | **不兼容变更**：msModeling / OptiX 必须安装在独立虚拟环境。vLLM、MindIE 和测评工具使用系统部署环境。不要在 msModeling 虚拟环境中执行 `pip install vllm`。 | [!577](https://gitcode.com/Ascend/msmodeling/merge_requests/577) |
| 2 | vLLM 寻优参数位置统一 | **不兼容变更**：vLLM 命令行寻优字段的 `config_position` 统一为 `run`。仍使用旧位置的配置需要改为 `run` 后才能被追加到 `vllm serve`。环境变量继续使用 `env`，MindIE 仍使用服务配置路径。 | [!877](https://gitcode.com/Ascend/msmodeling/merge_requests/877) |
| 3 | 编译路径默认开启多流 | 多流调度改为默认开启。未显式关闭时，同一配置的算子时延和端到端结果可能与 26.1.0 不同。 | [!410](https://gitcode.com/Ascend/msmodeling/merge_requests/410) |
| 4 | DFC 默认启用 | TensorCast 与吞吐寻优默认启用 DFC。依赖旧默认关闭行为的对比实验需要显式关闭后再对比。 | [!206](https://gitcode.com/Ascend/msmodeling/merge_requests/206) |
| 5 | DeepSeek V4 主 KV 存储口径 | 未开启 attention 量化时，V4 主 KV 固定按 BF16 估算，不再跟随模型默认 dtype。开启 FP8 时改为混合行存储。显存和带宽结果会随之变化。 | [!840](https://gitcode.com/Ascend/msmodeling/merge_requests/840) |
| 6 | GLM5 依赖的 Transformers 下限 | GLM5 系列要求 Transformers 5.16.x。使用更低 Transformers 版本时需要先升级环境。 | [!510](https://gitcode.com/Ascend/msmodeling/merge_requests/510) |
| 7 | 解析模型校准为显式选项 | 新增 `--performance-model calibrated` 及 calibration profile 参数。不传 profile 时，原有 analytic 行为保持不变；profile 与设备或软件栈不匹配时回退原始解析结果。 | [!816](https://gitcode.com/Ascend/msmodeling/merge_requests/816) |

## 5. 修复缺陷

| 序号 | 问题描述 | 影响范围 | 关联 Issue/PR |
| --- | --- | --- | --- |
| 1 | 修复 Windows 本地模型绝对路径被字符白名单拒绝的问题。已存在的盘符路径可以继续作为模型路径使用，非法字符串仍会被拒绝。 | Windows 上通过 CLI 指定本地模型路径 | [!953](https://gitcode.com/Ascend/msmodeling/merge_requests/953) |
| 2 | 修复多进程仿真中单个 worker 初始化失败导致其余进程永久等待的问题。失败后会中止屏障、清理任务并返回明确错误，worker 全部退出时也不再无限阻塞。 | `serving_cast` 多进程仿真 | [!764](https://gitcode.com/Ascend/msmodeling/merge_requests/764)、[!823](https://gitcode.com/Ascend/msmodeling/merge_requests/823) |
| 3 | 修复重计算状态漏计、decode 请求上限用量错误，以及 Chunked Prefill 按阶段工期计算 QPS 不准确的问题。 | Chunked Prefill 与服务化调度估算 | [!806](https://gitcode.com/Ascend/msmodeling/merge_requests/806)、[!826](https://gitcode.com/Ascend/msmodeling/merge_requests/826) |
| 4 | 修复 PD 混部把 TTFT 重复计入端到端时延、从而低估输出吞吐的问题。 | PD 混部吞吐结果 | [!552](https://gitcode.com/Ascend/msmodeling/merge_requests/552) |
| 5 | 修复 Qwen-Image-Edit 在多 source 输入时提示不明确，以及 Qwen-Image-Edit-2511 在 `--compile` 下因索引张量类型混用而失败的问题。基础模型会明确提示只支持单张 source。 | Qwen-Image-Edit / 2511 图像编辑仿真 | [!765](https://gitcode.com/Ascend/msmodeling/merge_requests/765) |
| 6 | 修复 OptiX PSO 首轮候选全部失败时直接退出的问题。首轮失败后会重新生成候选并继续迭代。 | OptiX PSO 寻优 | [!578](https://gitcode.com/Ascend/msmodeling/merge_requests/578) |

## 6. 已知问题

不涉及。

## 7. 致谢

感谢以下贡献者对本版本的贡献：

| 序号 | 贡献者 | 贡献内容 | 关联 PR |
| --- | --- | --- | --- |
| 1 | minghang_c | GLM-5.2 IndexShare 与 FLUX / Qwen-Image 图像生成仿真 | [!535](https://gitcode.com/Ascend/msmodeling/merge_requests/535)、[!700](https://gitcode.com/Ascend/msmodeling/merge_requests/700) |
| 2 | jia_ya_nan | GLM-5.3 Flash 仿真与多硬件 workload 复用 | [!786](https://gitcode.com/Ascend/msmodeling/merge_requests/786)、[!655](https://gitcode.com/Ascend/msmodeling/merge_requests/655) |
| 3 | wangshen001 | Kimi K3 模型适配 | [!663](https://gitcode.com/Ascend/msmodeling/merge_requests/663) |
| 4 | weixin_43113933 | MiniMax M3 模型适配 | [!566](https://gitcode.com/Ascend/msmodeling/merge_requests/566) |
| 5 | hanxinlong1999 | 流水线并行搜索及其与投机解码的协同 | [!711](https://gitcode.com/Ascend/msmodeling/merge_requests/711)、[!773](https://gitcode.com/Ascend/msmodeling/merge_requests/773) |
| 6 | Abstrey | 按真实切分构建流水线并行模型 | [!367](https://gitcode.com/Ascend/msmodeling/merge_requests/367) |
| 7 | stormchasingg | DSA 上下文并行与 Chunked Prefill 变长输入 | [!555](https://gitcode.com/Ascend/msmodeling/merge_requests/555)、[!653](https://gitcode.com/Ascend/msmodeling/merge_requests/653) |
| 8 | liu977803265 | DFlash/DSpark、EvalScope 插件与多机混合部署寻优 | [!675](https://gitcode.com/Ascend/msmodeling/merge_requests/675)、[!686](https://gitcode.com/Ascend/msmodeling/merge_requests/686) |
| 9 | yikangLin | 基于 vLLM 的双机服务化参数寻优插件 | [!403](https://gitcode.com/Ascend/msmodeling/merge_requests/403) |
| 10 | lutean | 解析性能模型实测校准 | [!816](https://gitcode.com/Ascend/msmodeling/merge_requests/816) |
| 11 | zhenyu_zhang | Profiling 插值与外推 | [!262](https://gitcode.com/Ascend/msmodeling/merge_requests/262)、[!388](https://gitcode.com/Ascend/msmodeling/merge_requests/388) |
| 12 | zhenghaojie | GLM-5.2 实测算子性能数据库 | [!805](https://gitcode.com/Ascend/msmodeling/merge_requests/805) |
| 13 | Secluded_Ocean | GLM 实测算子数据与 MicroBench 重放 | [!755](https://gitcode.com/Ascend/msmodeling/merge_requests/755)、[!723](https://gitcode.com/Ascend/msmodeling/merge_requests/723) |
| 14 | Hudingyi | DSA 稀疏注意力算子与吞吐寻优 Profiling 模式 | [!363](https://gitcode.com/Ascend/msmodeling/merge_requests/363)、[!411](https://gitcode.com/Ascend/msmodeling/merge_requests/411) |
| 15 | elrond-g | DCP 场景下按并行度切分 KV Cache | [!444](https://gitcode.com/Ascend/msmodeling/merge_requests/444) |
| 16 | Horacehxw | MTP 投机解码 shape 建模 | [!362](https://gitcode.com/Ascend/msmodeling/merge_requests/362) |
| 17 | genius52 | MoE 字段配置与 GMM、SwiGLU 融合 | [!128](https://gitcode.com/Ascend/msmodeling/merge_requests/128)、[!83](https://gitcode.com/Ascend/msmodeling/merge_requests/83) |
| 18 | weixin_43368449 | Qwen3-VL 的 ViT 张量并行与 MoE | [!46](https://gitcode.com/Ascend/msmodeling/merge_requests/46) |
| 19 | liu_jiaxu | Qwen3-VL 从配置读取 resize 参数 | [!126](https://gitcode.com/Ascend/msmodeling/merge_requests/126) |
| 20 | zwt__ | Vue 3 + FastAPI Web 控制台 | [!632](https://gitcode.com/Ascend/msmodeling/merge_requests/632) |
| 21 | sunguozhong | msModeling 可视化界面 | [!161](https://gitcode.com/Ascend/msmodeling/merge_requests/161) |
| 22 | wendellX | OptiX Agent 寻优模式 | [!778](https://gitcode.com/Ascend/msmodeling/merge_requests/778) |
| 23 | h7star | Atlas A5 设备画像与 OptiX 环境隔离 | [!731](https://gitcode.com/Ascend/msmodeling/merge_requests/731)、[!577](https://gitcode.com/Ascend/msmodeling/merge_requests/577) |
| 24 | cmh1056291129 | 提升 serving_cast 搜索效率 | [!199](https://gitcode.com/Ascend/msmodeling/merge_requests/199) |
| 25 | yuyinkai1 | Qwen3.8 适配与服务化仿真稳定性修复 | [!747](https://gitcode.com/Ascend/msmodeling/merge_requests/747)、[!764](https://gitcode.com/Ascend/msmodeling/merge_requests/764) |
