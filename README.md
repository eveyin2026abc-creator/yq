# MindStudio-Modeling (msmodeling) 26.1.0 Release Note

- **仓库地址（master 主干）**：<https://gitcode.com/Ascend/msmodeling>
- **仓库地址（develop 开发分支）**：<https://gitcode.com/Ascend/msmodeling/tree/develop>
- **版本号**：26.1.0
- **时间范围**：2026 Q1（2026-01-01 ~ 2026-03-31）

> 说明：msmodeling（MindStudio-Modeling）是面向昇腾 AI 处理器的神经网络推理性能仿真与分析框架，包含模型推理性能仿真（TensorCastCore）、服务化性能仿真（ServingCastCore）与服务化实测寻优（optix）三大能力模块。仓库采用 master / develop 双分支模式：日常特性先合入 develop，稳定后再"同步"回 master 演进并出包（仓库历史中可见"代码从 develop 同步到 master"的合并记录）。本说明基于仓库 README 版本动态与公开可抓取到的 master 分支 PR 记录整理，仅收录可在公开信息中确认的、时间落在 2026 Q1 窗口内的变更；develop 分支的 Tree/Commit 列表页面依赖前端脚本渲染，公开抓取渠道未能获取到其独立于 master 的额外变更内容。仓库本身未提供按版本号归档的独立 CHANGELOG，如需逐条 commit 级别的完整记录，建议查阅仓库 Releases 页面或 Git 历史。

## 一、模型支持新增

| 模型系列 | 新增内容 | 时间 |
| --- | --- | --- |
| Qwen 系列 | 新增 **Qwen3.5 Dense / MoE** 文本输入支持 | 2026-03-31 |
| GLM 系列 | 新增 **GLM-4 MoE** 模型支持 | 2026-03-31 |

## 二、新增特性一览（来源：master 分支可见的合入记录）

> ⚠️ **日期核实说明**：develop 分支的 Tree/Commit 页面无法通过公开渠道抓取，以下内容改为读取 master 分支（仓库历史显示 develop 会定期"同步"合入 master）当前可见的 PR 合入记录整理。gitcode 页面上这些 PR 展示的是"N 天前 / N 小时前"这类**相对时间**，而非绝对日期，我无法据此确认它们是否严格落在 2026 Q1（1~3 月）窗口内——从相对时间推算，这些合入记录很可能更晚。因此下表**仅作为"近期新增特性"清单**呈现，是否计入 26.1.0 / 2026 Q1 范围，请你结合内部版本分支切分点再确认；如有需要我可以按你提供的确切时间再筛选。

### 1. 模型 / 推理能力

| 特性 | 说明 | 来源 PR |
| --- | --- | --- |
| **GLM-5.2 IndexShare 适配** | 在 GLM-5/GLM-5.1 基础上新增 GLM-5.2 支持：新增 GLM5 专用 IndexShare 辅助逻辑，使全量层执行 indexer、共享层复用上一全量层的 top-k indices；扩展 MTP `indexer_types` 以支持 IndexShare 模式；仅为 GLM-5.2 全量/源层分配稀疏注意力 indexer cache | !535 |
| **GLM-5 / GLM-5.2 MTP 兼容性修复** | 修复 repetition、MTP、torch.compile 同时开启时的模型兼容与进程池序列化问题，恢复 `throughput_optimizer` 并行搜索能力；实测同一寻优任务耗时从 23037.61s 降至 1305.07s | !563 |
| **transformers 版本下限提升** | 因 GLM5 DSA 从 tuple return 到三元组返回（含 topk slot）的源码契约变化，将 transformers 最低版本要求从 5.3.0 提升到 5.6.0，避免低版本环境契约不匹配 | !509 |

### 2. 服务化寻优 / Benchmark

| 特性 | 说明 | 来源 PR |
| --- | --- | --- |
| **throughput_optimizer 自适应 max-batched-tokens** | 将 `--max-batched-tokens` 默认值由固定 8192 改为 `None`（自动模式）：按 4× → 2× → 1× input_length 顺序尝试，仅在 Prefill OOM 时降级；为 optimizer early-stop 增加原因标记（区分 Prefill OOM / Decode OOM / TTFT-TPOT 超限）；补齐 PD 分离单 chunk 场景下的 token budget wave 切分逻辑 | !538 |
| **新增 evalscope 插件** | 相比 AISBench，EvalScope 在模型类型覆盖、功能维度、生态集成和评估精度上更具优势，新增该插件作为寻优 benchmark 的补充选项 | !515 |
| **optix 优化器可靠性重构** | 新增结构化 loguru 日志（`run_id`/`stage`/`engine` 上下文，`OPTIX_LOG_LEVEL` 三级）；新增域异常体系（配置缺失、TOML 解析失败、Benchmark 不可用、无可行解等）；Benchmark 启动前 fail-fast 校验；`remove_file` 安全清理守卫，避免误删工作目录；391 项回归测试覆盖 | !518 |

### 3. 工程 / 研发工具链

| 特性 | 说明 | 来源 PR |
| --- | --- | --- |
| **根目录 build.py 统一构建入口** | 新增 `build.py` 作为部门统一构建/测试委托入口：默认构建 wheel（委托 `scripts/build.sh`）、`python build.py test` 委托增量门禁测试（`scripts/run_ci_gate.sh`），支持 `-v/--version` 指定制品版本并在构建后自动恢复 `pyproject.toml` | !485 |
| **build.py test 默认全量回归** | `python build.py test` 在未设置 `MSMODELING_TEST_MAP_PATH` 时，默认执行全量 `pytest tests`（markers 沿用 `pyproject.toml`），设置后才走 CI Gate 增量模式，改善本地日常回归体验 | !557 |
| **build.py 去除 pydantic 强依赖并非交互自举 uv** | 切断 `build.py` 对 pydantic 的 import-time 依赖；新增 bootstrap 逻辑：检测到缺少 `uv` 时非交互安装，保证仅有 Python 环境也能启动构建/测试入口 | !546 |
| **pre-commit 集成 Gitleaks 本地离线密钥扫描** | 新增 `gitleaks-offline-scan` local hook，在提交前对暂存文件做密钥扫描，防止 AK/SK、Token 等敏感信息进入 Git 历史；扫描完全本地离线，不依赖外部服务 | !541 |
| **新增 sig-review AI 代码检视 skill** | 让 PR 作者与检视者通过自然语言与 AI agent 交互完成"请求检视 → 检视 → 完成移交"全流程；按目录自动路由到 10 个子 SIG 并指派检视人，无需安装外部工具 | !553 |

## 四、核心能力模块（截至本版本已具备）

### 1. 模型推理性能仿真（TensorCastCore）

- 支持多种昇腾硬件仿真（Atlas 800 A2/A3、Atlas 350 等），并支持自定义设备画像。
- 支持 LLM Prefill / Decode 分阶段仿真、Prefix Cache 仿真、MTP 投机解码仿真。
- 支持编译与图优化、多流通算掩盖。
- 支持量化仿真（W8A8 / W4A8 / FP8 / MXFP4 等）。
- 支持并行与 MoE 扩展仿真（TP / DP / EP，及 Embedding TP、Vision TP 等细粒度并行）。
- 支持性能模型切换（Roofline / Profiling）、Chrome Trace 可视化与 Debug。
- 支持视频生成 DiT 仿真（Ulysses、CFG、DiT Cache）。

### 2. 服务化性能仿真（ServingCastCore / throughput_optimizer）

- 支持在 SLO 约束（TTFT / TPOT / 服务成本）下自动搜索最优部署参数。
- 支持 PD 混部、PD 分离、PD 配比三种服务模式。
- 支持并行策略搜索（TP / EP / MOE-DP）与 MTP 配置搜索。
- 支持 Chunked Prefill 仿真、Prefix Cache 仿真、变长负载仿真、多流通算掩盖及跨硬件对比。

### 3. 服务化实测寻优（optix）

- 基于 PSO 粒子寻优算法，在 vLLM、MindIE 等真实服务框架上自动搜索满足时延约束的最优部署参数。
- 支持自定义寻优配置与断点续跑。

### 4. Web UI

- 支持 LLM / VL 前向仿真与视频生成仿真的可视化配置。
- 支持吞吐寻优实验（PD 混部 / 分离 / 配比）的命令预览与任务缓存。
- 支持结果展示与导出（曲线、表格、显存/算子明细、Excel）。
"绘制的点会与legend图例部分重叠，绘图部分的xaxis，yaxis的设置需要调整
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --compile --quantize-linear-action W8A8_DYNAMIC --quantize-attention-action DISABLED --tpot-limits 50 --ttft-limits 1300 "
## 五、说明与限制

- 本 release note 中"模型支持新增"章节仅列出了在仓库 README 版本动态中标注时间落在 2026 Q1（2026-01-01 ~ 2026-03-31）区间内的条目；同期还有若干模型支持更新（如 Qwen3 MoE 等）落在 2025 Q4，未计入本版本范围。
- "新增特性一览"章节内容来自 master 分支当前可见的 PR 合入记录（develop 分支页面无法抓取，改用 master 作为代理），页面展示的是相对时间（"N 天前"），无法据此确认这些合入是否严格落在 2026 Q1 窗口内，请结合内部 26.1.0 tag 的实际切分点核实取舍。
- 由于 gitcode 平台的 Releases/Tags 列表页面以及 develop 分支的 Tree 页面均依赖前端渲染，公开抓取渠道无法获取到 26.1.0 tag 或 develop 分支对应的逐条 commit / 完整 PR 清单，因此本文档未罗列具体的 Bugfix / Refactor 类变更条目，避免出现无法核实的内容。如需完整、可追溯的版本变更清单，建议：
  1. 直接查看仓库 [Releases 页面](https://gitcode.com/Ascend/msmodeling/releases)；
  2. 对比 master 与 develop 分支差异（`git log master..develop`），了解已合入 develop、尚未随 26.1.0 发布的变更；
  3. 或对比 26.1.0 tag 与上一个版本 tag 之间的 Git 历史（`git log v<上一版本>..v26.1.0`）。

## 六、参考链接

- 仓库首页（master）：<https://gitcode.com/Ascend/msmodeling>
- develop 开发分支：<https://gitcode.com/Ascend/msmodeling/tree/develop>
- Issues：<https://gitcode.com/Ascend/msmodeling/issues>
- Releases：<https://gitcode.com/Ascend/msmodeling/releases>
- 模型支持与特性支持矩阵：<https://gitcode.com/Ascend/msmodeling/blob/master/docs/zh/user_guide/support_matrix/support_matrix_user_guide.md>
