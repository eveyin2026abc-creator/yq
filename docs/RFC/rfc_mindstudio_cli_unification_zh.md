# RFC: msmodeling 公开命令行统一规范化

## 元数据

| 项目 | 内容 |
|:---|:---|
| **状态** | Draft |
| **作者** | eveyin1 |
| **创建日期** | 2026-08-16 |
| **更新日期** | 2026-08-16 |
| **相关链接** | 分支 `cli`，提交 `4904a2f`；依据《MindStudio 工具链命令行统一规范化设计方案》§4.7 |

---

## 1. Overview (概述)

### 1.1 Summary (简介)

本 RFC 将 msmodeling 公开控制台对齐 MindStudio 工具链命令行规范：长选项 kebab-case、短选项单字符且语义统一、`--version/-V` 与日志分级开关齐全、`--help` 按固定段落输出，枚举取值统一小写 kebab-case。

实现上不删除旧参数。旧写法降为隐藏兼容别名：继续解析、在 stderr 打印一次性弃用提示、不出现在 `--help`。内部 `argparse` dest 与 `UserInputConfig` / ServingCast / OptiX 字段名保持不变，避免仿真内核随 CLI 表面一起改动。

### 1.2 Motivation (动机)

整改前，公开入口混用 `--tp-size` / `--tensor-parallel-size` 风格缩写、`--chrome-trace` 与路径语义不符、optix 残留 `--load_breakpoint` 与多字符短选项 `-lb`，且缺少统一的 `--version/-V` 与 `--log-level`。用户跨工具记忆成本高，Agent 难以从 `--help` 可靠推断参数形态，也不符合部门已评审通过的命令行规范。

不做此提案的影响：

- 存量脚本可继续跑，但新用户与 AI 工作流会持续接触两套互斥写法。
- 后续增量参数会继续发散，4.7 验收无法对 msmodeling 闭环。

### 1.3 Goals (目标)

**目标**

- 公开入口 `msmodeling`、`inference text-generate` / `throughput-optimizer` / `model-adapter` / `video-generate`、`optix` 满足规范 §4.7 中适用于本工具的条款。
- 提供公共词表中语义命中的标准写法：`--model-path`、`--log-level`、`--log-file`、`--verbose/-v`、`--quiet/-q`、`--debug`、`--version/-V`、`--jobs/-j`、`--config/-c`、`--output-file/-o`。
- `--help` 固定输出 `Description` / `Usage` / `Commands`（有子命令时）/ `Required arguments` / `Optional arguments` / `Examples`，有落盘产物时附加 `Output`。
- 带值参数打印语义化 metavar；枚举在 help 中展示小写 kebab-case，默认值同样 kebab-case。
- 旧参数全部可解析；使用旧名时 stderr 一次性告警。

**非目标**

- 不整改 `tools/`、`serving_cast/main.py`、测试辅助脚本等非公开控制台。
- 不实现 wrapper 类 `-- <prog> [args]`（§4.7 第 7 条针对 mssanitizer / msmemscope / msopprof / msprof，msmodeling 不是 launcher）。
- 不把 `--device` 改成规范词表中的 `cpu/npu` 设备类型；本工具 `--device` 表示 TensorCast `DeviceProfile` 名称（如 `TEST_DEVICE`）。
- 不重命名内部 dest（如 `tp_size`、`disagg`、`chrome_trace`），以免破坏 `UserInputConfig.from_args` 与 optimizer 下游。
- 不在本 RFC 同步改 Web UI 命令拼装、Skill 文档中的旧参数示例（列为后续演进）。

### 1.4 方案要求改什么（简表）

依据《MindStudio 工具链命令行统一规范化设计方案》：整改不是只统一参数名，而是 **参数名 (K)、参数值 (V)、帮助信息 (解释 KV)** 三块，外加兼容别名。

| 块 | 规范章节 | 要改的内容 |
|:---|:---|:---|
| 参数名 (K) | 4.2 | 长选项 kebab-case；短选项单字符且 `-V/-v/-o/-c` 语义固定；目录 `-path`、文件 `-file`；并行度写成 `--<维度>-parallel-size`；词表命中项必须提供标准写法 |
| 参数值 (V) | 4.3 | 布尔用 flag / `--no-*`，help 不用 True/False、yes/no；枚举小写 kebab-case；多值用复数名 + nargs |
| 帮助与版本 | 4.4–4.5 | `--help` 含 Description / Usage / Required / Optional / Examples；带值参数有 `<N>` / `<FILE>` / `{a,b,c}`；默认值 `[default: xxx]`；全部支持 `--version/-V` |
| 兼容 | 4.1、4.6、4.7.11–12 | 旧名不删，能解析；不进 `--help`；用到时 stderr 一次性弃用提示 |

对 **msmodeling**：上表 1–4 都要做。§4.7 第 7 条 wrapper `-- <prog> [args]`（mssanitizer / msmemscope / msopprof / msprof）不适用，本工具跳过。内部 dest（`tp_size`、`disagg` 等）不改。

#### 1.4.1 参数一共改哪三类

下列每条都写清：正式名、旧名、内部 dest、作用范围、取值形态。dest 一律不改，旧名一律可解析并告警。

**第一类：改名字（旧名留下当隐藏别名）**

依据 **4.2.2 第 3 条**：并行度统一 `--<维度>-parallel-size`；**4.2.3.2**：`tp`/`pp`/`dp`/`ep`/`dcp` 不在缩写白名单，正式接口须全拼。

单值并行度（`text-generate`；`model-adapter` 的 doctor/verify 覆盖其中标了「适配器」的项）：

| 正式接口 | 隐藏别名 | dest | metavar | 具体改动 |
|:---|:---|:---|:---|:---|
| `--tensor-parallel-size` | `--tp-size` | `tp_size` | `<N>` | 整模型张量并行度，默认 1。适配器同样改名 |
| `--pipeline-parallel-size` | `--pp-size` | `pp_size` | `<N>` | 流水线并行度，默认 1。仅 text-generate |
| `--data-parallel-size` | `--dp-size` | `dp_size` | `<N>` | 数据并行度，默认空。适配器同样改名 |
| `--decode-context-parallel-size` | `--dcp-size` | `dcp_size` | `<N>` | Decode Context Parallel，须整除 TP。仅 text-generate |
| `--expert-parallel-size` | `--ep-size` | `ep_size` | `<N>` | 专家并行度，默认 1。适配器同样改名 |
| `--o-proj-tensor-parallel-size` | `--o-proj-tp-size` | `o_proj_tp_size` | `<N>` | attn o_proj 的 TP。仅 text-generate |
| `--o-proj-data-parallel-size` | `--o-proj-dp-size` | `o_proj_dp_size` | `<N>` | attn o_proj 的 DP。仅 text-generate |
| `--mlp-tensor-parallel-size` | `--mlp-tp-size` | `mlp_tp_size` | `<N>` | MLP 的 TP。仅 text-generate |
| `--mlp-data-parallel-size` | `--mlp-dp-size` | `mlp_dp_size` | `<N>` | MLP 的 DP。仅 text-generate |
| `--lmhead-tensor-parallel-size` | `--lmhead-tp-size` | `lmhead_tp_size` | `<N>` | LM Head 的 TP。仅 text-generate |
| `--lmhead-data-parallel-size` | `--lmhead-dp-size` | `lmhead_dp_size` | `<N>` | LM Head 的 DP。仅 text-generate |
| `--moe-tensor-parallel-size` | `--moe-tp-size` | `moe_tp_size` | `<N>` | 专家 TP。适配器同样改名 |
| `--moe-data-parallel-size` | `--moe-dp-size` | `moe_dp_size` | `<N>` | 专家 DP，默认 1。适配器同样改名 |
| `--vision-tensor-parallel-size` | `--vision-tp-size` | `vision_tp_size` | `<N>` | 视觉模块 TP，默认 1。适配器同样改名 |
| `--word-embedding-tensor-parallel` | `--word-embedding-tp` | `word_embedding_tp` | `{col,row}` | 词嵌入 TP 模式。text-generate 与 throughput-optimizer |

示例：`--tp-size 8` 与 `--tensor-parallel-size 8` 都写入 `args.tp_size == 8`。

多值并行搜索（仅 `throughput-optimizer`，`nargs=*`，空列表表示按 world size 自动生成 2 的幂）：

| 正式接口 | 隐藏别名 | dest | 具体改动 |
|:---|:---|:---|:---|
| `--tensor-parallel-sizes` | `--tp-sizes` | `tp_sizes` | TP 搜索候选，如 `--tensor-parallel-sizes 1 2 4` |
| `--expert-parallel-sizes` | `--ep-sizes` | `ep_sizes` | EP 搜索候选 |
| `--moe-data-parallel-sizes` | `--moe-dp-sizes` | `moe_dp_sizes` | MoE-DP 搜索候选 |
| `--decode-context-parallel-sizes` | `--dcp-sizes` | `dcp_sizes` | DCP 搜索候选 |

视频生成并行（仅 `video-generate`）：

| 正式接口 | 隐藏别名 | dest | 具体改动 |
|:---|:---|:---|:---|
| `--num-devices` | `--world-size` | `world_size` | 设备数，默认 1。公开名与 text-generate 的 `--num-devices` 对齐 |
| `--ulysses-parallel-size` | `--ulysses-size` | `ulysses_size` | Ulysses 序列并行度，默认 1，须整除 `world_size` |

路径与产物后缀（4.2.1 第 5 条：文件 `-file`，目录 `-path`）：

| 正式接口 | 隐藏别名 | dest | 范围 | 具体改动 |
|:---|:---|:---|:---|:---|
| `--chrome-trace-file` | `--chrome-trace` | `chrome_trace` | text-generate、throughput-optimizer、video-generate | 写出 Chrome trace JSON，metavar `<FILE>` |
| `--graph-log-file` | `--graph-log-url` | `graph_log_url` | text-generate | 编译图 dump 路径，`<FILE>` |
| `--profiling-database-path` | `--profiling-database` | `profiling_database` | text-generate、throughput-optimizer、model-adapter verify | profiling CSV 目录，`<DIR>` |
| `--export-empirical-metrics-file` | `--export-empirical-metrics` | `export_empirical_metrics` | text-generate | M1–M5 JSON，须搭配 profiling |
| `-o, --output-file` | `--output` | `output` | model-adapter doctor / verify / export-evidence | JSON 或 evidence YAML；`-o` 为词表标准短选项 |
| `--profile-draft-output-file` | `--profile-draft-output` | `profile_draft_output` | model-adapter doctor | ModelProfile 草稿文件 |
| `--st-case-output-path` | `--st-case-output` | `st_case_output` | model-adapter verify | ST case 文件或目录 |
| `--doctor-report-file` | `--doctor-report` | `doctor_report` | model-adapter export-evidence | 输入 doctor JSON；缺省时 handler 报错（别名不能带 `required=True`） |

其它改名：

| 正式接口 | 隐藏别名 | dest | 范围 | 具体改动 |
|:---|:---|:---|:---|:---|
| `--model-path` / `--model-id` | `--model_id` | `model_id` | 公共 parser、video-generate、model-adapter | 与位置参数 `model_id` 等价；缺一则 `require_model_id` 报错。词表标准名是 `--model-path` |
| `--disaggregation` | `--disagg` | `disagg` | throughput-optimizer | store_true，分离 PD 寻优 |
| `--ttft-limit` | `--ttft-limits` | `ttft_limits` | throughput-optimizer | 单个 TTFT 约束，`<FLOAT>`。旧名是复数但只收一个值，按 4.3.3 改为单数 |
| `--tpot-limit` | `--tpot-limits` | `tpot_limits` | throughput-optimizer | 单个 TPOT 约束，同上 |
| `--mtp-acceptance-rates` | `--mtp-acceptance-rate` | `mtp_acceptance_rate` | throughput-optimizer | `nargs=+`，默认 `[0.9, 0.6, 0.4, 0.2]`，按 4.3.3 改为复数 |
| `--no-repetition` | `--disable-repetition` | `disable_repetition` | text-generate、model-adapter | store_true；另提供对偶 `--repetition`（store_false）。4.3.1 要求 `--name` / `--no-name` |
| `--no-profiling-interpolation` | `--disable-profiling-interpolation` | `disable_profiling_interpolation` | text-generate | 关闭 profiling 插值 |
| `--ignore-existing-profiles` | `--ignore-existing-profile` | `ignore_existing_profile` | model-adapter doctor | `action=append`，多值用复数 |
| `--load-breakpoint` | `--load_breakpoint`、`-lb` | `load_breakpoint` | optix | store_true。废除 snake_case 与多字符短选项 `-lb` |
| `--benchmark-policy` | `--benchmark_policy` | `benchmark_policy` | optix | 短选项仍是词表外的 `-b`；取值见第三类 |

**第二类：补公共参数（原来没有或不全）**

实现集中在 `add_version_option` / `add_log_options`（`cli/spec_cli.py`），由 `get_common_argparser`、各 inference 入口和 optix 挂上。

| 正式接口 | dest / 行为 | 范围 | 具体改动 |
|:---|:---|:---|:---|
| `-V, --version` | `VersionAction`，打印后退出 | 顶层 `msmodeling`、`inference`、text-generate、throughput-optimizer、video-generate、model-adapter 及子命令、optix | 输出 Logo、`msmodeling {ver} ({7 位 git})`、版权、Mulan PSL v2、Repo。`-v` 不再表示 version |
| `--log-level {debug,info,warning,error}` | `log_level`，默认解析为 `info` | 同上（export-evidence 仅 version，无完整仿真日志栈） | 隐藏别名 `--log_level`。help metavar 为 `{debug,info,warning,error}` |
| `-v, --verbose` | `verbose`，store_true | 同上 | 未写 `--log-level` 时等价 debug |
| `-q, --quiet` | `quiet`，store_true | 同上 | 未写 `--log-level` 时等价 error |
| `--debug` | `debug`，store_true | 同上 | 与 `--verbose` 同级，取 debug |
| `--log-file` | `log_file`，`<FILE>` | 同上 | 日志写文件而非 stderr |
| `-j, --jobs` | 默认 8，`<N>`，正整数 | 仅 throughput-optimizer | 寻优进程并发，不是模型 TP/DP |
| `-c, --config` | `config`，`<FILE>` | 仅 optix | TOML 配置；原先已有，本次按词表保留 `-c` 并纳入统一 help |

日志冲突裁决（4.2.3.1，`resolve_log_level`）：

1. 命令行出现 `--log-level` → 以它为准。
2. 否则出现 `--verbose` 或 `--debug` → `debug`。
3. 否则仅 `--quiet` → `error`。
4. 都没有 → `info`。

optix 在解析后再调用 `set_log_level`，接到 loguru。

**第三类：改取值写法（参数名大多不动，合法值变了）**

| 参数 | 正式取值（help 展示） | 旧取值（仍可解析，会告警） | 写入 dest 的实际对象 | 范围 |
|:---|:---|:---|:---|:---|
| `--quantize-linear-action`、`--quantize-non-expert-linear-action` | `{disabled,w8a16-static,w8a8-static,w4a8-static,w8a16-dynamic,w8a8-dynamic,w4a8-dynamic,fp8,mxfp4}` | `W8A8_DYNAMIC`、`DISABLED` 等 UPPER_SNAKE | `QuantizeLinearAction` 成员 | text-generate、throughput-optimizer、video-generate、model-adapter doctor |
| `--quantize-attention-action` | `{disabled,int8,fp8}` | `DISABLED`、`INT8`、`FP8` | `QuantizeAttentionAction` 成员 | 同上 |
| `--compilation-config` | `{enable-multistream,enable-sequence-parallel,enable-matmul-allreduce,enable-dispatch-ffn-combine}`，`nargs=*` | `enable_multistream` 等 snake_case | 内部仍存 snake_case 配置键 | text-generate、throughput-optimizer |
| `--concurrency-search-strategy` | `{exponential,linear-exponential}` | `linear_exponential` | 内部仍存 snake_case | throughput-optimizer |
| `--attention-backend` | `{dense,block-sparse-attention}` | `block_sparse_attention` | `AttentionBackend` 成员 | video-generate |
| `--benchmark-policy` | `{ais_bench,vllm_benchmark}`（注册表固定名，不改成 kebab） | 无 | 注册表键 `ais_bench` / `vllm_benchmark` | optix |
| `--engine` | 注册表固定名（如 `vllm`、`mindie`） | 无 | 内部仍是注册表键 | optix |
| `--remote-source` | `{huggingface,modelscope}` | 原本已是小写，仅补 metavar | 字符串 | text-generate、video-generate、model-adapter |
| `--performance-model` | `{analytic,profiling}` | 原本已是小写，仅补 metavar | 字符串或 append 列表 | text-generate、throughput-optimizer、model-adapter verify |
| `--devices` / `--device` | `<NAME> [<NAME> ...]` | 单值 `--device TEST_DEVICE` 仍合法 | `device`：optimizer 为 list，其它入口为单个 profile 名 | optimizer 公开复数 `--devices`，同时保留 `--device` 为正式入口（不告警） |

布尔：help 用 `[default: off]` / `[default: on]`，不再写 True/False。`--enable-redundant-experts` 等说明改为陈述句，去掉 “When this flag is True”。

多值 metavar：`nargs=+` 显示 `<N> [<N> ...]`，`nargs=*` 显示 `[<N> ...]`，由 `SpecHelpFormatter` 按 nargs 展开，避免 Usage 行重复包一层。

**第四类：帮助与版本体例（不是改某个 dest，但每个命令都改了输出）**

| 项 | 具体改动 |
|:---|:---|
| 帮助段落 | `SpecArgumentParser.format_help()` 固定输出 Description / Usage / Commands（有子命令时）/ Required arguments / Optional arguments / Examples；有落盘时加 Output |
| 必填/可选 | 靠段落分组，禁止 `<Required>`、`[Mandatory]` 等行内标签 |
| 默认值 | 行尾 `[default: xxx]`；无默认不写；禁止 `(default: None)`。枚举默认展示 kebab，如 `[default: w8a8-dynamic]` |
| metavar | `<N>` / `<FILE>` / `<DIR>` / `<FLOAT>` / `<NAME>` / `<RANGE>` / `{a,b,c}` |
| `--help` 文案 | `Show help message.` |
| 示例 | 每个公开子命令至少 1 条可运行命令，推荐带 `#` 注释 |
| 版本 | 见第二类 `-V, --version` |

#### 1.4.2 旧接口是否还能用

能用。旧参数不删除，只是降为隐藏别名：

1. **能解析**：`--tp-size 8` 与 `--tensor-parallel-size 8` 效果相同。
2. **stderr 打一次弃用提示**：`WARNING: --tp-size is deprecated; use --tensor-parallel-size instead.`
3. **不出现在 `--help`**：新用户只看到正式名。

存量脚本、CI 与现有 UT/ST 不必先改参数名。内部变量名（dest）也不改。

---

## 2. Use Case Analysis (用例分析)

| 用例 | 行为 | DFX |
|:---|:---|:---|
| 新用户查看帮助 | `msmodeling --help` 与各子命令 `--help` 只展示标准名、metavar、默认值与至少 1 条可运行示例 | 可学习、可被 Agent 解析 |
| 查询版本 | 任意公开入口 `--version` / `-V` 打印 Logo、`msmodeling {ver} ({git})`、版权与 Mulan PSL v2 | 排障可确认安装版本 |
| 调节日志 | `--log-level {debug,info,warning,error}` 默认 info；`--verbose/-v` 与 `--debug` 等价 debug，`--quiet/-q` 等价 error；显式 `--log-level` 优先 | 与规范 4.2.3.1 冲突裁决一致 |
| 新脚本使用标准名 | `--tensor-parallel-size 8`、`--chrome-trace-file out.json`、`--disaggregation`、`--load-breakpoint` | 正式接口无 snake_case、无多字符短选项 |
| 存量脚本 | `--tp-size`、`--chrome-trace`、`--disagg`、`--load_breakpoint`、`-lb` 仍可解析 | 兼容性；stderr 引导迁移 |
| 模型标识 | 位置参数 `model_id` 与 `--model-path` / `--model-id` 等价；缺一不可 | 对齐词表 `--model-path`，不强制删除位置参数 |
| 多硬件寻优 | `throughput-optimizer` 公开 `--devices`（可重复），`--device` 仍作为同 dest 的公开单/多值入口 | 多值用复数名，不把 `--device` 标成弃用 |
| 适配器导出 | `model-adapter` 子命令用 `--output-file/-o` 写 JSON/YAML；`--output` 为隐藏别名 | `-o` 语义符合词表 |

使用约束：

- `--device` 取值是已注册 DeviceProfile 名，不是 `cpu`/`npu`。
- 量化等枚举对外接受 `w8a8-dynamic`，对内仍落到 `QuantizeLinearAction.W8A8_DYNAMIC`。
- `--compilation-config` help 展示 kebab-case，解析后仍写入内部 snake_case 选项名。

---

## 3. Design (方案设计)

### 3.1 Overall Design (总体方案)

在 `cli/spec_cli.py` 集中实现规范内核，各公开 parser 复用，避免每个子命令各写一套 help/别名逻辑。

```text
msmodeling (cli/main.py, SpecArgumentParser)
├── --version / -V
├── inference
│   ├── text-generate      parents=get_common_argparser()
│   ├── throughput-optimizer
│   ├── model-adapter {doctor, verify, export-evidence}
│   └── video-generate
└── optix                  optix/optimizer/optimizer.py
        │
        ▼
cli/spec_cli.py
├── SpecArgumentParser / SpecHelpFormatter
├── add_option(..., aliases=...)     # 隐藏别名 + 一次性告警
├── add_version_option / add_log_options
├── make_enum_type / make_token_type
└── parse_args → warn + resolve_log_level
```

核心逻辑：

1. **公开选项**用 kebab-case 长选项；标准短选项仅 `-h/-V/-v/-q/-o/-j/-c/-e/-b` 等单字符。
2. **兼容别名**经 `add_option(..., aliases=)` 注册，`help=argparse.SUPPRESS`，命中时 `WARNING: {old} is deprecated; use {new} instead.`
3. **dest 不变**：例如公开 `--tensor-parallel-size` 的 dest 仍是 `tp_size`。
4. **帮助重排**：`SpecArgumentParser.format_help()` 重写全文，不依赖 argparse 默认分组标题。
5. **日志**：`resolve_log_level` 在 `parse_args` 后写回 `args.log_level`；optix 再调用 `set_log_level` 接到 loguru。

### 3.2 Technology Selection (技术选型)

| 方案 | 结论 | 理由 |
|:---|:---|:---|
| 新建 `cli/spec_cli.py` 作为唯一规范内核 | 采用 | 各入口共享 formatter、version、别名与枚举解析 |
| 删除旧参数，只保留新名 | 不采用 | 违反规范「兼容别名 + 存量脚本零中断」 |
| 同步重命名内部 dest / UserInputConfig 字段 | 不采用 | 改动面扩大到仿真内核，与 CLI 表面整改解耦 |
| 把 `--device` 改为 cpu/npu | 不采用 | 仿真目标是 DeviceProfile；改语义会破坏现有脚本与测试 |
| 为每个子命令手写 help 字符串 | 不采用 | 体例无法保证跨命令一致 |

### 3.3 Security, Privacy, and DFX Design (安全隐私与 DFX 设计)

| 属性 | 设计 |
|:---|:---|
| 兼容性 | 旧长选项、旧枚举大写/下划线取值、`-lb` 均可解析；dest 与默认值语义不变 |
| 可维护性 | 别名、metavar 常量、help 段落集中在 `spec_cli.py` |
| 可测试性 | `tests/regression/cli/test_spec_cli.py` 扫描 help 无 snake_case、无多字符短选项、无 `(default: None)`，并断言别名告警 |
| 可靠性 | 别名告警按 `old->new` 去重；枚举非法值走 `ArgumentTypeError`，choices 在 help 可见 |
| 安全 | `--version` 不打印 Token 或本机路径；模型路径校验仍走 `check_string_valid`；optix 保留 root 运行告警 |

### 3.4 Programming and Integration Design (编程与调用设计)

#### 3.4.1 Basic Programming Model Design (编程模型基本设计)

- 语言与框架：Python 3.10+，stdlib `argparse`。
- 新增公开命令必须使用 `SpecArgumentParser`，通过 `add_option` / `add_version_option` / `add_log_options` 注册参数，经 `parse_args` 收尾。
- 子 parser 若 `parents=` 了 `get_common_argparser()`，需 `inherit_deprecated` 以便父级别名告警仍生效。
- 验收：对照规范 §4.7；以各入口 `--help` 扫描与现有 ST/UT 解析旧写法为准。

#### 3.4.2 API Definition and Design (接口定义与设计)

##### `add_option`

- **描述**：注册公开选项及隐藏兼容别名。
- **原型**：`add_option(target, *option_strings, aliases=(), **kwargs) -> argparse.Action`
- **约束**：`aliases` 的 `help` 为 `SUPPRESS`，并从 kwargs 去掉 `required`，避免「必填 + 别名」在 argparse 中失效；必填在 handler 内校验（如 `--doctor-report-file`）。
- **变更**：新增公共辅助，不改变仿真 API。

##### `parse_args` / `resolve_log_level`

- **描述**：`parse_args` 在 `parser.parse_args` 之后扫描 argv 中的别名、解析日志级别。
- **冲突裁决**：显式 `--log-level` 优先；否则 `--verbose` 或 `--debug` → `debug`；仅 `--quiet` → `error`；默认 `info`。

##### `make_enum_type` / `make_token_type`

- **描述**：对外接受 kebab-case；旧 UPPER_SNAKE / snake_case 作为值别名并告警。
- **存储**：枚举返回成员实例；`compilation-config` 等 token 可 `store_canonical="snake"` 以对接内部配置键。

##### 公开参数映射（节选）

| 概念 | 正式接口 | 隐藏别名（仍可解析） | dest（不变） |
|:---|:---|:---|:---|
| 张量并行 | `--tensor-parallel-size` | `--tp-size` | `tp_size` |
| 流水线并行 | `--pipeline-parallel-size` | `--pp-size` | `pp_size` |
| 数据并行 | `--data-parallel-size` | `--dp-size` | `dp_size` |
| 专家并行 | `--expert-parallel-size` | `--ep-size` | `ep_size` |
| DCP | `--decode-context-parallel-size` | `--dcp-size` | `dcp_size` |
| MoE TP/DP | `--moe-tensor-parallel-size` / `--moe-data-parallel-size` | `--moe-tp-size` / `--moe-dp-size` | `moe_tp_size` / `moe_dp_size` |
| 寻优并行集合 | `--tensor-parallel-sizes` 等复数 | `--tp-sizes` 等 | `tp_sizes` 等 |
| 分离部署 | `--disaggregation` | `--disagg` | `disagg` |
| Trace | `--chrome-trace-file` | `--chrome-trace` | `chrome_trace` |
| Profiling 库 | `--profiling-database-path` | `--profiling-database` | `profiling_database` |
| 模型路径 | `--model-path` / `--model-id` + 位置参数 | `--model_id` | `model_id` |
| 输出文件 | `-o, --output-file` | `--output` | `output` |
| 断点续跑 | `--load-breakpoint` | `--load_breakpoint`、`-lb` | `load_breakpoint` |
| 反向开关 | `--no-repetition`（对偶 `--repetition`） | `--disable-repetition` | `disable_repetition` |

optix 引擎/benchmark 取值是注册表固定名（`ais_bench`、`vllm_benchmark`），help 与文档保持该写法，不改成 kebab-case。

#### 3.4.3 Usage Instructions (使用说明)

标准写法：

```bash
msmodeling inference text-generate Qwen/Qwen3-32B \
  --num-queries 1 --query-length 128 --device TEST_DEVICE \
  --tensor-parallel-size 8 --chrome-trace-file trace.json

msmodeling inference throughput-optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE --num-devices 8 \
  --input-length 1024 --output-length 512 --disaggregation

msmodeling inference model-adapter doctor --model-id Qwen/Qwen3-32B -o doctor.json
msmodeling optix -e vllm -b ais_bench --config ./config.toml
```

约束：

- `-v` 只表示 verbose，版本只用 `-V`。
- 旧写法可用，但会告警，且不会出现在 `--help`。
- `throughput-optimizer` 的 `--jobs/-j` 表示寻优进程并发，不是模型并行度。

---

## 4. Test Design (测试设计)

| 类型 | 位置 | 覆盖 |
|:---|:---|:---|
| 规范回归 | `tests/regression/cli/test_spec_cli.py` | help 段落、kebab 默认值、隐藏旧名、别名告警、`--version`、verbose/quiet 裁决 |
| 入口回归 | `tests/regression/cli/test_main.py`、`test_cli_utils.py`、`test_logo_cli_hooks.py`、`test_compile.py` | 分发、公共 parser、Logo 在 `--help` 时抑制 |
| 存量解析 | 现有 `test_throughput_optimizer.py`、optix `test_main_cli.py`、smoke | `--disagg`、`--ttft-limits`、`-lb`、`--tp-sizes` 等旧写法仍可解析 |

验收对应 §4.7：

1. 词表命中项均有标准写法。
2. 公开入口 `--version/-V` 可用。
3. `--help` 无正式 snake_case 长选项、无单破折号长选项。
4. 短选项无跨子命令一词多义；`-v`/`-V` 符合约定。
5. 正式 help 无 `-lb` 等多字符短选项。
6. `--log-level` 默认 info，快捷开关按 4.2.3.1 生效。
7. wrapper `--`：不适用，本工具跳过。
8. metavar / kebab choices 在 help 可见。
9. help 不以 yes/no、True/False 表达布尔；多值用复数 + nargs。
10. 每个公开子命令 help 含 Usage、必填分段、默认值与示例。
11. 旧参数现有 UT/ST 仍通过。
12. 别名触发一次性 stderr 告警且不进 help。

---

## 5. Drawbacks and Risks (缺点和风险)

| 风险 | 影响 | 应对 |
|:---|:---|:---|
| dest 与公开选项名不一致 | 二次开发若按 dest 猜 CLI 会出错 | RFC 与 help 只承诺 option string；dest 视为内部 |
| Web UI / Skill 仍拼旧名 | 功能可用但会告警 | 别名保留；后续单独改命令拼装与 Skill 文档 |
| `--device` 与规范词表不完全同义 | 跨工具用户可能误解为 cpu/npu | help 写明 DeviceProfile；本 RFC 明确不改语义 |
| 枚举默认从 `W8A8_DYNAMIC` 改为 help 展示 `w8a8-dynamic` | 仅展示变化 | 解析结果仍是同一 StrEnum 成员 |
| 本机缺 `pydantic_settings` 时 optix `--help` 不可用 | 测试跳过 | CI 完整依赖环境仍覆盖 optix |

---

## 6. Existing Technology (现有技术)

参考 POSIX / IEEE Std 1003.1 短选项惯例，以及 argparse/clap 生态对 kebab-case 与 nargs 多值的普遍预期。规范相对 POSIX Guideline 8（逗号单参数）主动选择 nargs/可重复选项，本实现与之保持一致。

与仓库内 `text-generate-executor` 等 Skill RFC 的关系：那些 RFC 描述如何**生成** CLI 命令；本 RFC 定义命令**表面**本身。Skill 文档中的 `--tp-size`、`--chrome-trace` 等示例应在后续演进中改为正式名。

---

## 7. Unresolved Questions (未解决问题)

- 是否在后续版本把 `--device` 拆成 `--device`（类型）+ `--device-profile`（画像名），需产品确认，不在本实现范围。
- Web UI 命令构造与各 Skill `references/*-params.md` 何时切换到新参数名。
- 兼容别名的下线周期（规范未规定删除时间点）。

---

## 附录

### 修改文件

| 文件 | 说明 |
|:---|:---|
| `cli/spec_cli.py` | 规范内核：formatter、version、别名、日志、枚举解析 |
| `cli/utils.py` | 公共 parser：version、log、`--model-path` |
| `cli/main.py` | 顶层 help / version / Commands |
| `cli/inference/text_generate.py` | 并行度家族、路径后缀、枚举 kebab |
| `cli/inference/throughput_optimizer.py` | 复数搜索项、`--disaggregation`、`--jobs/-j` |
| `cli/inference/model_adapter.py` | 子命令 help、`--output-file/-o` |
| `cli/inference/video_generate.py` | `--num-devices`、`--ulysses-parallel-size`、log/version |
| `optix/optimizer/optimizer.py` | kebab 选项、`--load-breakpoint`、引擎/benchmark 取值 |
| `optix/logging.py` | `set_log_level` 对接 `--log-level` |
| `tests/regression/cli/test_spec_cli.py` | 4.7 回归 |

### References (参考资料)

- 《MindStudio 工具链命令行统一规范化设计方案》§4.2–4.7
- [RFC 模板](rfc_template.md)
- [text-generate-executor Skill RFC](rfc_text_generate_executor_skill_zh.md)
- [throughput-optimizer-executor Skill RFC](rfc_throughput_optimizer_executor_skill_zh.md)

### Glossary (术语表)

| 术语 | 含义 |
|:---|:---|
| 正式接口 | 出现在 `--help` 中的选项名 |
| 隐藏别名 | 可解析但不进 help 的旧选项名或旧枚举取值 |
| dest | argparse 写入 `Namespace` 的内部字段名 |
| DeviceProfile | TensorCast 设备画像，`--device` 的取值空间 |

### Documentation Update Plan (文档更新计划)

| 文档 | 变更 |
|:---|:---|
| 本 RFC | 记录公开 CLI 规范落地与兼容策略 |
| 中英文 user guide / quick start | 示例与参数表已改为正式参数名；旧名仍可解析 |
| Web UI 命令拼装 | 后续改为生成 `--chrome-trace-file` 等新名 |
