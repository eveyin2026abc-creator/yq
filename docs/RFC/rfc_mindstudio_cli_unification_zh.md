# RFC: msmodeling 公开命令行统一规范化

## 元数据

| 项目 | 内容 |
|:---|:---|
| **状态** | Draft |
| **作者** | eveyin1 |
| **创建日期** | 2026-08-16 |
| **更新日期** | 2026-08-17 |
| **相关链接** | 分支 `cli`，提交 `4904a2f`；依据《MindStudio 工具链命令行统一规范化设计方案》§4.7 |

---

## 1. Overview (概述)

### 1.1 Summary (简介)

本 RFC 将 msmodeling 公开控制台对齐 MindStudio 工具链命令行规范中的**必须项与值得改项**：长选项 kebab-case（去掉 snake 与多字符短选项）、短选项单字符且语义统一、`--version/-V` 与日志分级开关齐全、`--help` 按固定段落输出。并行度短名（`--tp-size`）与量化原取值（`W8A8_DYNAMIC`）保持正式接口。

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
| 参数名 (K) | 4.2 | 长选项 kebab-case；短选项单字符且 `-V/-v/-o/-c` 语义固定；目录 `-path`、文件 `-file`；词表命中项必须提供标准写法。并行度短名（`--tp-size` 等）保持正式接口，不强制全拼 |
| 参数值 (V) | 4.3 | 布尔用 flag / `--no-*`，help 不用 True/False、yes/no；多值用复数名 + nargs。量化等枚举保持原取值（如 `W8A8_DYNAMIC`），kebab 拼写仍可解析 |
| 帮助与版本 | 4.4–4.5 | `--help` 含 Description / Usage / Required / Optional / Examples；带值参数有 `<N>` / `<FILE>` / `{a,b,c}`；默认值 `[default: xxx]`；全部支持 `--version/-V` |
| 兼容 | 4.1、4.6、4.7.11–12 | 被改名的旧写法不删，能解析；不进 `--help`；用到时 stderr 一次性弃用提示 |

对 **msmodeling**：上表 1–4 都要做。§4.7 第 7 条 wrapper `-- <prog> [args]`（mssanitizer / msmemscope / msopprof / msprof）不适用，本工具跳过。内部 dest（`tp_size`、`disagg` 等）不改。

#### 1.4.1 参数一共改哪三类

落地范围只覆盖**必须改**和**值得改**。并行度短名（`--tp-size`、`--disagg`、量化 `W8A8_DYNAMIC` 等）保持正式接口，与 aiconfigurator / 业界习惯对齐。dest 一律不改；被改名的旧写法可解析并告警。

**第一类：改名字（旧名留下当隐藏别名）**

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
| `--num-devices` | `--world-size` | `world_size` | video-generate | 与其它入口 `--num-devices` 对齐 |
| `--ttft-limit` | `--ttft-limits` | `ttft_limits` | throughput-optimizer | 单个 TTFT 约束，`<FLOAT>`。旧名是复数但只收一个值，按 4.3.3 改为单数 |
| `--tpot-limit` | `--tpot-limits` | `tpot_limits` | throughput-optimizer | 单个 TPOT 约束，同上 |
| `--mtp-acceptance-rates` | `--mtp-acceptance-rate` | `mtp_acceptance_rate` | throughput-optimizer | `nargs=+`，默认 `[0.9, 0.6, 0.4, 0.2]`，按 4.3.3 改为复数 |
| `--no-repetition` | `--disable-repetition` | `disable_repetition` | text-generate、model-adapter | store_true；另提供对偶 `--repetition`（store_false）。4.3.1 要求 `--name` / `--no-name` |
| `--ignore-existing-profiles` | `--ignore-existing-profile` | `ignore_existing_profile` | model-adapter doctor | `action=append`，多值用复数 |
| `--load-breakpoint` | `--load_breakpoint`、`-lb` | `load_breakpoint` | optix | store_true。废除 snake_case 与多字符短选项 `-lb` |
| `--benchmark-policy` | `--benchmark_policy` | `benchmark_policy` | optix | 短选项仍是词表外的 `-b`；取值见第三类 |

**第二类：补公共参数（原来没有或不全）**

实现集中在 `add_version_option` / `add_log_options`（`cli/spec_cli.py`），由 `get_common_argparser`、各 inference 入口和 optix 挂上。

| 正式接口 | dest / 行为 | 范围 | 具体改动 |
|:---|:---|:---|:---|
| `-V, --version` | `VersionAction`，打印后退出 | 顶层 `msmodeling`、`inference`、text-generate、throughput-optimizer、video-generate、model-adapter 及子命令、optix | 输出 Logo、`msmodeling {ver} ({7 位 git})`、版权、Mulan PSL v2、Repo。`-v` 不再表示 version |
| `--log-level {debug,info,warning,error,critical}` | `log_level`，默认解析为 `info` | 同上（export-evidence 仅 version，无完整仿真日志栈） | 隐藏别名 `--log_level`。help metavar 含 `critical` |
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

**第三类：取值与 metavar（参数名大多不动）**

量化、编译开关、attention backend 的正式取值保持原样（`W8A8_DYNAMIC`、`enable_multistream` 等）；kebab 拼写仍可解析，不告警。`--log-level` 含 `critical`。optix `--benchmark-policy` 仍是 `ais_bench` / `vllm_benchmark`。`--device` 为正式入口，throughput-optimizer 同时接受 `--devices`（不告警）。`--remote-source` / `--performance-model` 仅补 metavar。

布尔：help 用 `[default: off]` / `[default: on]`，不再写 True/False。`--enable-redundant-experts` 等说明改为陈述句，去掉 “When this flag is True”。

多值 metavar：`nargs=+` 显示 `<N> [<N> ...]`，`nargs=*` 显示 `[<N> ...]`，由 `SpecHelpFormatter` 按 nargs 展开，避免 Usage 行重复包一层。

**第四类：帮助与版本体例（不是改某个 dest，但每个命令都改了输出）**

| 项 | 具体改动 |
|:---|:---|
| 帮助段落 | `SpecArgumentParser.format_help()` 固定输出 Description / Usage / Commands（有子命令时）/ Required arguments / Optional arguments / Examples；有落盘时加 Output |
| 必填/可选 | 靠段落分组，禁止 `<Required>`、`[Mandatory]` 等行内标签 |
| 默认值 | 行尾 `[default: xxx]`；无默认不写；禁止 `(default: None)`。枚举默认展示原值，如 `[default: W8A8_DYNAMIC]` |
| metavar | `<N>` / `<FILE>` / `<DIR>` / `<FLOAT>` / `<NAME>` / `<RANGE>` / `{a,b,c}` |
| `--help` 文案 | `Show help message.` |
| 示例 | 每个公开子命令至少 1 条可运行命令，推荐带 `#` 注释 |
| 版本 | 见第二类 `-V, --version` |

#### 1.4.2 旧接口是否还能用

能用。旧参数不删除，只是降为隐藏别名：

1. **能解析**：`--chrome-trace out.json` 与 `--chrome-trace-file out.json` 效果相同。
2. **stderr 打一次弃用提示**：`WARNING: --chrome-trace is deprecated; use --chrome-trace-file instead.`
3. **不出现在 `--help`**：新用户只看到正式名。

存量脚本、CI 与现有 UT/ST 不必先改参数名。内部变量名（dest）也不改。

---

## 2. Use Case Analysis (用例分析)

| 用例 | 行为 | DFX |
|:---|:---|:---|
| 新用户查看帮助 | `msmodeling --help` 与各子命令 `--help` 只展示标准名、metavar、默认值与至少 1 条可运行示例 | 可学习、可被 Agent 解析 |
| 查询版本 | 任意公开入口 `--version` / `-V` 打印 Logo、`msmodeling {ver} ({git})`、版权与 Mulan PSL v2 | 排障可确认安装版本 |
| 调节日志 | `--log-level {debug,info,warning,error,critical}` 默认 info；`--verbose/-v` 与 `--debug` 等价 debug，`--quiet/-q` 等价 error；显式 `--log-level` 优先 | 与规范 4.2.3.1 冲突裁决一致 |
| 新脚本使用标准名 | `--tp-size 8`、`--chrome-trace-file out.json`、`--disagg`、`--load-breakpoint` | 正式接口无 snake_case、无多字符短选项 |
| 存量脚本 | `--chrome-trace`、`--load_breakpoint`、`-lb`、`--ttft-limits` 仍可解析 | 兼容性；stderr 引导迁移 |
| 模型标识 | 位置参数 `model_id` 与 `--model-path` / `--model-id` 等价；缺一不可 | 对齐词表 `--model-path`，不强制删除位置参数 |
| 多硬件寻优 | `throughput-optimizer` 以 `--device` 为正式入口，同时接受 `--devices` | 不把 `--device` 标成弃用 |
| 适配器导出 | `model-adapter` 子命令用 `--output-file/-o` 写 JSON/YAML；`--output` 为隐藏别名 | `-o` 语义符合词表 |

使用约束：

- `--device` 取值是已注册 DeviceProfile 名，不是 `cpu`/`npu`。
- 量化等枚举正式值为 `W8A8_DYNAMIC`；`w8a8-dynamic` 仍可解析。
- `--compilation-config` help 展示内部 snake_case 选项名。

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
3. **dest 不变**：例如公开 `--tp-size` 的 dest 仍是 `tp_size`。
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

- **描述**：help 展示原取值；kebab 拼写仍可解析，不告警。
- **存储**：枚举返回成员实例；`compilation-config` 等 token 可 `store_canonical="snake"` 以对接内部配置键。

##### 公开参数映射（节选）

| 概念 | 正式接口 | 隐藏别名（仍可解析） | dest（不变） |
|:---|:---|:---|:---|
| 张量并行 | `--tp-size` | （不改正式名） | `tp_size` |
| 分离部署 | `--disagg` | （不改正式名） | `disagg` |
| Trace | `--chrome-trace-file` | `--chrome-trace` | `chrome_trace` |
| Profiling 库 | `--profiling-database-path` | `--profiling-database` | `profiling_database` |
| 模型路径 | `--model-path` / `--model-id` + 位置参数 | `--model_id` | `model_id` |
| 输出文件 | `-o, --output-file` | `--output` | `output` |
| 断点续跑 | `--load-breakpoint` | `--load_breakpoint`、`-lb` | `load_breakpoint` |
| 反向开关 | `--no-repetition`（对偶 `--repetition`） | `--disable-repetition` | `disable_repetition` |
| TTFT 约束 | `--ttft-limit` | `--ttft-limits` | `ttft_limits` |

optix 引擎/benchmark 取值是注册表固定名（`ais_bench`、`vllm_benchmark`），help 与文档保持该写法，不改成 kebab-case。

#### 3.4.3 Usage Instructions (使用说明)

标准写法：

```bash
msmodeling inference text-generate Qwen/Qwen3-32B \
  --num-queries 1 --query-length 128 --device TEST_DEVICE \
  --tp-size 8 --chrome-trace-file trace.json

msmodeling inference throughput-optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE --num-devices 8 \
  --input-length 1024 --output-length 512 --disagg

msmodeling inference model-adapter doctor --model-id Qwen/Qwen3-32B -o doctor.json
msmodeling optix -e vllm -b ais_bench --config ./config.toml
```

约束：

- `-v` 只表示 verbose，版本只用 `-V`。
- 旧写法可用，但会告警，且不会出现在 `--help`。
- `throughput-optimizer` 的 `--jobs/-j` 表示寻优进程并发，不是模型并行度。

---

## 4. Test Design (测试设计)

本节面向**测试同事的手工 / ST 验收**，说明验什么、怎么验、怎样算通过。不要求测试改自动化用例。开发侧回归与 CI 另行覆盖。

验收重点四件事：公开命令能跑；`--help` / `--version` 符合规范；旧参数仍可用但会提示；同一组业务参数下新旧写法结果一致。

### 4.1 验收范围

- 公开入口：`msmodeling`、`inference text-generate` / `throughput-optimizer` / `model-adapter` / `video-generate`、`optix`
- `--help`、`--version/-V`、日志开关、正式参数名与取值
- 旧参数名、旧枚举值仍能跑通，stderr 有弃用提示
- 新旧写法仿真结果一致（同模型、同 device、同长度）
- `--device` 仍是 DeviceProfile 名（如 `TEST_DEVICE`）

中英文 user guide / quick start 已按正式名更新：文档抽测用新名；旧名只做兼容抽测。

### 4.2 环境

在仓库根目录、依赖已安装、`PYTHONPATH` 已指向仓库根的环境中执行。下面两组入口等价，任选：

```bash
msmodeling inference text-generate --help
python -m cli.inference.text_generate --help
```

optix：`msmodeling optix --help`。若因缺少 `pydantic_settings` 直接报错，记为环境问题，不要据此判定本需求失败。

### 4.3 验收步骤

#### A. 帮助与版本（每个公开入口各做一遍）

对下列命令执行 `--help`，并对顶层及至少一个子命令执行 `-V` / `--version`：

- `msmodeling --help`、`msmodeling -V`
- `python -m cli.inference.text_generate --help`
- `python -m cli.inference.throughput_optimizer --help`
- `python -m cli.inference.video_generate --help`
- `python -m cli.inference.model_adapter doctor --help`（`verify` / `export-evidence` 抽一个即可）
- `msmodeling optix --help`

**通过标准：**

1. `--help` 能看到 Description、Usage、Examples；有子命令时能看到 Commands；text-generate 能看到必选 / 可选分段。
2. `-V` / `--version` 成功，输出含 MindStudio / msmodeling 与 Mulan PSL v2。**不要用 `-v` 查版本**（`-v` 是更详细日志）。
3. help 中有 `--log-level {debug,info,warning,error,critical}`，以及 `-v`、`-q`、`--debug`、`--log-file`。
4. 量化正式取值仍是 `W8A8_DYNAMIC`、`DISABLED` 等；kebab 拼写也能解析。
5. 布尔用开关语义（如 `[default: off]`），不要用 True/False、yes/no。
6. 下列旧名**不应作为正式选项出现在 `--help`**：`--chrome-trace`（没有 `-file`）、`--load_breakpoint`、`-lb`、`--output`（adapter）、`--ttft-limits`。对应正式名分别是 `--chrome-trace-file`、`--load-breakpoint`、`--output-file`、`--ttft-limit`。`--tp-size`、`--disagg` **应**出现在 help 中。
7. optix help 里测评工具取值仍是 **`ais_bench`、`vllm_benchmark`**（固定名称，不要验收成 `ais-bench`）。
8. throughput-optimizer help 能看到 `--jobs` / `-j`（寻优进程并发，不是模型 TP）。

#### B. 新写法功能抽测

用正式参数跑最小可运行场景即可，不要求完整性能对标：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
  --num-queries 1 --query-length 128 --device TEST_DEVICE \
  --tp-size 1 --log-level info

python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE --num-devices 2 \
  --input-length 128 --output-length 16 \
  --tp-sizes 1 2 --disagg --jobs 2
```

有实测环境时再抽测：`msmodeling optix -e vllm -b ais_bench --config ./config.toml`。没有实测环境时，确认 optix `--help` 与 `-e` / `-b` / `-c` 说明正确即可。

**通过标准：** 命令能解析并进入原有流程；text-generate 能打出性能表；optimizer 能开始搜索或报出与改名前同类的配置错误。加上 `--chrome-trace-file trace.json` 时应生成文件。

#### C. 旧写法兼容抽测（必做）

同一组业务参数分别用旧名、新名各跑一遍，结果应一致。

| 场景 | 旧写法（应仍可用） | 正式写法 | 期望 |
|:---|:---|:---|:---|
| Trace | `--chrome-trace out.json` | `--chrome-trace-file out.json` | 都能写出文件；旧名 stderr 提示 deprecated |
| TTFT | `--ttft-limits 2000` | `--ttft-limit 2000` | 同上 |
| adapter 输出 | `doctor ... --output a.json` | `-o a.json` 或 `--output-file a.json` | 都能写出报告 |
| optix 断点 | `--load_breakpoint` 或 `-lb` | `--load-breakpoint` | 都能解析；旧名有提示；`--help` 里看不到 `-lb` |
| 量化 kebab | `--quantize-linear-action w8a8-dynamic` | `--quantize-linear-action W8A8_DYNAMIC` | 都能解析；help 展示大写正式值 |

互斥约束与改名前相同（例如不要把 `--disagg` 和 PD 配比优化一起用）。

**通过标准：** 旧命令不会被当成未知参数直接失败；功能与新名等价；提示出现在 stderr，不影响正常结果输出。

#### D. 日志开关抽测

| 操作 | 期望 |
|:---|:---|
| 不指定日志相关参数 | 默认 info（比改前默认 error 日志更多，属预期） |
| `--verbose` 或 `-v` 或 `--debug` | 更详细，等价 debug |
| `--quiet` 或 `-q` | 更少，等价 error |
| 同时写 `--log-level warning` 和 `-v` | 以 `--log-level` 为准（warning） |
| `--log-level critical` | 应成功，级别为 critical |

#### E. 文档抽测

中英文 TensorCast / 吞吐优化 / 快速入门中的示例应使用正式名。OptiX 指南里 `-b` 取值仍是 `ais_bench` / `vllm_benchmark`。

### 4.4 判定为不通过的典型现象

- `--help` 仍把 `-lb`、`--load_breakpoint`、无 `-file` 的 `--chrome-trace` 列成正式选项。
- `--tp-size 2` 报 unrecognized arguments。
- 排除拉模型失败、环境差异后，新旧写法指标仍明显不一致。
- `-v` 打印版本而不是详细日志。
- optix help 把测评工具写成 `ais-bench` / `vllm-benchmark`。
- `--device cpu` 被当成合法设备类型。

### 4.5 自动化（可选）

开发已有 CLI 规范回归。测试环境若要复跑：

```bash
python -m pytest tests/regression/cli/test_spec_cli.py tests/regression/cli/test_export.py
```

全量结论以 CI 为准，不要用本机 skip 的 optix 用例代替门禁。

## 5. Drawbacks and Risks (缺点和风险)

| 风险 | 影响 | 应对 |
|:---|:---|:---|
| dest 与公开选项名不一致 | 二次开发若按 dest 猜 CLI 会出错 | RFC 与 help 只承诺 option string；dest 视为内部 |
| Web UI / Skill 仍拼旧路径名 | `--chrome-trace` 等会告警 | 别名保留；后续单独改命令拼装 |
| `--device` 与规范词表不完全同义 | 跨工具用户可能误解为 cpu/npu | help 写明 DeviceProfile；本 RFC 明确不改语义 |
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
| `cli/inference/text_generate.py` | 路径后缀、`--no-repetition`、log/version；并行度正式名仍为 `--tp-size` |
| `cli/inference/throughput_optimizer.py` | `--ttft-limit`、`--jobs/-j`、路径后缀 |
| `cli/inference/model_adapter.py` | 子命令 help、`--output-file/-o` |
| `cli/inference/video_generate.py` | `--num-devices`、log/version；`--ulysses-size` 保持正式名 |
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
