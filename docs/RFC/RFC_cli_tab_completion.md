# RFC: Static Shell Tab Completion for msmodeling CLIs / msmodeling CLI 静态 Shell Tab 补全


## Metadata (元数据)
| Item / 项目 | Content / 内容 |
| :--- | :--- |
| **Status / 状态** | Accepted (已批准) |
| **Author(s) / 作者** | msmodeling contributors |
| **Creation Date / 创建日期** | 2026-04-28 |
| **Related Links / 相关链接** | `cli/completion.py`, `[project.scripts]` → `msmodeling-tab` / `msmodeling-completion` in `pyproject.toml`, README “Tab completion” 段落 |

---

## 1. Problem Statement（概述）

用户在命令行频繁调用推理与服务仿真入口（`python -m cli.inference.*`、`python serving_cast/main.py`）。若依赖动态解释器（如 Tab 时启动 Python + 导入 PyTorch/transformers），单次 Tab 延迟高、体验差。需要提供 **低开销、易分发** 的 Tab 补全，并在安装包后通过 **单行 shell 加载命令** 启用当前终端补全。

本 RFC 汇总当前已实现的 **静态 Bash/Zsh Tab 补全设计**：数据来源、覆盖入口、Shell 集成方式、安装 CLI 行为及维护约束。

### 1.1 Related files & artifacts / 涉及与修改的文件及产物

#### 1.1.1 仓库内需跟踪的典型变更（相对仅增加 Tab 能力）

| Path / 路径 | Role / 作用 |
| :--- | :--- |
| `cli/completion.py` | **核心实现**：定义 `MODULES`、`OPTIONS`、`COMMON_OPTIONS`；`render_bash_completion()` / `render_zsh_completion()` 输出 Shell 源码；实现 `main()`、`print`、仅写安装前缀的 `install`、以及显式 `cleanup-legacy` 迁移命令。**默认路径不写 `~/.bashrc` / `~/.zshrc` 等任何用户 shell 启动文件**。 |
| `pyproject.toml` | **`[build-system]`**：`setuptools.build_meta`，使 `pip install .` 可构建 wheel。**`[project.version]`**：满足 PEP 621。**`[project.scripts]`**：注册推荐短入口 `msmodeling-tab → cli.completion:main`，并保留兼容入口 `msmodeling-completion → cli.completion:main`。**`[tool.setuptools.packages.find]`**：包含 `cli*` 等，保证 `pip install` 后能 `import cli.completion`。 |
| `README.md` | 用户使用说明：`cd /home/yinqian/msmodeling/msmodeling && pip install .`、官方推荐 `. <(msmodeling-tab)`、静态补全选型理由、`python3 -m cli.inference.<Tab>` 与长选项覆盖说明。 |
| `.gitignore`（建议） | 可选增加 `build/`、`dist/`、`*.egg-info/`，避免本地 **`pip install`/`python -m build`** 产生的目录被误提交；与 Tab 逻辑无运行时耦合。 |

CLI 解析逻辑仍位于 **`cli/inference/text_generate.py`**、**`video_generate.py`**、**`throughput_optimizer.py`**、**`serving_cast/main.py`** 及 **`cli/utils.py`（通用参数）**；这些是 argparse **真源**，`cli/completion.py` 仅为镜像列表。

#### 1.1.2 运行时生成、通常不入版本库的路径

| Path / 路径 | Role / 作用 |
| :--- | :--- |
| `<prefix>/share/bash-completion/completions/msmodeling-python` | 可选 `msmodeling-completion install` 生成的 Bash 补全正文；路径位于 `sys.prefix`、`--prefix` 或 `--user` 对应的 Python 安装前缀内。 |
| `<prefix>/share/zsh/site-functions/_msmodeling-python` | 可选 `msmodeling-completion install` 生成的 Zsh 补全脚本；内容自包含并通过 `bashcompinit` 复用 Bash 补全逻辑。 |
| `~/.bashrc` / `~/.zshrc` 等 rc 文件 | **不会被默认或主路径写入**。仅 `cleanup-legacy` 在用户显式调用时删除旧版本留下的标记块。 |

#### 1.1.3 包安装后进入环境的内容

- **`msmodeling-tab`** 推荐短入口，以及兼容入口 **`msmodeling-completion`**（由 setuptools 安装到当前环境 **`bin/`**，如 venv）。
- **`cli`** 命名空间下的模块随 wheel 安装至 **`site-packages`**，运行时由上述入口调用 **`cli.completion:main`** 输出或可选写入补全脚本。

---

## 2. Proposals（方案设计）

### 2.1 Proposed solution（推荐方案）

#### 2.1.1 总体架构

采用 **静态生成 Bash 补全脚本**：

- Python 模块 `cli/completion.py` 内维护 **`MODULES`（`-m` 后继模块名白名单）**与 **`OPTIONS`（按入口划分的长选项表）**，运行时生成 Bash 源码字符串。
- 生成的脚本通过 **`complete -o default -F _msmodeling_complete python python3`** 将补全函数挂到 **`python` / `python3`**，从而在一条命令内同时覆盖「`-m` 模块路径」「脚本路径」「`--` 长选项」三种用法。
- **Zsh**：输出自包含脚本，在内开启 `compinit` / `bashcompinit` 并复用同一份静态补全逻辑；因此 `. <(msmodeling-tab)` 在 bash/zsh 中均可作为官方一步。

Tab 按下时 **不启动 Python**、不加载深度学习栈；仅 Bash 内置 `compgen` 与表中字符串匹配。

#### 2.1.1a Control flow — Tab press / 按 Tab 时的逻辑

Bash 在收到补全请求时调用 **`_msmodeling_complete`**。变量约定：`COMP_WORDS` 为已分词命令行，`COMP_CWORD` 为当前词下标，`cur` 为当前正在补全的片段。

```mermaid
flowchart TD
  A[_msmodeling_complete] --> B{上一词是否为 -m?}
  B -->|是| C[compgen -W MODULES 对 cur]
  B -->|否| D{cur 是否匹配 cli.inference.*?}
  D -->|是| C
  D -->|否| E[target = _msmodeling_completion_target]
  E --> F{target 非空 且 cur 以 -- 开头?}
  F -->|否| G[COMPREPLY 空或交给 default]
  F -->|是| H[compgen -W OPTIONS[target] 对 cur]
```

**`_msmodeling_completion_target`** 从 **`COMP_WORDS[0..COMP_CWORD-1]`** 扫描（不读当前未完成词），逻辑要点：

1. 若存在 **`word == -m`** 且下一词 **`next`** 为完整模块名（如 `cli.inference.text_generate`），则映射到对应逻辑 ID 并返回。
2. 否则按 **`word`** 是否匹配某段 **脚本路径**（如 `.../text_generate.py`、`serving_cast/main.py`）判定逻辑 ID。
3. 若无法识别，返回空，后续长选项分支不生效。

长选项分支要求 **`cur` 以 `--` 开头**，避免误对位置参数 `model_id` 等做选项补全。

#### 2.1.1b Control flow — msmodeling-tab / msmodeling-completion CLI

| 命令 | 逻辑 |
| :--- | :--- |
| `. <(msmodeling-tab)` | 官方推荐一步：`msmodeling-tab` 无子命令时输出自包含补全脚本，`.` 将脚本加载到当前 shell。该路径不写磁盘、不改 `$HOME`。 |
| `msmodeling-completion print` | **`sys.stdout`** 输出 **`render_bash_completion()`** 或 zsh 自包含脚本，不写磁盘、不改 rc；用于调试或等价的 `eval "$(msmodeling-completion print)"` 场景。 |
| `msmodeling-completion install` | 将渲染脚本写入 Python 安装前缀下的 `share/bash-completion/completions/` 与 `share/zsh/site-functions/`。默认前缀为 `sys.prefix`；`--user` 使用 `site.getuserbase()`；`--prefix` 可显式指定。**不写任何 shell rc 文件**。 |
| `msmodeling-completion cleanup-legacy [--dry-run]` | 兼容迁移命令：仅在用户显式调用时删除旧版本写入的 `# >>> msmodeling completion >>>` 标记块和旧数据目录。 |

`main()` 通过 **`build_parser()`** 子命令分发；**`cli.completion:main`** 无参时等价于 `print --shell auto`，这使 `. <(msmodeling-tab)` 足够短。

#### 2.1.2 入口与映射（`_msmodeling_completion_target`）

补全脚本需判断「当前命令行意图对应四类逻辑入口之一」，才在长选项模式下提供 `--` 列表：

| 逻辑 ID | 识别方式 |
| :--- | :--- |
| `text_generate` | `-m` 后为 `cli.inference.text_generate`；或 argv 中出现 `*/cli/inference/text_generate.py`、`cli/inference/text_generate.py`、`text_generate.py` |
| `video_generate` | `-m` 后为 `cli.inference.video_generate`；或路径匹配 `video_generate.py` |
| `throughput_optimizer` | `-m` 后为 `cli.inference.throughput_optimizer`；或路径匹配 `throughput_optimizer.py` |
| `serving_cast_main` | `-m` 后为 `serving_cast.main`；或路径匹配 `*/serving_cast/main.py`、`serving_cast/main.py` |

四类对应 README 所列四个用户入口：`text_generate`、`video_generate`、`throughput_optimizer`、`serving_cast/main`。

#### 2.1.3 模块路径补全（`-m` 与前缀）

- **条件 A**：光标前一词为 `-m`，则对当前词按白名单 **`MODULES`** 补全：
  - `cli.inference.text_generate`
  - `cli.inference.video_generate`
  - `cli.inference.throughput_optimizer`
- **条件 B**：当前补齐片段匹配 `cli.inference.*`（case 分支），同样用 **`MODULES`** 生成候选。

#### 2.1.4 长选项补全（`OPTIONS`）

仅当：

1. 已由 `_msmodeling_completion_target` 解析出 **非空 target**，且  
2. 当前词 **以 `--` 开头**  

时，按 target 选取对应元组中的字符串列表做 `compgen`。各入口包含的选项与 `cli/inference/*.py`、`serving_cast/main.py` 内 **argparse 长选项**对齐；维护责任见 2.1.7。

要点：

- **`text_generate` / `throughput_optimizer`** 中与「通用 CLI」重叠的部分通过 **`COMMON_OPTIONS`**（`--device`、`--num-devices`、`--reserved-memory-gb`、`--log-level`）复用表述（与 `cli/utils.py` 中 common 组一致）。
- 各表均显式包含 **`--help`**。
- **不**在长选项补全自动补 **位置参数 / 文件路径**（除 `--` 外依赖 `complete -o default` 的 fallback，行为以 Shell 为准）。

#### 2.1.5 安装与用户可见 CLI

`pyproject.toml` 注册：

```text
msmodeling-tab = "cli.completion:main"
msmodeling-completion = "cli.completion:main"
```

子命令：

| 子命令 | 作用 |
| :--- | :--- |
| 无子命令 | 输出当前 shell 的自包含补全脚本；官方推荐 `. <(msmodeling-tab)`。 |
| `print` | 将生成的 Bash 或 Zsh 脚本 **打印到 stdout**，便于人工重定向或调试。参数 `--shell auto|bash|zsh`。 |
| `install` | 将补全脚本写入 **`<prefix>/share/bash-completion/completions/msmodeling-python`** 和 **`<prefix>/share/zsh/site-functions/_msmodeling-python`**；只写安装前缀，不修改 rc。 |
| `cleanup-legacy` | 删除旧版本 `install` 写入的 rc 标记块和 `~/.local/share/msmodeling` 旧数据目录；默认实际删除，`--dry-run` 仅预览。 |

安装后用户在当前会话执行 **`. <(msmodeling-tab)`**，使 `complete` 注册生效。每个新 shell 都需要加载一次，除非用户自行选择把该行加入自己的 rc 文件；项目不会自动写入。

#### 2.1.6 打包与依赖

- **无**额外 PyPI 依赖专用于补全逻辑；仅使用标准库 `argparse`、`pathlib` 等。
- 与主包一同通过 `pip install .` 安装，entry point 随环境 `PATH` 可用。

#### 2.1.7 维护与一致性

- **`OPTIONS` / `MODULES` 与 argparse 不同步**将导致补全多出或漏项；CLI 变更时需同步改 `cli/completion.py`。用户升级后重新执行 **`. <(msmodeling-tab)`** 即可在当前 shell 使用新表。
- 可选后续增强：由 CI 或脚本从 argparse 自动生成列表（本 RFC 描述之当前实现为 **手写元组**）。

---

### 2.2 Alternatives Considered（替代方案）

| 方案 | 说明 |
| :--- | :--- |
| **argcomplete** | 与 argparse 同源、维护量低，但 Tab 需走 Python `_ARGCOMPLETE` 路径；且本仓库 CLI 多在 import 阶段拉起重依赖，难以保证「每次 Tab 极低延迟」。 |
| **仅按脚本文件名补全、不挂 `python3`** | 可为 `text_generate.py` 单独 `complete`，但无法实现统一的 `python3 -m cli.inference.*` 体验。 |
| **`data_files` 安装到 `share/bash-completion`** | 依赖发行版 / bash-completion 是否在用户活跃环境中加载该前缀；当前实现提供可选 prefix 内 `install`，但主路径仍是显式 `. <(msmodeling-tab)`，以适配 venv / `pip --user`。 |

---

### 2.3 Pros and Cons（方案分析）

| | 内容 |
| :--- | :--- |
| **优点** | Tab 延迟低（纯 Shell）；无额外 PyPI 依赖；实现集中在一模块，易审计；覆盖 `python`/`python3` 与 `-m`、脚本路径多种调用习惯。 |
| **缺点** | 长选项与 argparse **双份维护**；每个当前 shell 仍需 `. <(msmodeling-tab)` 这一步加载补全；对 fish 等 Shell 未在本实现中覆盖。 |
| **主推方案局限** | 不保证与系统已有 **`python3` 补全**同时存在时的优先级；若其他包也 `complete` 覆盖 `python3`，行为以最后加载为准。 |

---

## 3. Plan（实施计划）

- **当前状态**：`cli/completion.py`、`msmodeling-tab`、兼容入口 `msmodeling-completion`、README 说明已落地；本 RFC 作为设计归档与评审基线。
- **测试建议**：
  - 在 Bash 下：执行 `. <(msmodeling-tab)` 后 `complete -p python3`，确认含 `_msmodeling_complete`。
  - 手工键入：`python3 -m cli.inference.te<Tab>`、`python3 -m cli.inference.text_generate --de<Tab>`、`python3 serving_cast/main.py --inst<Tab>`。
- **后续**：评估 **生成器** 从 argparse 导出选项；评估系统级 prefix 安装在不同发行版 bash-completion / zsh `fpath` 中的自动加载表现。
