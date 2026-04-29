# RFC: msmodeling CLI 静态 Tab 补全

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | 已批准 |
| **作者** | YinQian |
| **创建日期** | 2026-04-28 |
| **相关链接** | `cli/completion.py`、`pyproject.toml`、README “Tab completion” 段落 |

---

## 1. 概述

本提案旨在为 msmodeling 常用命令提供低延迟 Tab 补全，覆盖 `python -m cli.inference.*`、`python3 -m cli.inference.*` 以及 `python serving_cast/main.py` 等调用方式。

补全采用静态 Bash/Zsh/PowerShell 脚本实现。按 Tab 时只在 shell 内匹配固定列表，不启动 Python，也不导入 PyTorch、transformers 等重型依赖。

安装包后，用户在当前 shell 中执行一行命令即可启用补全。

Bash/Zsh：

```bash
. <(msmodeling-tab)
```

PowerShell：

```powershell
msmodeling-tab | iex
```

如果 PowerShell 自动识别失败，可使用显式写法：`msmodeling-tab --shell powershell | iex`。

项目不会自动修改用户的 `~/.bashrc`、`~/.zshrc`、`~/.profile`、PowerShell profile 等 shell 配置文件。

## 2. 详细设计

**保持现有 CLI 调用方式不变**：用户仍然使用 `python -m cli.inference...`、`python serving_cast/main.py` 等原有命令，只增加 shell 侧补全能力。

- 在 `cli/completion.py` 中维护静态模块列表和长选项列表。
- 在 `pyproject.toml` 中新增短入口 `msmodeling-tab`，保留兼容入口 `msmodeling-completion`。
- 在 README 中说明 Bash/Zsh/PowerShell 的安装、启用和验证方式。

### 2.1 实现方案

#### 2.1.1 静态补全脚本

`cli/completion.py` 负责生成 Bash/Zsh/PowerShell 补全脚本，核心数据包括：

1. `MODULES`：用于补全 `python3 -m cli.inference.<Tab>`。
2. `OPTIONS`：用于补全各入口的 `--` 长选项。
3. `COMMON_OPTIONS`：复用通用参数，如 `--device`、`--log-level`。

生成的 Bash 脚本通过以下方式挂到 `python` 和 `python3`：

```bash
complete -o default -F _msmodeling_complete python python3
```

Zsh 通过 `bashcompinit` 复用同一份静态补全逻辑。PowerShell 通过 `Register-ArgumentCompleter -Native` 为 `python` 和 `python3` 注册补全。

#### 2.1.2 用户启用方式

推荐流程：

Bash/Zsh：

```bash
cd /home/yinqian/msmodeling/msmodeling && pip install .
. <(msmodeling-tab)
```

PowerShell：

```powershell
cd /home/yinqian/msmodeling/msmodeling; pip install .
msmodeling-tab | iex
```

第二行的作用是把补全函数加载到当前 shell。`pip install .` 只能安装命令，不能把补全函数注入到已经打开的父 shell，因此仍需要这一行 shell 侧加载命令。若 PowerShell 自动识别失败，使用 `msmodeling-tab --shell powershell | iex`。

#### 2.1.3 覆盖范围

当前补全覆盖：

- `python3 -m cli.inference.<Tab>` 模块名补全
- `python -m cli.inference.text_generate --<Tab>` 长选项补全
- `python -m cli.inference.video_generate --<Tab>` 长选项补全
- `python -m cli.inference.throughput_optimizer --<Tab>` 长选项补全
- `python serving_cast/main.py --<Tab>` 长选项补全
- `python cli/inference/throughput_optimizer.py --<Tab>` 等脚本路径形式

示例验证：

```bash
python3 -m cli.inference.te<Tab>
python -m cli.inference.text_generate --de<Tab>
python serving_cast/main.py --in<Tab>
python cli/inference/throughput_optimizer.py --tp<Tab>
```

#### 2.1.4 不修改用户 shell 配置

默认方案只把补全脚本输出到当前 shell，不写用户目录下的 shell 启动文件。

`msmodeling-completion install` 仅作为可选命令，用于把补全脚本写入 Python 安装前缀，例如：

- `<prefix>/share/bash-completion/completions/msmodeling-python`
- `<prefix>/share/zsh/site-functions/_msmodeling-python`
- `<prefix>/share/powershell/Completions/msmodeling.ps1`



### 2.2 替代方案

**使用 argcomplete**

argcomplete 可以和 argparse 共享参数定义，但每次 Tab 可能启动 Python，并触发 PyTorch、transformers 等重依赖导入，延迟不可控。

**只补全脚本文件名**

这种方式实现简单，但不能覆盖 `python3 -m cli.inference.<Tab>`，也不符合当前用户常用调用方式。

**自动写入 shell rc 文件**

这样新终端可以自动生效，但会修改用户个人配置文件，不符合本方案约束。

### 2.3 方案分析

**选择当前方案的原因：**

1. **低延迟**：Tab 时只执行 shell 字符串匹配，不启动 Python。
2. **兼容性好**：不改变现有 CLI 调用方式。
3. **安全可控**：默认不修改 `~/.bashrc`、`~/.zshrc`、PowerShell profile 等用户配置。
4. **使用简单**：Bash/Zsh 执行 `. <(msmodeling-tab)`，PowerShell 执行 `msmodeling-tab | iex`。

**当前限制：**

1. `MODULES` 和 `OPTIONS` 是静态列表，CLI 参数变化时需要同步更新。
2. 每个新 shell 都需要重新加载补全脚本，除非用户自己选择写入 shell 启动文件。
3. 当前覆盖 Bash/Zsh/PowerShell，暂不覆盖 fish。

## 3. 实施计划

### 已完成功能开发

- [x] 在 `cli/completion.py` 中实现静态 Bash/Zsh/PowerShell 补全生成逻辑
- [x] 支持 `python` / `python3` 的模块名和长选项补全
- [x] 新增 `msmodeling-tab` 短入口
- [x] 保留 `msmodeling-completion` 兼容入口
- [x] 默认不写用户 shell 配置文件
- [x] README 中增加安装、启用和验证说明
- [x] 支持 PowerShell 通过 `Register-ArgumentCompleter` 启用补全

### 后续优化

- [ ] 增加脚本校验，检查 `OPTIONS` 是否与 argparse 参数保持一致
- [ ] 评估从 argparse 自动生成静态补全表
- [ ] 评估 fish shell 补全支持
