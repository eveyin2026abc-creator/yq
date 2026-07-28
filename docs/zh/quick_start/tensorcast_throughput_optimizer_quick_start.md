# msModeling快速入门

<br>

## 1. 概述

msModeling 提供模型推理性能仿真与服务化性能仿真能力。本文档面向首次体验模型推理性能仿真和服务化性能仿真的用户，带您从环境检查开始，依次跑通 LLM 文本生成仿真与服务化性能仿真，快速理解工具的核心输入、输出与适用场景。

### 1.1 前言

**体验地图（核心操作约 10 分钟）**

> **执行顺序建议**：步骤 1 为基础环境检查；步骤 2 用于体验模型推理性能仿真；步骤 3 用于体验服务化性能仿真。

| 步骤 | 环节 | 核心模块 | 参考操作耗时 | 建议原理学习 |
| :---: | :---: | :--- | :---: | :---: |
| **1** | **环境准备** | `msModeling` | 2 分钟 | 5 分钟 |
| **2** | **模型推理性能仿真** | `TensorCast` | 1 分钟 | 10 分钟 |
| **3** | **服务化性能仿真** | `Throughput Optimizer` | 2 分钟 | 15 分钟 |

### 1.2 环境准备

👉 **【重要】请先按《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》完成环境安装和配置。**

> [!CAUTION]注意
> 本文档默认在 msModeling 仓库根目录执行命令。若在其他目录运行，请先设置 `PYTHONPATH`，否则可能出现 `No module named cli`、`No module named tensor_cast` 等模块导入错误。

## 2. 操作步骤

> [!NOTE]说明
> 后续命令均支持直接复制执行。建议先使用 `TEST_DEVICE` 跑通流程，确认输出符合预期后，再替换为实际目标硬件设备和业务模型。

### 2.1【环境】确认运行环境

开始体验前，请先完成《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》中的环境搭建，包括克隆仓库、创建虚拟环境、安装依赖以及配置 `PYTHONPATH`。

本文后续命令默认在 msModeling 仓库根目录执行。若不在仓库根目录运行，请先设置：

```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
```

模型推理性能仿真运行 Hugging Face 模型时需要读取模型配置。如果当前环境无法直接访问 Hugging Face，可设置镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

可使用以下命令快速确认命令行入口可访问：

```bash
python -m cli.inference.text_generate --help
python -m cli.inference.throughput_optimizer --help
```

若上述命令无法正常输出帮助信息，请优先检查虚拟环境是否已激活、依赖是否安装完成，以及 `PYTHONPATH` 是否指向 msModeling 仓库根目录。

### 2.2【可选】启用 Tab 补全

在 Linux Bash 环境中，可启用静态 Tab 补全，补全 `python`、`python3` 和 `msmodeling` 命令下的常用 msModeling 模块名、长选项以及部分枚举参数值。补全时只执行 shell 字符串匹配，不会启动 Python 或加载模型依赖。

如果已通过 `pip install -e .`、`pip install .` 或 `uv sync` 安装当前项目，可执行：

```bash
msmodeling -tab
```

`msmodeling -tab` 是 `msmodeling --enable-tab-completion` 的短别名。该命令会生成静态补全脚本到 `$HOME/.local/share/msmodeling/completion.bash`，并在 `~/.bashrc` 中加入由 msModeling 管理的加载块。完成后重新进入 Bash，当前终端和新终端都可以使用补全。

也可以使用完整命令启用：

```bash
msmodeling --enable-tab-completion
```

如果不想安装 `msmodeling` entry point，也可以在仓库根目录通过 `PYTHONPATH` 使用模块入口启用：

```bash
cd /path/to/msmodeling
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
python -m cli.main -tab
```

新增命令说明如下：

| 命令或参数 | 作用 | 说明 |
| --- | --- | --- |
| `msmodeling -tab` | 启用静态 Tab 补全 | 推荐使用的短命令，等价于 `msmodeling --enable-tab-completion`。 |
| `msmodeling --enable-tab-completion` | 启用静态 Tab 补全 | 生成补全脚本并写入 `~/.bashrc` 中由 msModeling 管理的加载块。 |
| `python -m cli.main -tab` | 免安装 entry point 启用补全 | 适用于未安装 `msmodeling` 命令、但已通过 `PYTHONPATH` 指向仓库源码的场景。 |
| `msmodeling --disable-tab-completion` | 禁用 Tab 补全 | 移除 `~/.bashrc` 中由 msModeling 管理的加载块，并删除由 `msmodeling -tab` 生成的 `$HOME/.local/share/msmodeling/completion.bash`。 |

启用后可通过以下方式验证模块名、长选项和参数值补全：

```bash
python -m cli.inference.te<Tab>
python -m cli.inference.text_generate Qwen/Qwen3-32B --num<Tab>
python -m cli.inference.text_generate Qwen/Qwen3-32B --device AT<Tab>
python -m cli.inference.text_generate Qwen/Qwen3-32B --quantize-linear-action W8A8_<Tab>
msmodeling inference text-generate Qwen/Qwen3-32B --remote-source <Tab>
```

当前支持参数值补全的参数如下：

| 参数 | 补全内容示例 | 说明 |
| --- | --- | --- |
| `--device` | `TEST_DEVICE`、`ATLAS_800_A2_376T_64G` | 目标硬件设备画像。 |
| `--log-level` | `debug`、`info`、`warning`、`error`、`critical` | 日志级别。 |
| `--remote-source` | `huggingface`、`modelscope` | 远端模型配置来源。 |
| `--performance-model` | `analytic`、`profiling` | 性能模型来源。 |
| `--quantize-linear-action` | `DISABLED`、`W8A8_STATIC`、`W8A8_DYNAMIC`、`FP8`、`MXFP4` | 线性层量化策略。 |
| `--quantize-non-expert-linear-action` | `DISABLED`、`W8A8_STATIC`、`W8A8_DYNAMIC`、`FP8`、`MXFP4` | 非 expert 线性层量化策略。 |
| `--quantize-attention-action` | `DISABLED`、`INT8`、`FP8` | Attention 量化策略。 |

如需取消 Tab 补全，可执行：

```bash
msmodeling --disable-tab-completion
```

### 2.3 模型推理性能仿真

TensorCast（模型推理性能仿真）面向 PyTorch 程序进行性能建模。它不会在真实加速器上执行模型，而是拦截计算图，并基于目标设备画像估算算子耗时、显存占用和整体推理性能。

> [!NOTE]知识点：模型推理性能仿真输出
> TensorCast 默认输出算子级性能汇总、总执行时间、TPS/Device 与显存占用。若指定 `--chrome-trace`，还可以生成 Chrome Trace 文件用于可视化分析。

#### 2.3.1 执行 LLM 文本生成仿真

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
    --num-queries 2 \
    --query-length 3500 \
    --device TEST_DEVICE
```

#### 2.3.2 查看仿真结果

命令执行成功后，终端会输出类似以下内容：

```text
Model compilation and execution time: 0.192 s
----------------------------------------------  --------------  ------------  ----------
                     Name                       analytic total  analytic avg  # of Calls
----------------------------------------------  --------------  ------------  ----------
tensor_cast.static_quant_linear.default              884.004ms       1.973ms         448
tensor_cast.attention.default                        259.855ms       4.060ms          64
...
Total time for analytic: 1.744s
[analytic] TPS/Device: 4013 token/s
Total device memory: 64.000 GB
```

关键指标说明：

- `analytic total`：算子估算总耗时。
- `analytic avg`：算子每次调用的平均耗时。
- `# of Calls`：算子被调用的次数。
- `TPS/Device`：每设备每秒 token 数。
- `Total device memory`：权重、KV cache、activation 等显存估算结果。

成功标准：

- 终端输出算子级性能表。
- 输出 `Total time for analytic` 或 `[analytic] TPS/Device`。
- 输出显存估算结果，例如 `Total device memory`。

#### 2.3.3 生成 Chrome Trace（可选）

如需查看更细粒度的时间线，可增加 `--chrome-trace` 参数：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
    --num-queries 2 \
    --query-length 3500 \
    --device TEST_DEVICE \
    --chrome-trace ./tensorcast_trace.json
```

生成后，可通过 `chrome://tracing` 或 MindStudio Insight 打开 trace 文件。

### 2.4 服务化性能仿真

Throughput Optimizer（服务化性能仿真）可在 TTFT、TPOT 等 SLO 约束下，自动搜索最优并行策略和 batch 配置，帮助评估给定模型在目标硬件上的最大服务吞吐。

> [!NOTE]知识点：PD 混部
> PD 混部表示 Prefill 与 Decode 运行在同一实例中，适合快速评估整体服务吞吐。若需要分别评估 Prefill 与 Decode，可继续阅读《[服务化性能仿真使用指南](../user_guide/msmodeling_throughput_optimizer_user_guide.md)》。

#### 2.4.1 执行服务化性能仿真

以下命令用于快速体验 PD 混部场景。首次体验时不额外指定搜索维度，工具会使用默认 TP 搜索范围；如果运行耗时较长，可减少 `--num-devices` 或在进阶使用时显式指定 `--tp-sizes` 缩小搜索范围。

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --tpot-limits 50
```

#### 2.4.2 查看优化结果

命令执行成功后，终端会先打印输入配置和最优配置摘要，随后展示候选并行配置表。例如：

```text
Input Configuration:
  Model: Qwen/Qwen3-32B
  Devices: 8 TEST_DEVICE
  TPOT Limits: 50.0 ms

Overall Best Configuration:
  Best Throughput: 2161.56 tokens/s
  TTFT: 13848.08 ms
  TPOT: 49.98 ms

Top 4 PD Aggregated Configurations:
| Top | Throughput (token/s) | TTFT (ms) | TPOT (ms) | concurrency | num_devices | parallel           | batch_size |
|  1  | 2161.56              | 13848.08  | 49.98     | 128         | 8           | TP=4 | PP=1 | DP=2 | 64         |
```

重点关注以下字段：

- `TP` / `DP`：推荐的并行策略。
- `concurrency`：当前候选配置支持的并发请求数。
- `batch size`：满足 SLO 约束下的批大小。
- `TTFT` / `TPOT`：首 token 时间与每输出 token 时间。
- `Throughput (token/s)`：系统级输出 token 吞吐，数值越大表示吞吐越高。

成功标准：

- 终端输出 `Overall Best Configuration` 或候选配置表。
- 输出 `Throughput`、`TTFT`、`TPOT` 等指标。
- 没有出现模型配置加载失败或参数冲突报错。

## 3. 结果校验与下一步

如果上述命令执行成功，说明已完成模型推理性能仿真与服务化性能仿真的核心流程体验：

- 模型推理性能仿真：已完成 TensorCast 仿真流程，并获得算子级耗时、TPS/Device 与显存估算。
- 服务化性能仿真：已完成 Throughput Optimizer 仿真流程，并获得推荐并行策略与吞吐指标。

常见问题：

- 如果提示无法下载模型配置，请确认网络可访问 Hugging Face，或设置 `HF_ENDPOINT` 镜像。
- 如果提示无法找到 `cli` 或 `tensor_cast` 模块，请确认当前目录为仓库根目录，或已正确设置 `PYTHONPATH`。
- 如果服务化性能仿真运行耗时较长，可先减小搜索范围，例如减少 `--num-devices` 或显式指定 `--tp-sizes`。

更多用法请继续阅读：

- 《[模型推理性能仿真使用指南](../user_guide/msmodeling_tensor_cast_user_guide.md)》
- 《[服务化性能仿真使用指南](../user_guide/msmodeling_throughput_optimizer_user_guide.md)》

## 4. 【可选】Web UI 体验

若更偏好可视化操作，可在完成上述 CLI 流程后，通过 Web UI 在页面完成模型推理性能仿真、服务化性能仿真等配置，并以曲线和表格查看结果。

在仓库根目录启动 Web UI：

```bash
python -m web_ui.web_ui_start --port 2345
```

浏览器访问 `http://127.0.0.1:2345`，即可通过页面配置模型、芯片、并行、量化和 workload 参数，并查看仿真曲线、明细表格与导出结果。

更多页面说明、参数配置与结果解读，请参见《[Web UI 使用指南](../user_guide/msmodeling_web_ui_user_guide.md)》。
