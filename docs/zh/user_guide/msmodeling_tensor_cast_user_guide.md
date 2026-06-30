# TensorCast 使用指南

完整模型列表与特性明细请参见《[模型支持与特性支持矩阵](./support_matrix/support_matrix_user_guide.md)》。

## 阅读路径

| 目标 | 推荐章节 |
| --- | --- |
| 1. 快速跑通 LLM 文本生成仿真 | [2.1 快速入门：文本生成](#21-快速入门文本生成) |
| 2. 理解输出中的耗时、调用次数和显存指标 | [2.2 结果（文本生成）](#22-结果文本生成) |
| 3. 运行视频生成模型仿真 | [2.3 快速入门：视频生成](#23-快速入门视频生成) |
| 4. 查看或自定义硬件设备画像 | [3 支持的加速器与自定义设备](#3-支持的加速器与自定义设备) |
| 5. 配置 prefill / decode 场景 | [5.2 运行 Prefill](#52-运行-prefill) 与 [5.3 运行 Decode](#53-运行-decode) |

## 1 简介

TensorCast 是一个面向 PyTorch 程序的性能仿真与分析框架。它使开发者和研究人员能够在无需访问物理机器的情况下，预测其神经网络模型在特定硬件配置上的性能表现。

从本质上讲，TensorCast 充当"虚拟机"或运行时仿真器。它不会在实际加速器上执行计算，而是拦截 PyTorch 程序的计算图，并在用户定义的 MachineConfig 上仿真其执行过程。该配置指定了目标硬件的特性，例如理论算力（TFLOPS）、内存带宽、缓存层次结构以及互连速度。为了准确估算模型在给定硬件上的最优性能，TensorCast 提供了模型优化流水线，包括自动模型分片、量化以及 FX 图优化，在开展分析前将源程序转换为最优形式。

通过在"虚拟"硬件上运行模型，TensorCast 提供详细的性能洞察，包括：

- 开箱即用的 Huggingface transformer 模型支持。
- 通过简单配置支持多种硬件加速器设备。
- 算子级执行时间：使用可扩展模型进行估算，例如解析 roofline 模型、经验数据或基于机器学习的预测器。
- 内存占用：跟踪总内存分配量和峰值内存分配量。
- 计算特征：分析每个算子的 FLOPs（浮点运算次数）和内存访问量。
- 高级调度仿真：建模复杂执行模式，例如跨多个 stream 的并发计算。

最终输出包括全面的汇总表和详细的 Chrome Trace 文件，便于深入可视化并识别性能瓶颈。

首次使用前，请先参阅《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》完成环境搭建与第一次仿真。

## 2 概览

### 2.1 快速入门：文本生成

**功能说明：** 仿真一批 query 的 LLM 推理性能。

**命令：**

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
```

**关键参数：** `--context-length`、`--decode`、`--quantize-linear-action`、`--chrome-trace`、`--device`

**输出：** 性能汇总表；若设置了 `--chrome-trace`，还可选输出 Chrome trace 文件。

### 2.2 结果（文本生成）

示例输出（已截断）：

```text
Model compilation and execution time: 0.192 s
----------------------------------------------  --------------  ------------  ----------
                     Name                       analytic total  analytic avg  # of Calls
----------------------------------------------  --------------  ------------  ----------
tensor_cast.static_quant_linear.default              884.004ms       1.973ms         448
tensor_cast.attention.default                        259.855ms       4.060ms          64
aten.mul.Tensor                                      198.215ms     237.668us         834
aten._to_copy.default                                100.528ms     195.580us         514
tensor_cast.dynamic_quantize_symmetric.default        76.519ms     170.802us         448
...
Total time for analytic: 1.744s
[analytic] Execution time: 1.744174 s
[analytic] TPS/Device: 4013 token/s
Total device memory: 64.000 GB
  Model weight size: 31.981 GB
  KV cache: 1.719 GB
  Model activation size: 0.601 GB
  Reserved memory: 0.000 GB
  Memory available: 29.699 GB
```

说明：`Model compilation and execution time` 是仿真器在宿主机上的运行时间，而非模型在硬件上的真实编译或执行时间。

指标说明：

- `analytic total`：算子估算总耗时。
- `analytic avg`：算子每次调用的平均耗时。
- `# of Calls`：算子被调用的次数。
- `Total time for analytic`：解析算子耗时的总和。
- `TPS/Device`：每设备每秒 token 数。
- `Total device memory` 及其细分项：权重、KV cache 和 activation 的估算内存占用。

### 2.3 快速入门：视频生成

**功能说明：** 仿真视频生成模型的 diffusion transformer 前向传播。

**命令：**

```bash
python -m cli.inference.video_generate docs/fixtures/hunyuanvideo_mock_model --batch-size 1 --seq-len 16 --height 576 --width 1024 --frame-num 14 --sample-step 25 --device TEST_DEVICE
```

**关键参数：** `--height`、`--width`、`--frame-num`、`--sample-step`、`--chrome-trace`、`--device`

**输出：** 性能汇总表；若设置了 `--chrome-trace`，还可选输出 Chrome trace 文件。

### 2.4 结果（视频生成）

示例输出（已截断）：

```text
Model compilation and execution time: 25.44349410000723s
----------------------------------------------  --------------  ------------  ----------
                     Name                       analytic total  analytic avg  # of Calls
----------------------------------------------  --------------  ------------  ----------
aten.addmm.default                                      8.546s       1.280ms        6675
tensor_cast.attention.default                           7.943s       5.125ms        1550
aten.mul.Tensor                                         2.597s     126.510us       20525
aten._to_copy.default                                   2.450s     142.242us       17225
tensor_cast.static_quant_linear.default                 2.266s     323.720us        7000
...
Total time for analytic: 29.350s
```

说明：`Model compilation and execution time` 是仿真器在宿主机上的运行时间，而非模型在硬件上的真实编译或执行时间。

指标说明：

- `analytic total`：算子估算总耗时。
- `analytic avg`：算子每次调用的平均耗时。
- `# of Calls`：算子被调用的次数。
- `Total time for analytic`：解析算子耗时的总和。

## 3 支持的设备与自定义设备

我们为以下设备配置提供内置支持（定义于 `tensor_cast/device.py`）：

- `TEST_DEVICE`
- `ATLAS_800_A2_376T_64G`
- `ATLAS_800_A2_313T_64G`
- `ATLAS_800_A2_280T_64G`
- `ATLAS_800_A2_280T_64G_PCIE`
- `ATLAS_800_A2_280T_32G_PCIE`
- `ATLAS_800_A3_752T_128G_DIE`
- `ATLAS_800_A3_560T_128G_DIE`
- `ATLAS_800_A3_560T_128G_DIE_ROCE`
- `ATLAS_350_425T_112G`
- `ATLAS_350_425T_84G`

### 3.1 自定义设备类型

对于其他硬件，可在 `tensor_cast/device_profiles` 下以 Python 文件形式定义自定义设备配置。TensorCast 会自动加载，之后即可在 CLI 中引用该配置名称。自定义设备指南：[device_profiles/README.md](../../../tensor_cast/device_profiles/README.md)

## 4 详细用法


推荐使用本地安全模式：先下载并审核模型仓库，然后将 `model_id` 填为完整本地绝对路径，例如
`/data/models/Qwen3-32B`。本地路径加载会在运行前校验路径属主、软链接和权限；不建议使用软链接目录、
共享可写目录或未审核来源中的模型文件。

工具仍支持直接传入 Hugging Face 或 ModelScope 的 model id，例如 `Qwen/Qwen3-32B`，并可通过
`--remote-source` 选择来源。该 model id 模式可能在 `trust_remote_code=True` fallback 时执行远端
Python 代码，msmodeling 不对远端代码安全性做保证；运行时会打印 `trust_remote_code` 风险提示。

### 4.1 文本生成仿真详细用法

我们提供 `text_generate.py` 命令行接口来仿真文本生成。该脚本支持对一批具有相同输入长度、可选相同 context 长度的 query 进行文本生成仿真。默认提供算子性能分解的表格汇总。也可选择导出 Chrome trace。

其一般用法如下：

```text
usage: text_generate.py [-h]
                        [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,ATLAS_800_A3_560T_128G_DIE_ROCE,ATLAS_350_425T_112G,ATLAS_350_425T_84G}]
                        [--num-devices NUM_DEVICES] [--reserved-memory-gb RESERVED_MEMORY_GB] [--log-level {debug,info,warning,error,critical}] --num-queries NUM_QUERIES --query-length QUERY_LENGTH
                        [--context-length CONTEXT_LENGTH] [--decode] [--num-mtp-tokens NUM_MTP_TOKENS] [--disable-repetition] [--compile] [--compile-allow-graph-break] [--enable-multistream]
                        [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}]
                        [--quantize-non-expert-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}]
                        [--quantize-lmhead] [--mxfp4-group-size MXFP4_GROUP_SIZE]
                        [--quantize-attention-action {DISABLED,INT8,FP8}] [--graph-log-url GRAPH_LOG_URL] [--dump-input-shapes] [--chrome-trace CHROME_TRACE]
                        [--num-hidden-layers-override NUM_HIDDEN_LAYERS_OVERRIDE] [--tp-size TP_SIZE] [--dp-size DP_SIZE] [--ep-size EP_SIZE] [--o-proj-tp-size O_PROJ_TP_SIZE]
                        [--o-proj-dp-size O_PROJ_DP_SIZE] [--mlp-tp-size MLP_TP_SIZE] [--mlp-dp-size MLP_DP_SIZE] [--lmhead-tp-size LMHEAD_TP_SIZE] [--lmhead-dp-size LMHEAD_DP_SIZE]
                        [--moe-tp-size MOE_TP_SIZE] [--moe-dp-size MOE_DP_SIZE] [--word-embedding-tp {col,row}] [--enable-redundant-experts] [--enable-external-shared-experts] [--host-external-shared-experts]
                        [--image-batch-size IMAGE_BATCH_SIZE] [--image-height IMAGE_HEIGHT] [--image-width IMAGE_WIDTH] [--remote-source {huggingface,modelscope}]
                        [--performance-model {analytic,profiling}] [--profiling-database PROFILING_DATABASE]
                        model_id

Run a simulated LLM inference pass and dump the perf result.
```

主要参数说明如下：

| 参数 | 是否必选 | 说明 |
| --- | --- | --- |
| `model_id` | 是 | 模型 ID 或本地模型路径，例如 `Qwen/Qwen3-32B` 或 `/data/models/Qwen3-32B`。 |
| `--num-queries` | 是 | 本次仿真的 query 数量。 |
| `--query-length` | 是 | 每个 query 的新输入 token 长度。 |
| `--context-length` | 否 | 每个 query 的上下文 token 长度；prefill 场景通常与 `--query-length` 一起设置，decode 场景通常大于 `--query-length`。 |
| `--decode` | 否 | 启用 decode 模式；不设置时按 prefill 模式运行。 |
| `--device` | 否 | 指定用于仿真的设备配置，默认为内置设备配置之一。 |
| `--num-devices` | 否 | 指定参与仿真的设备数量。 |
| `--reserved-memory-gb` | 否 | 指定每张设备预留的显存大小，单位为 GB。 |
| `--quantize-linear-action` | 否 | 指定线性层量化方式，例如 `W8A8_DYNAMIC`、`W8A8_STATIC`、`FP8` 或 `MXFP4`。 |
| `--quantize-non-expert-linear-action` | 否 | 指定非 expert 线性层的量化方式。 |
| `--quantize-lmhead` | 否 | 对 lm head 启用量化。 |
| `--quantize-attention-action` | 否 | 指定 attention 的量化方式，例如 `INT8` 或 `FP8`。 |
| `--tp-size`、`--dp-size`、`--ep-size` | 否 | 分别指定 tensor parallel、data parallel 和 expert parallel 的并行规模。 |
| `--compile` | 否 | 启用编译路径执行仿真。 |
| `--compile-allow-graph-break` | 否 | 在编译路径中允许 graph break。 |
| `--enable-multistream` | 否 | 在 `--compile` 路径启用编译期多流仿真。 |
| `--chrome-trace` | 否 | 指定 Chrome trace 输出路径，用于导出性能时间线。 |
| `--dump-input-shapes` | 否 | 输出输入 shape 信息，便于排查模型输入配置。 |
| `--image-batch-size`、`--image-height`、`--image-width` | 否 | 用于 VL 模型，分别描述输入图像数量、高度和宽度。 |
| `--remote-source` | 否 | 指定远端模型来源，可选 `huggingface` 或 `modelscope`。 |
| `--performance-model` | 否 | 指定性能模型，可选 `analytic` 或 `profiling`。 |
| `--profiling-database` | 否 | 使用 `profiling` 性能模型时指定 profiling 数据库路径。 |

`--enable-multistream` 用于在 `--compile` 路径启用编译期多流仿真。该能力默认开启，因此已有 compile 命令会保持当前行为。

对于 VL 模型，可同时设置 `--image-batch-size`、`--image-height` 和 `--image-width` 来描述输入图像数量与分辨率；纯文本模型可省略这些参数。

运行 `python -m cli.inference.text_generate --help` 查看详情。

### 4.2 视频生成仿真详细用法

我们提供 `video_generate.py` 命令行接口来仿真 diffusion transformer 模型的前向传播与性能。该脚本支持仿真视频生成模型（例如类 Stable Video Diffusion 架构）的推理过程，可配置输入维度、采样步数及并行设置。默认提供详细的算子性能分解表格汇总。也可选择将性能时间线导出为 Chrome Trace 文件。

其一般用法如下：

```text
usage: video_generate.py [-h]
                         [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,ATLAS_800_A3_560T_128G_DIE_ROCE,ATLAS_350_425T_112G,ATLAS_350_425T_84G}]
                         --batch-size BATCH_SIZE --seq-len SEQ_LEN [--chrome-trace CHROME_TRACE] [--height HEIGHT] [--width WIDTH] [--frame-num FRAME_NUM] [--sample-step SAMPLE_STEP]
                         [--log-level {debug,info,warning,error,critical}] [--dtype {float16,float32,bfloat16}]
                         [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}] [--use-cfg] [--world-size WORLD_SIZE]
                         [--ulysses-size ULYSSES_SIZE] [--cfg-parallel] [--dit-cache] [--cache-step-range CACHE_STEP_RANGE] [--cache-step-interval CACHE_STEP_INTERVAL]
                         [--cache-block-range CACHE_BLOCK_RANGE]
                         model_id

Run a simulated diffusion transformer forward and dump perf stats.
```

主要参数说明如下：

| 参数 | 是否必选 | 说明 |
| --- | --- | --- |
| `model_id` | 是 | 视频生成模型 ID 或本地模型路径。 |
| `--batch-size` | 是 | 输入 batch 大小。 |
| `--seq-len` | 是 | 输入序列长度。 |
| `--device` | 否 | 指定用于仿真的设备配置，默认为内置设备配置之一。 |
| `--height`、`--width` | 否 | 指定输入视频或图像帧的高度和宽度。 |
| `--frame-num` | 否 | 指定视频帧数。 |
| `--sample-step` | 否 | 指定 diffusion 采样步数。 |
| `--dtype` | 否 | 指定模型计算数据类型，可选 `float16`、`float32` 或 `bfloat16`。 |
| `--quantize-linear-action` | 否 | 指定线性层量化方式，例如 `W8A8_DYNAMIC`、`W8A8_STATIC`、`FP8` 或 `MXFP4`。 |
| `--use-cfg` | 否 | 启用 classifier-free guidance 相关仿真路径。 |
| `--world-size` | 否 | 指定参与分布式仿真的总设备数。 |
| `--ulysses-size` | 否 | 指定 Ulysses 并行规模。 |
| `--cfg-parallel` | 否 | 启用 CFG 并行策略。 |
| `--dit-cache` | 否 | 启用 DiT cache 仿真。 |
| `--cache-step-range` | 否 | 指定启用 cache 的采样步范围。 |
| `--cache-step-interval` | 否 | 指定 cache 的采样步间隔。 |
| `--cache-block-range` | 否 | 指定启用 cache 的 block 范围。 |
| `--chrome-trace` | 否 | 指定 Chrome trace 输出路径，用于导出性能时间线。 |
| `--log-level` | 否 | 指定日志级别，可选 `debug`、`info`、`warning`、`error` 或 `critical`。 |

运行 `python -m cli.inference.video_generate --help` 查看详情。

## 5 进阶说明

### 5.1 External Shared Experts 与 Redundant Experts 实现

以下概述 External Shared Experts 与 Redundant Experts 的实现逻辑。

1. 仅 Redundant Experts：
每个设备会额外托管一个 redundant expert。

2. 仅 External Shared Experts：
设备按 1:`top_k` 的比例分配给 external shared experts 与 routing experts。如有需要，使用 redundant experts 对 routing experts 进行填充。
例如，若 `world_size` 为 64、`top_k` 为 8、routing experts 数量为 256，则有 8 个设备用于托管 external shared experts。
其余 56 个设备用于分布 256 个 routing experts。32 个设备各托管 5 个 routing experts。24 个设备各托管 4 个 routing experts 和 1 个 redundant expert。

3. 同时启用 External Shared Experts 与 Redundant Experts：
分配逻辑与"仅 External Shared Experts"模式相同，另有一项补充：若无需 redundant experts 来填充 routing experts（即 routing experts 在各设备间均匀分布），则每个托管 routing experts 的设备会额外托管一个 redundant expert。

### 5.2 运行 Prefill

要在 A2 上对 Qwen3-32B 运行 prefill，两个请求各 3500 token 输入长度，可运行以下命令：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 3500 --device TEST_DEVICE
```

Prefill 模式下不添加 `--decode`；`--query-length` 表示新输入长度，`--context-length` 表示每个请求的 context 长度。

也可使用多种量化方案对线性层进行量化，例如 W8A8 动态量化，并以 4500 token 的 context 作为前缀：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC
```

### 5.3 运行 Decode

运行 decode 类似，只需调整输入长度和 context 长度。通常输入长度为 1。

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 10 --query-length 1 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_STATIC
```
