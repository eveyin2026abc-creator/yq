# vLLM 寻优插件

服务化参数寻优工具的 vLLM 插件包，支持 vLLM PD（Prefill-Decode）分离部署场景的参数寻优。

| 场景 | `-e` 策略 | 说明 | 详细文档 |
|------|-----------|------|----------|
| vLLM PD 分离部署 | `vllm_pd` | Prefill-Decode 分离部署集群的参数寻优 | [README_vllm_pd.md](README_vllm_pd.md) |

## 概念说明

- **vLLM PD 分离部署**：将 vLLM 推理的预填充（Prefill）与解码（Decode）两个阶段拆分到不同节点执行，以提升长上下文、高并发场景下的整体吞吐与首 token 时延。本插件将多节点的启停、健康检查封装为黑盒，配合优化器自动寻优服务参数。
- **evalscopeperf**：基于 evalscope 的推理性能评测后端，作为 `-b` 参数指定的 benchmark 后端执行压测。
- **vllm_pd_simulator**：本插件包名（`vllm_pd` + `vllm_cluster` 统一模拟器插件），通过 SSH 远程管理 vLLM PD 分离集群。

## 环境要求

- 操作系统：Linux（如 Ubuntu 20.04+ / openEuler 22.03+），各节点间网络互通
- Python >= 3.10，已创建并激活虚拟环境
- pip 可用，具备 editable 安装权限
- 各节点已安装 NPU 驱动与 CANN 环境，已安装 vLLM 及 vllm-ascend
- 主节点已安装 fabric >= 2.0（插件依赖）

## 硬件要求

- 加速卡：昇腾 NPU（型号与数量以实际部署为准，配置示例按 4 卡/节点）
- CPU / 内存 / 磁盘：满足 vLLM 推理、模型权重与日志/评测结果的存储需求
- 网络：节点间需 RDMA / HCCL 链路用于 KV 传输与集合通信

## 安装

前提：已克隆仓库并位于其上级目录，已创建并激活 Python 虚拟环境。

```bash
# 安装优化器主体
cd MindStudio-Service-Profiler_xql
pip install -e ./ms_serviceparam_optimizer
pip install -e ./evalscopeperf --no-deps 2>/dev/null || true

# 安装 vLLM 插件
cd plugins/optimizer/vllm_pd_simulator
pip install -e .
```

预期回显：各 `pip install` 成功时输出 `Successfully installed ...`（如 `Successfully installed vllm_pd_simulator-0.1.0`）。

### 验证安装

```bash
ms_serviceparam_optimizer --version
```

能正常输出版本号即安装成功。

### 版本配套

| 组件 | 版本要求 |
|------|----------|
| Python | >= 3.10 |
| fabric | >= 2.0 |
| vllm_pd_simulator | 0.1.0（本插件） |
| msmodeling | 0.2.0 |
| vLLM / vllm-ascend | 以实际安装为准 |
| CANN | 以实际安装为准 |

> 功能支持级别：技术预览，不建议直接用于生产环境。

## 使用

```bash
# vLLM PD 分离部署寻优
ms_serviceparam_optimizer optimizer -e vllm_pd -b evalscopeperf
```

### 参数说明

| 参数 | 含义 | 类型 | 必填 | 默认值 | 取值 |
|------|------|------|------|--------|------|
| `-e` | 寻优策略（模拟器/场景插件） | string | 是 | - | `vllm_pd`（vLLM PD 分离部署） |
| `-b` | benchmark 后端 | string | 是 | - | `evalscopeperf`（evalscope 性能评测） |

具体配置和部署场景请参考 [README_vllm_pd.md](README_vllm_pd.md)。

## 相关资源

- 完整文档入口：[README_vllm_pd.md](README_vllm_pd.md)
- vLLM 上游：[vllm-project/vllm](https://github.com/vllm-project/vllm)
- vLLM Ascend 上游：[vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)
- 问题反馈：通过 GitCode / 对应仓库 Issue 提交
