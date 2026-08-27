# 最优配置复现指南

寻优完成后，最优轮次对应的启动脚本已保存在 `result/scripts/<时间戳>/round_XXXX/` 目录下。

> 最后更新：2026-08-13（vllm_pd_simulator 0.1.0）

以下步骤说明如何使用这些脚本复现最优配置环境。

## 脚本目录说明

最优轮次目录结构及各文件功能：

```text
round_XXXX/
├── P-I0-R0-T1-D1_<IP>_<端口>_none.sh            ← Prefill 节点启动脚本
├── D-I0-R0-T1-D1_<IP>_<端口>_none.sh            ← Decode 节点启动脚本
├── proxy_<IP>_<端口>_none.sh                    ← Proxy 启动脚本
├── load_balance_proxy_server_example.py            ← Proxy 负载均衡服务（由 proxy 脚本调用）
├── check_pd_process.sh                            ← 检查残留进程
├── stop_pd_process.sh                             ← 停止所有 PD 相关进程
└── log/                                           ← 运行日志（寻优时从远端收集）
```

脚本命名规则：`<角色>-I<实例号>-R<DP排名>-T<TP大小>-D<DP大小>_<IP>_<SSH端口>_<容器ID>.sh`

占位符取值含义：

| 占位符 | 含义 |
|--------|------|
| `<角色>` | `P`（Prefill）、`D`（Decode）、`proxy` |
| `<实例号>` | 同一角色内的实例编号（`I0` 起） |
| `<DP排名>` | DP 全局排名（`R0` 起，由数据并行拓扑决定） |
| `<TP大小>` | 张量并行（Tensor Parallel）大小 |
| `<DP大小>` | 数据并行（Data Parallel）大小 |
| `<IP>` | 节点 SSH IP |
| `<SSH端口>` | 节点 SSH 端口 |
| `<容器ID>` | Docker 容器 ID；非 Docker 环境显示为 `none` |

## 术语速查

- **PD 分离**：Prefill（预填充）/ Decode（解码）分离部署，两类节点可独立扩缩与寻优
- **TP**：Tensor Parallel，张量并行，将模型层切分到多卡
- **DP**：Data Parallel，数据并行，将请求/批次切分到多个实例
- **`R{DP排名}`**：脚本名中的 DP 全局排名，标识该实例在数据并行拓扑中的位置
- **Proxy**：负载均衡代理，转发 P/D 节点间的 KV Cache

启动脚本已包含完整运行环境（CANN、HCCL、寻优参数等），可直接执行。

## 1. 上传脚本到远端节点

将最优轮次目录下的所有文件上传到各节点：

```bash
ROUND_DIR="result/scripts/20260611-143000/round_0003"
REMOTE_DIR="/tmp/vllm_pd_0"

# 上传到各节点（按实际 IP 和端口替换）
scp -P <ssh端口> $ROUND_DIR/* root@<节点IP>:$REMOTE_DIR/
```

> Docker 环境：上传后还需将文件复制到容器内
>
> ```bash
> docker cp /tmp/vllm_pd_0 <容器ID>:/tmp/vllm_pd_0
> ```

## 2. 启动服务

SSH 登录到各节点，按顺序执行：

```bash
# 1. 启动 Prefill 节点
bash -l /tmp/vllm_pd_0/P-I0-R0-T1-D1_<IP>_<端口>_<容器ID>.sh

# 2. 启动 Decode 节点
bash -l /tmp/vllm_pd_0/D-I0-R0-T1-D1_<IP>_<端口>_<容器ID>.sh

# 3. 等待 P/D 就绪后，启动 Proxy
bash -l /tmp/vllm_pd_0/proxy_<IP>_<端口>_<容器ID>.sh
```

> Docker 环境：在容器内执行
>
> ```bash
> docker exec -it <容器ID> bash -l /tmp/vllm_pd_0/<脚本名>.sh
> ```

## 3. 检查服务状态

```bash
curl http://<bind_ip>:<port>/health        # P/D 节点
curl http://<bind_ip>:<port>/healthcheck    # Proxy
```

返回 `200` 表示就绪。

## 4. 运行性能测试

```bash
evalscope perf \
    --url http://<proxy_bind_ip>:<proxy_port>/v1/chat/completions \
    --model <模型名> \
    --tokenizer-path <tokenizer路径> \
    --dataset <数据集> \
    --outputs-dir ./evalscope_result
```

参数说明：

| 参数 | 取值要求 | 示例 |
|------|----------|------|
| `--url` | Proxy 的 Chat 接口地址，格式 `http://<proxy_bind_ip>:<proxy_port>/v1/chat/completions` | `http://192.0.2.10:8000/v1/chat/completions` |
| `--model` | 与 vLLM 服务端 `--served-model-name` 一致的模型名 | `Qwen2.5-72B` |
| `--tokenizer-path` | 与模型匹配的 tokenizer 目录或路径 | `/data/models/Qwen2.5-72B` |
| `--dataset` | 评测数据集（evalscope 内置或本地数据集） | `ceval` / `/data/datasets/...` |
| `--outputs-dir` | 评测结果输出目录 | `./evalscope_result` |

## 验证

- **启动到就绪**：P/D 节点与 Proxy 全部返回 `200` 后进入下一步；等待时间取决于模型加载与节点规模
- **非 `200` / 连接拒绝**：查看 `log/` 目录对应节点日志（或远端 `/tmp/optix_*.log`），确认进程存活与端口监听后重试
- **完整验证清单**：
  1. P/D 节点 `/health` 与 Proxy `/healthcheck` 均返回 `200`
  2. 各节点日志无 fatal / 端口冲突报错
  3. `evalscope perf` 能正常输出评测结果

## 5. 停止服务

```bash
bash -l /tmp/vllm_pd_0/stop_pd_process.sh
# 注意：共享物理节点场景禁用 --all，避免跨集群误杀（改用 --gpus/--port）
```

## 常见问题 / 排错

- **脚本启动失败**：确认脚本可执行、路径正确，检查日志中是否有 CANN / 模型加载错误
- **残留进程**：先运行 `check_pd_process.sh` 检查；存在残留时用 `stop_pd_process.sh` 清理
- **健康检查不通过**：按「检查服务状态」确认 `200`；失败时查看 `log/` 目录日志定位原因
- **端口被占用**：用 `ss -tlnp` 确认占用方，调整脚本中的 `service_port` / `kv_port`

## 说明

- 脚本已包含完整环境（CANN、HCCL、寻优参数等），无需额外 source 或 export
- 寻优参数（`PD_` 开头的环境变量）已在脚本生成时硬编码为具体值
- 脚本硬编码使用 `/opt/mamba` 路径和 `ascend-infer` 环境名，不支持通过环境变量自定义
- 多节点在同一台机器时，只需上传一次，按顺序执行即可

## 版本配套

- 脚本依赖 CANN、HCCL、vLLM 与 evalscope；具体版本以实际安装为准
- 插件：vllm_pd_simulator 0.1.0（Python >= 3.10，fabric >= 2.0）

## 相关资源

- vLLM PD 架构：见插件文档 [README_vllm_pd.md](../../../README_vllm_pd.md) 的「术语表」与「架构设计」
- evalscope：[github.com/modelscope/evalscope](https://github.com/modelscope/evalscope)
- CANN / HCCL：以华为官方渠道发布为准
