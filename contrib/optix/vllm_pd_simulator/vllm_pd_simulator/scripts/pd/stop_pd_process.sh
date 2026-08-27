#!/bin/bash
# Stop vLLM PD processes: kill GPU/port occupants, or --all pkill vllm+proxy (case-insensitive).
# Usage: bash stop_pd_process.sh <pid>... | --gpus <id,id> | --port <p> | --all | (no-arg = --all)
REMOTE_DIR="${REMOTE_DIR:-/tmp/vllm_pd}"
PROCESS_PATTERN="${PROCESS_PATTERN:-vllm serve|VLLM::|proxy_server|proxy_layerwise_server|${REMOTE_DIR}/}"
SELF_PID=$$

# 属 vllm/proxy 目标？（排除 simulator/optimizer/本脚本）：0=是
_is_target() {
    local cmd; cmd=$(ps -p "$1" -o args= 2>/dev/null) || return 1
    [ -z "$cmd" ] && return 1
    echo "$cmd" | grep -qiE 'simulator|optimizer|stop_pd_process' && return 1
    echo "$cmd" | grep -qiE "$PROCESS_PATTERN"
}
# 沿 ppid 有界回溯，输出 vllm/proxy 祖先 PID；遇首个非目标祖先即停（不触及 init/sshd/框架）。
# 起点持卡 PID 由调用方无条件收录（追加②）；此处只取祖先。防环：seen；命中 1/0 止。
_ancestor_pids() {
    local pid="$1" seen=""
    pid="$(awk '/^PPid:/{print $2}' /proc/"$1"/status 2>/dev/null)"
    while [ -n "$pid" ] && [ "$pid" != 1 ] && [ "$pid" != 0 ]; do
        case " $seen " in *" $pid "*) break ;; esac
        seen="$seen $pid"
        _is_target "$pid" || break
        echo "$pid"
        pid="$(awk '/^PPid:/{print $2}' /proc/"$pid"/status 2>/dev/null)"
    done
}

# <pid> 路径：直接 SIGTERM + SIGKILL
if [ $# -ge 1 ] && [ "$1" != "--gpus" ] && [ "$1" != "--port" ] && [ "$1" != "--all" ]; then
    for pid in "$@"; do
        kill "$pid" 2>/dev/null && echo "[STOPPED] pid=$pid (SIGTERM)"
        kill -9 "$pid" 2>/dev/null && echo "[KILLED] pid=$pid (SIGKILL)"
    done
    exit 0
fi

GPUS="" PORT=""
while [ $# -gt 0 ]; do
    case "$1" in --gpus) GPUS="$2"; shift 2 ;; --port) PORT="$2"; shift 2 ;; --all) shift ;; *) shift ;; esac
done

# 收集占用 GPU / 端口的 PID（--gpus: npu-smi 持卡 PID 无条件 + 有界 ppid 回溯 vllm 祖先，命中 API Server）
TARGET=""
if [ -n "$GPUS" ]; then
    if command -v npu-smi >/dev/null 2>&1; then
        # 优化：npu-smi info 一次输出所有卡信息，只调一次存变量，for 循环内复用（避免 8/16 张卡重复调用）
        NPU_SMI_OUTPUT="$(npu-smi info 2>/dev/null)"
        # npu-smi info 一次输出卡信息表+进程表：卡信息表每卡两行，NPU 级行（$2=NPU Name）
        # 记录 NPU_ID，Chip 级行（$2=Chip_within 全局ID）以 (NPU,Chip_within) 为 key 与进程表
        # 对齐；排序后按索引映射 --gpus 第 N 个 gid，再按 NPU+Chip 精确匹配进程表 PID（修复
        # 旧逻辑将 Chip 级行 $2 误作 (NPU,Chip)，与进程表 key 永不匹配，导致 --gpus 全部漏杀）。
        for wpid in $(echo "$NPU_SMI_OUTPUT" | awk -F'|' -v gpus="$GPUS" '
            BEGIN {
                split(gpus, gid_arr, ",")
                for (i in gid_arr) gid_target[gid_arr[i]] = 1
            }
            # 卡信息表+进程表：$2 首 token 必须为数字，否则跳过表头/边界行
            /^\| [0-9]/ && NF >= 2 {
                col2 = $2; sub(/^[ \t]+/, "", col2); split(col2, a, " ")
                if (!(a[1] ~ /^[0-9]+$/)) next
                # 卡信息表 NPU 级行：$2 含非数字 token（Name）→ 记录当前 NPU_ID
                if (a[2] !~ /^[0-9]+$/) {
                    current_npu = a[1]
                    next
                }
                # flat 单行格式兼容：$2 含 3+ token 且第 3 个非数字（Name），a[1]=NPU,a[2]=Chip
                if (a[3] != "" && a[3] !~ /^[0-9]+$/) {
                    current_npu = a[1]
                    key = a[1] "," a[2]
                    if (!(key in card_seen)) {
                        card_seen[key] = 1
                        card_npus[++card_cnt] = a[1]
                        card_chips[card_cnt] = a[2]
                    }
                    next
                }
                # a[1]/a[2] 均数字，用 $3 区分卡信息表 Chip 级行 vs 进程行
                col3 = $3; sub(/^[ \t]+/, "", col3); sub(/[ \t]+$/, "", col3)
                if (col3 !~ /^[0-9]+$/) {
                    # 卡信息表 Chip 级行：a[1]=Chip_within,a[2]=Global_ID,col3=Bus-Id
                    # key = (NPU, Chip_within)，与进程表 key 对齐（修复卡键映射）
                    if (current_npu != "") {
                        key = current_npu "," a[1]
                        if (!(key in card_seen)) {
                            card_seen[key] = 1
                            card_npus[++card_cnt] = current_npu
                            card_chips[card_cnt] = a[1]
                        }
                    }
                } else if (NF >= 4) {
                    # 进程行：a[1]=NPU,a[2]=Chip_within,col3=PID，按 (NPU,Chip) 分组
                    key = a[1] "," a[2]
                    pid_col = $3; sub(/^[ \t]+/, "", pid_col); split(pid_col, b, " ")
                    if (b[1] ~ /^[0-9]+$/) {
                        proc_pids[key] = (key in proc_pids ? proc_pids[key] " " : "") b[1]
                    }
                }
            }
            END {
                # 卡列表按 (NPU,Chip) 排序（冒泡，卡数 ≤ 64 足够）
                for (i = 1; i <= card_cnt; i++) {
                    for (j = i + 1; j <= card_cnt; j++) {
                        if (card_npus[i] > card_npus[j] ||
                            (card_npus[i] == card_npus[j] && card_chips[i] > card_chips[j])) {
                            tmp = card_npus[i]; card_npus[i] = card_npus[j]; card_npus[j] = tmp
                            tmp = card_chips[i]; card_chips[i] = card_chips[j]; card_chips[j] = tmp
                        }
                    }
                }
                # 对每个 gid_target，输出对应 (NPU,Chip) 上的所有 PID
                for (i = 1; i <= card_cnt; i++) {
                    idx = i - 1  # 0-based 索引
                    if (idx in gid_target) {
                        key = card_npus[i] "," card_chips[i]
                        if (key in proc_pids) print proc_pids[key]
                    }
                }
            }
        '); do
            TARGET="$TARGET $wpid $(_ancestor_pids "$wpid")"
        done
    else
        echo "[WARN] npu-smi not found; --gpus needs npu-smi, skip" >&2
    fi
fi
if [ -n "$PORT" ]; then
    if command -v ss >/dev/null 2>&1; then TARGET="$TARGET $(ss -tlnp 2>/dev/null | awk -v p="$PORT" '$4 ~ ":"p"$" {print $NF}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u | tr '\n' ' ')"
    else TARGET="$TARGET $(netstat -tlnp 2>/dev/null | awk -v p="$PORT" '$4 ~ ":"p"$" {print $NF}' | cut -d/ -f1 | sort -u | tr '\n' ' ')"; fi
fi
# 补捞孤儿 VLLM:: 进程（ppid=1，父进程已死被 init 收养）：npu-smi/ss 与 ppid 回溯均捞不到
if [ -n "$GPUS" ] || [ -n "$PORT" ]; then
    for opid in $(pgrep -i -f "VLLM::" 2>/dev/null); do
        oppid=$(awk '/^PPid:/{print $2}' /proc/$opid/status 2>/dev/null)
        [ "$oppid" = "1" ] && TARGET="$TARGET $opid"
    done
fi
TARGET=$(echo "$TARGET" | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ')

# 定向路径（--gpus/--port）
if [ -n "$GPUS" ] || [ -n "$PORT" ]; then
    if [ -z "$TARGET" ]; then
        echo "[OK] no process on specified GPUs/ports"
    else
        echo "[STOP] Killing: $TARGET"
        for pid in $TARGET; do
            kill "$pid" 2>/dev/null && echo "[STOPPED] pid=$pid (SIGTERM)"
            kill -9 "$pid" 2>/dev/null && echo "[KILLED] pid=$pid (SIGKILL)"
        done
        echo "[DONE] targeted cleanup completed"
    fi
    exit 0
fi

# --all / 无参：pgrep -i -f 忽略大小写匹配 vllm + worker + proxy，排除 simulator/optimizer/自身
echo "[STOP] pkill all matching ($PROCESS_PATTERN)..."
for pid in $(pgrep -i -f "$PROCESS_PATTERN" 2>/dev/null); do
    [ "$pid" = "$SELF_PID" ] && continue
    cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
    echo "$cmdline" | grep -qiE 'simulator|optimizer|stop_pd_process|check_pd_process' && { echo "[SKIP] pid=$pid (framework/self)"; continue; }
    kill "$pid" 2>/dev/null && echo "[STOPPED] pid=$pid (SIGTERM)"
    kill -9 "$pid" 2>/dev/null && echo "[KILLED] pid=$pid (SIGKILL)"
done
echo "[DONE] all cleanup completed"
exit 0
