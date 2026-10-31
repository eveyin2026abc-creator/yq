# msmodeling Nightly 流水日历

该工具把 Nightly 机器上的 `/tmp/nightly-*.log` 解析为脱敏静态快照。Git 仓库中只保存页面所需的汇总 JSON，不上传原始日志、token、Webhook 或环境变量。静态快照可在 Windows 或 Linux 上部署。

## 目录

- `server.py`：Nightly 机器实时模式，读取本机日志并查询 GitCode issue。
- `export_static.py`：生成 `public/` 静态快照。
- `publish_snapshot.py`：生成快照，并可选择提交、推送到个人 fork 数据分支。
- `tracking.json`：日志文件与 commit、issue、CI 漏检原因的人工映射。
- `serve_windows.cmd`、`serve_windows.ps1`、`serve_linux.sh`：静态网站启动脚本。

## Nightly 机器生成快照

在仓库根目录执行：

```bash
python3 tools/nightly_board/export_static.py
```

确认 `public/` 内容后发布：

```bash
python3 tools/nightly_board/publish_snapshot.py \
  --publish \
  --remote fork \
  --branch feature/nightly-board-static
```

建议在独立 clone 中执行发布脚本，避免切换 Nightly 工作仓的业务分支。脚本只暂存 `tools/nightly_board/public`。

可通过环境变量覆盖输入位置：

```bash
export NIGHTLY_BOARD_LOG_DIR=/tmp
export NIGHTLY_BOARD_TRACKING=/path/to/tracking.json
```

定时任务示例（北京时间每天 09:00；服务器若为 UTC，应换算为 UTC 01:00）：

```cron
0 1 * * * cd /home/yinqian/nightly-board-publisher && /usr/bin/python3 tools/nightly_board/publish_snapshot.py --publish >> /tmp/nightly-board-publish.log 2>&1
```

## Windows 部署

安装 Python，拉取数据分支后，在 PowerShell 执行：

```powershell
git pull --ff-only
cmd.exe /c ".\tools\nightly_board\serve_windows.cmd 0.0.0.0 8766"
```

第一个参数是监听 IP，第二个参数是端口。默认为 `0.0.0.0 8766`。本机浏览器打开 <http://127.0.0.1:8766/>，同事通过 `http://<你的局域网IP>:8766/` 访问。

如需供同一内网其他机器访问：

```powershell
cmd.exe /c ".\tools\nightly_board\serve_windows.cmd 192.168.1.20 9000"
```

开放 Windows 防火墙端口前，应确认该网络可信。

## Linux 部署

```bash
git pull --ff-only
bash tools/nightly_board/serve_linux.sh
```

浏览器打开 <http://127.0.0.1:8766/>。若需内网访问，设置 `NIGHTLY_BOARD_HOST=0.0.0.0`。

## 数据安全

- 不提交 `/tmp` 原始日志。
- `public/` 只包含失败 node id、错误摘要、commit、覆盖率和 issue 快照。
- `tracking.json` 不得填写 token、密码、Webhook 或私有环境变量。
- 对外开放前仍应增加访问认证；私仓只能保护源码和快照存储，不能替代网站访问控制。
