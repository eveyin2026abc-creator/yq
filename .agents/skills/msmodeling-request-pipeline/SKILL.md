---
name: msmodeling-request-pipeline
description: >-
  按当前 diff 跑这次改动会碰到的全部本地流水：CI 门禁波 + nightly/benchmark/network 相关波。
  两波都过，这次改动不应再把 nightly 里对应用例打断。
  在用户说请求流水、跑流水、跑一下流水、本地流水、涉及到修改的所有流水、change pipeline 时使用。
  不是 GitCode compile，也不是整场 nightly。
metadata:
  version: 1.0.0
  source: local
---

# msmodeling 请求流水

用户说「请求流水」或「跑这次修改涉及的所有流水」时：**先同步最新主仓，再选测并跑两波**。

## 先同步主仓

跑任何 pytest 之前必须跟上 canonical `origin/<base>`（默认 `master`，分支名含 `26.2.0` 则用 `26.2.0`），避免在过旧基线上绿了、合入后 nightly 被别人的 master 打断。

脚本 `--sync` 默认开：

1. `git fetch origin <base>`
2. 打印 `ahead` / `behind`
3. `behind=0`：继续
4. 工作区脏且 behind：停，先 commit/stash，不要在落后的树上跑
5. 工作区干净且 behind：`git merge --no-edit origin/<base>`；冲突则 `--abort` 并停下
6. 不要 `reset --hard`，不要 rebase，不要 push

`--no-sync` 仅在用户明确说不要拉主仓时用。

## 保证（以及保证不了什么）

- **能保证**：两波都绿 ⇒ 这次 diff 映射到的 CI 用例，以及同文件/同目录里的 nightly、benchmark、network 用例，全场 nightly 再跑时不应被这批改动打断（失败 → 复跑/归因）。
- **保证不了**：别人合入的代码、flaky、未映射的产品路径、整场 8000+ 里和本次 diff 无关的失败。那不是本 skill，去跑 `scripts/run_nightly.sh`。

## 不要做的事

- 不要评论 GitCode `compile`（没点名就不触发远端）
- 不要跑整仓 `scripts/run_nightly.sh`
- 不要只用 `build.py test --suite ci_gate`（它丢掉 nightly，过了也会在 nightly 被打断）
- 不要用默认 `addopts`（排除 nightly/network）
- 不要杀掉已在跑的全量 nightly

## 立刻执行

```bash
REPO="$(git rev-parse --show-toplevel)"
PY="${REPO}/.venv/bin/python"
if [ ! -x "$PY" ]; then PY="uv run --directory ${REPO} python"; fi
SCRIPT="${REPO}/.agents/skills/msmodeling-request-pipeline/scripts/run_change_pipeline.py"
if [ ! -f "$SCRIPT" ]; then
  SCRIPT="${REPO}/.cursor/skills/msmodeling-request-pipeline/scripts/run_change_pipeline.py"
fi
if [ ! -f "$SCRIPT" ]; then
  SCRIPT="$HOME/.cursor/skills/msmodeling-request-pipeline/scripts/run_change_pipeline.py"
fi
$PY "$SCRIPT" --run --repo "$REPO"
```

已说「请求流水」则直接 `--run`（会先 fetch/merge 主仓）。选出 >200 个 node 时先报数量再跑。

## 两波流水

1. **ci**：选出的 node 上 `-m 'not npu and not nightly and not network'`（和门禁/test_map 同一波）
2. **nightly_related**：同一批 node 上 `-m 'not npu and (nightly or benchmark or network)'`（nightly Wave A 里的 `@nightly` + Wave B）

选测：改过的测试文件全量 collect；产品文件走 `test_map`；映射文件再 collect 以补 `@nightly` 兄弟；这些文件所在目录再 collect `nightly or benchmark or network`。

`test_map`：`MSMODELING_TEST_MAP_PATH` → `.msmodeling_cache/test_map/<base>/test_map.json`。产品改动没有 map 必须写明，不能声称 nightly 不会被打断。

## GitCode 留言

两波都绿后，脚本默认在对应 PR 上发：

`本地流水已经通过。` 外加 HEAD、两波均绿、选测条数。

- `gitcode pr comment <PR> -R Ascend/msmodeling --body ...`
- PR 号来自 `--pr`、`GITCODE_PR_NUMBER`，或当前分支 `gitcode pr list --head`
- 当前不是 PR 分支（例如 nightly 工作树）时必须带 `--pr`
- 没过 / 找不到 PR / 没有 gitcode：**不发**
- `--no-comment-on-pass` 可关
- 没跑绿时不要手写一句假装通过

## 跑完怎么回

- 主仓 sync：fetch 的 base、ahead/behind、是否 merge
- base、两波 node 数、各自 exit
- 都绿：写明不应打断 nightly 对应子集；已留言则给 PR 链接
- 有红：failed node + 异常类型；**不要**留言说通过
- 未要求不 commit / push
