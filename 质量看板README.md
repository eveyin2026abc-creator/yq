# 质量看板关键代码

代码仓是 [ergon-studio/mindstudio-quality-monitor](https://gitcode.com/ergon-studio/mindstudio-quality-monitor)。下面按使用顺序摘已经在仓里的实现。跟进和认领的原因分析另见 [管理者处理跟进耗时长](./管理者处理跟进耗时长.md)、[管理者分配认领耗时长](./管理者分配认领耗时长.md)。四段链路的更细摘录见 [数据接入到结果确认](./数据接入到结果确认.md)。

## 数据接入

社区 Issue 按已绑定仓库去 GitCode 或 GitHub 拉取。DTS 由浏览器桥抓取，成功后整批替换本地表。

`backend/src/dts.rs`

```rust
async fn fetch_and_store(pool: &sqlx::SqlitePool) -> anyhow::Result<()> {
    let captured = crate::browser_bridge::run("dts.capture", None).await?;
    let raw = serde_json::to_string(&captured.data)?;
    let (source_total, mut rows) = parse_payload(&raw)?;
    enrich_people(pool, &mut rows).await?;
    db::replace_dts_snapshot(pool, source_total, &rows).await
}
```

Issue 落库时，已经划转过的组不会被仓库绑定覆盖。

`backend/src/db/issues.rs`

```rust
group_name=CASE WHEN EXISTS (
  SELECT 1 FROM issue_transfers t
  WHERE t.issue_url = excluded.url AND t.status = 'active'
) THEN issues.group_name ELSE excluded.group_name END
```

## 条件配置

展示口径和入库口径分开。名单或规则命中就不展示，行仍留在库里。

`backend/src/view_filter.rs`

```rust
let out_lists = sets.matched_list_settings(lists, record);
if !out_lists.is_empty() {
    return Verdict { passed: false, out_lists };
}
```

Issue 默认不展示已关闭的单。规则是 `state` 等于 `closed` 或 `close`，再整棵取反。

`backend/src/catalog.rs`

```rust
fn issue_display_scope() -> RuleNode {
    RuleNode {
        kind: "or".into(),
        children: vec![cond("state", "equals", "closed"), cond("state", "equals", "close")],
        negate: true,
    }
}
```

## 进展跟踪

当天进展按单、按日期写入。同一天再改是覆盖，清空则删掉这一天。

`backend/src/db/attack.rs` 的 `set_issue_progress` 写入 `issue_progress_history`，冲突键是 `issue_url, progress_day`。

## 结果确认

`resolved` 只表示待测试，单仍然开着。`pending` 和 `stale` 退出闭环窗口。生态侧 bug 是 7 个工作日，feature 是 30 个工作日。

`backend/src/ecosystem.rs`

```rust
pub const BUG_WORKDAYS: i64 = 7;
pub const FEATURE_WORKDAYS: i64 = 30;
const CLOSURE_EXEMPT_LABELS: &[&str] = &["pending", "stale"];
// resolved 不在豁免名单里：打了 resolved 仍未关闭的单，还要满足闭环期限。
```

## 超时提醒

工作日每天 09:00 之后发一次。当天发过就不再发。收件人是各组接口人和生态接口人。正文只有标题和链接。

`backend/src/scheduler.rs`

```rust
fn daily_alert_due(now: chrono::DateTime<chrono::Utc>, sent_day: Option<&str>) -> bool {
    let now = now.with_timezone(&chrono_tz::Asia::Shanghai);
    let today = now.date_naive();
    !crate::timeutil::is_weekend(today)
        && now.hour() >= 9
        && sent_day != Some(today.to_string().as_str())
}
```

看板上的 bug 超时从创建日算 5 个工作日。剩下不超过 2 天是黄，不超过 1 天是红，过了截止是已超期。非 bug 没有手工日期就不会进提醒。

`backend/src/db/issues.rs`

```rust
pub const BUG_SLA_WORKDAYS: i64 = 5;

let status = if is_overdue(&due.to_string(), now) {
    "overdue"
} else if remaining <= 1 {
    "red"
} else if remaining <= 2 {
    "yellow"
} else {
    "normal"
};
```

`backend/src/notify.rs`

```rust
let line = format!("- {} {}", issue.text("title"), issue.text("url"));
match sla_status.as_str() {
    "overdue" => overdue.push(line),
    "red" => urgent.push(line),
    "yellow" => watch.push(line),
    _ => {}
}
for person in [group.liaison.as_ref(), group.ecosystem_liaison.as_ref()].into_iter().flatten() {
    // 发给接口人和生态接口人
}
```

## 处理人与归属

Issue 明细上「业务组」和「处理人」是两列。没选人时显示「-」。选中后写入 `issue_owners`。

`backend/src/catalog.rs`

```rust
ColumnSpec { label: "业务组".into(), cell: "text".into(), ..column("group") },
ColumnSpec {
    label: "处理人".into(),
    cell: "issue-owner".into(),
    edit: Some(ColumnEdit { action: "set_owner".into() }),
    ..column("owner_person_name")
},
```

Issue 的组来自仓库绑定：

```rust
FROM issues i
JOIN group_repos gr ON gr.repo = i.project
```

DTS 同一行上有抓来的当前处理人、源系统责任组、手选归属组。归属组空着显示「未指定」。

`backend/src/record.rs`

```rust
text("current_handler", "当前处理人"),
text("subsystem_name", "责任组"),
text("group_name", "归属组"),
```

## 各组剩余问题

Issue 首页是「业务组总览」。待闭环数、已超期、超过目标的组、今日新增在上面。下面「业务组健康度」一行一组，剩余数超过基础目标就标红。点组名进入该组剩下的单。

`frontend/src/views/issues-view.ts`

```ts
<h2>业务组总览</h2>
<strong>${S.snap.total}</strong>          // 待闭环 Issue
<strong class="${group.count > group.threshold ? "over-count" : ""}">${group.count}</strong>
```

这个数只数关注范围内、还没关闭的 bug 类。

`backend/src/db/issues.rs`

```rust
if row.flag("is_bug_class") {
    *counts.entry(group).or_insert(0) += 1;
}
let total: i64 = counts.values().sum();
```

攻关周期另有「攻关进展」表，按组拆成待开发（bug、feature、pending、其他）和待测试（resolved）。
