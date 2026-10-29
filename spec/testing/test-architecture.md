# 测试责任分层与迁移门禁

## 1. 责任层

- L1（model）必须使用真实模型身份或模型执行器，以 family 为复用边界。L1 必须使用
  `L1ScenarioExecutor`，一次 build/run 产物供 numeric、structural、capability 三组断言复用。
- L2（module）必须使用 `tests/helpers/` 中经评审的 canonical synthetic builder，禁止
  调用真实模型执行入口。没有 canonical builder 的新构建物必须先提案，不得在用例内复制。
- L3（unit）必须验证纯函数、解析、校验、异常或退出码。
- `cov_helper` 不是第四层，而是 L3 的透明标签。它不得成为安全、精度、性能路径的唯一 watcher。

## 2. 基线与清单

存量清单必须从目标 PR 的 canonical `master` 运行
`pytest tests --collect-only -q -o addopts=` 生成，以完整 node ID 为主键。当前路径只能写
master 路径；目标路径单独记录。未合入用例不得混入存量清单。

CSV 是事实源。对账必须输出 missing、extra、renamed 三份 diff，missing 和 extra 非空时阻断。

## 3. 断言守恒

迁移前，每个 L1 family 必须在 `tests/.ci/assertion_manifest.csv` 逐条登记 numeric、
structural、capability 三组断言及对应产品符号。迁移后 active assertion ID 集合不得缩小。
确需放弃时，必须把条目标为 waived，并填写 reason 和 approver；无签字的删除阻断合入。

## 4. 增量预算

L1 增量预算单位是唯一 `(build_signature, run_signature)` 场景，不是 pytest node 数。
`tests/.ci/l1_scenarios.csv` 必须记录 family、场景、build/run 签名、实际 build/forward
计数、实测秒数与调度范围。每个场景必须满足一次 build、一次 forward；所有 incremental
场景实测总时长不得超过 180 秒。

## 5. 分批迁移与回滚

L1 迁移按 Qwen、dense、MoE、多模态四批进行。每批必须独立通过：collect 对账、assertion
manifest 守恒、build/forward 计数、test_map watcher 不缩水，以及适用的 coverage-context
对比。任一门禁失败只回滚当前 family 批次。

多模态机制没有 canonical builder 时必须落到 L1（真实单 block）或 L3（纯规则），不得
保留“待定”或临时 L2 fixture。

## 6. cov_helper 准入

`cov_helper` 与 L3 共用 unit 目录。单条目标耗时小于 50 ms，必须有真实返回值、异常或类型
断言，docstring 必须用 `COV_HELPER: product/path.py::symbol` 映射产品符号，禁止 mock 被测
产品对象。nightly 报告必须拆分核心覆盖率与 cov_helper 贡献，总覆盖率口径保持不变。

## 7. 机械执行

`scripts/helpers/test_architecture.py` 校验断言 manifest、场景预算、新增测试路由与清单对账。
`scripts/run_ci_gate.sh` 在 pytest 前运行这些门禁。例外沿用 `tests/.ci/gate_policy.yaml`
exemptions，必须填写 reason、applicant、approver、deadline。

## 8. 已迁移范围与执行证据

本轮把 tensor_cast、CLI 和 model_diagnostics 范围内的存量文件实际迁入 `model/`、
`module/`、`unit/`。四个 L1 family 的离线真实场景位于
`tests/regression/tensor_cast/model/test_l1_family_scenarios.py`，覆盖 Qwen2.5、Qwen3 Dense、
Qwen3 MoE、DeepSeek V3/V3.2、Kimi K2、MiniMax M2、Qwen3-VL 与 GLM-4.5V。
`tests/.ci/l1_test_map.json` 是这些场景用 `--cov-context=test` 生成的 watcher 证据；
`l1_watchers.csv` 中的必要产品符号必须继续出现在对应 family 的映射中。

不在本轮责任分层范围的 4758 条基线节点不表示“不重要”。其维护边界记录在
`tests/.ci/out_of_scope_boundaries.csv`：web_ui、optix、serving_cast 保持各自子系统语义；
scripts 与 helpers 由测试基础设施维护；smoke 继续作为跨层发布守卫；Operator-DB 继续作为
ST 数据基线。它们不会因本轮重构被移动或降级，后续由对应维护者单独评审。
