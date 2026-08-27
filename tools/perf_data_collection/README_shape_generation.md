# Shape 网格生成

`generate_shape_grid.py` 会针对目标 HuggingFace 模型自动运行多组 throughput optimizer，捕获性能数据库实际收到的
Kernel 查询，并把可 replay 的 Shape 追加到数据库 CSV。用户显式指定但查询未命中的 `--ops` 会自动使用通用理论规则
生成，结果统一供 microbench 实测回填。

```powershell
python tools/perf_data_collection/generate_shape_grid.py `
  --database-path <性能数据库目录> `
  --target-models <HuggingFace模型ID> `
  --rows 1000
```

公开参数只有：

- `--database-path`：必选，目标数据库目录；
- `--rows`：每个 CSV 本轮最多新增行数，默认 1000；
- `--target-models`：必选，可传一个或多个 HF 模型 ID；
- `--ops`：可选，指定最终需要扩展的 op replay 支持算子；
- `--seed`：可选，控制候选稳定顺序，默认 0。

已有行、重复行和拒绝行不占 `--rows`。以相同参数再次运行会继续补后续候选，而不是重新采样同一批行。

`--ops` 的优先级高于模型查询。例如模型查询到 `A/B/C/D`，而用户传入 `--ops C D E`，最终只生成 `C/D/E`：
`C/D` 使用查询网格，未查询到的 `E` 使用通用理论网格。未传 `--ops` 时，才以模型实际查询到的 replay 算子作为生成集合。
理论配置明确标记为 `skip` 的算子不会兜底生成。数据库不区分 Shape 来源。

工具内部会自动运行单卡长度/batch 基线、TP/EP/MoE-DP/DCP/MTP 分轴扫描、少量并行交叉边界、长序列
chunked-prefill，以及数据库支持时的 compile+SP+DFC。W8A8 动态量化之外还会覆盖 BF16 基线和代表性的 INT8 KV
cache；这些策略不增加新的公开参数。

生成结束后再运行 microbench：

```powershell
python tools/perf_data_collection/start_microbench.py `
  --database-path <性能数据库目录> `
  --update-mode missing-only
```

Shape 网格生成不执行 NPU 实测，新增行的耗时为 0。A3 replay、coverage、插值误差和 text_generate B2B 仍需单独验收。

详细设计见 [查询驱动的 Shape 网格生成设计](../../docs/design/query_driven_shape_grid_generation.md)。
