/*
 * -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

/**
 * E2E cases: throughput_optimizer full lifecycle on the 吞吐优化 tab, as
 * declarative FormCases (support/case.ts). The tab defaults to BENCHMARK
 * mode — a 14-field form submitted under module id
 * `throughput_optimizer_benchmark` (ModuleWorkspace expands the convention
 * fields pd-mode/quantize-preset before the POST), which is exactly the id
 * MODULES.throughputBenchmark carries, so the payload AND History assertions
 * both key off it. pd-mode stays 聚合 (pd_mixed) → aggregation result view:
 * mode badge + best-config card + sweep table.
 *
 * Case 1 (minimal): tp-sizes "" → "4". The empty default triggers an auto
 *   power-of-2 sweep (tp ∈ {1,2,4} under num-devices=4); pinning "4"
 *   collapses the search to a single parallel combo (~40s measured).
 * Case 2 (rich): nine params across all three benchmark form groups —
 *   request lengths, cache-hit rate, both SLO limits, concurrency cap and a
 *   fixed (tp=2, pp=2) combo under num-devices=4 (dp derived = 1). SLOs are
 *   deliberately LOOSE (5000ms / 500ms): the case asserts the constrained
 *   pipeline end-to-end, not SLO filtering itself — tight limits that no
 *   config meets would empty best_config and turn the case into a flaky
 *   probe of simulator performance.
 */
import { test } from 'playwright/test'
import { runFormCase, fill, type FormCase } from './support/case'
import { FIELDS, MODULES, TEXT } from './support/locators'

const TP_SIZES = '4'

/** tp-sizes → single-combo search, full lifecycle — pure config. */
const benchmarkLifecycle: FormCase = {
  module: MODULES.throughputBenchmark,
  fills: [fill(FIELDS.throughput.tpSizes, TP_SIZES)],
  expectParams: { [FIELDS.throughput.tpSizes.id]: TP_SIZES },
  result: {
    kind: 'bestConfig',
    mode: TEXT.modeAggregated,
    labels: [TEXT.metricOutputThroughput, TEXT.metricInputThroughput, TEXT.metricQps],
  },
}

const F = FIELDS.throughput

/** Nine-param sweep of the benchmark form: request / SLO / constraint / search groups. */
const richParams: FormCase = {
  module: MODULES.throughputBenchmark,
  fills: [
    // 通用: 4 devices, search pinned to the single (tp=2, pp=2) combo → dp=1.
    fill(F.numDevices, 4),
    fill(F.tpSizes, '2'),
    fill(F.ppSizes, '2'),
    // 请求: short prompt/completion keeps the real optimization run ~30s.
    fill(F.inputLength, 1024),
    fill(F.outputLength, 256),
    // PrefixCache hit rate — float into an el-input-number ([0,1) validator).
    fill(F.prefixCacheHitRate, 0.2),
    // SLO: loose limits — exercise the constrained path, never filter everything out.
    fill(F.ttftLimit, 5000),
    fill(F.tpotLimit, 500),
    // 约束: cap the exponential concurrency search.
    fill(F.maxConcurrency, 64),
  ],
  expectParams: {
    [F.numDevices.id]: 4,
    [F.tpSizes.id]: '2',
    [F.ppSizes.id]: '2',
    [F.inputLength.id]: 1024,
    [F.outputLength.id]: 256,
    [F.prefixCacheHitRate.id]: 0.2,
    [F.ttftLimit.id]: 5000,
    [F.tpotLimit.id]: 500,
    [F.maxConcurrency.id]: 64,
  },
  result: {
    kind: 'bestConfig',
    mode: TEXT.modeAggregated,
    labels: [TEXT.metricOutputThroughput, TEXT.metricInputThroughput, TEXT.metricQps],
  },
}

test.describe('Console · throughput_optimizer lifecycle', () => {
  test('benchmark tp-sizes=4 → run → succeeded best-config → history', async ({ page }) => {
    await runFormCase(page, benchmarkLifecycle)
  })

  test('benchmark rich params (9 fields: lengths/SLO/concurrency/tp+pp) → succeeded best-config', async ({
    page,
  }) => {
    await runFormCase(page, richParams)
  })
})
