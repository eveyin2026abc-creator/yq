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
 * E2E case: video_generate full lifecycle on the 视频生成 tab, as a
 * declarative FormCase (support/case.ts) with one form mutation:
 * 采样步数 (sample-step) 50 → 4.
 *
 * Everything else stays at schema defaults — notably the parallelism trio
 * (num-devices=8, ulysses-size=4, cfg-parallel=on), which satisfies the
 * numDevicesMatchesUlyssesSize validator (8 = 2 × 4 with CFG parallel).
 * Single-value device/quantization defaults → single case → the result pane
 * renders VideoGenerateResult (not the multi-case comparison view).
 * Measured runtime: ~45s on a dev box (the sample-step cut barely moves it —
 * the simulator prices steps analytically).
 */
import { test } from 'playwright/test'
import { runFormCase, fill, type FormCase } from './support/case'
import { FIELDS, MODULES, TEXT } from './support/locators'

const SAMPLE_STEP = 4

/** 采样步数 50 → 4, full lifecycle — pure config, interpreted by runFormCase. */
const sampleStepLifecycle: FormCase = {
  module: MODULES.video,
  fills: [fill(FIELDS.video.sampleStep, SAMPLE_STEP)],
  expectParams: { [FIELDS.video.sampleStep.id]: SAMPLE_STEP },
  result: {
    kind: 'metrics',
    labels: [TEXT.metricVideoTotalTime, TEXT.metricVideoPerfModels, TEXT.metricVideoOpTypes],
  },
}

test.describe('Console · video_generate lifecycle', () => {
  test('sample-step=4 → run → succeeded result → history', async ({ page }) => {
    await runFormCase(page, sampleStepLifecycle)
  })
})
