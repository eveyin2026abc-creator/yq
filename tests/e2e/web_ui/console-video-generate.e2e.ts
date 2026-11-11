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
 * declarative FormCase (support/case.ts): local model id, and
 * 采样步数 (sample-step) 50 → 4.
 *
 * The model id is the vendored fixture under tests/assets/model_config,
 * not the schema default Hub id. Nightly must not download Diffusers
 * configs while the job runs. Everything else stays at schema defaults —
 * notably the parallelism trio (num-devices=8, ulysses-size=4,
 * cfg-parallel=on), which satisfies the numDevicesMatchesUlyssesSize
 * validator (8 = 2 × 4 with CFG parallel). Single-value device/quantization
 * defaults → single case → the result pane renders VideoGenerateResult
 * (not the multi-case comparison view).
 */
import { test } from 'playwright/test'
import { runFormCase, fill, type FormCase } from './support/case'
import { FIELDS, MODULES, TEXT } from './support/locators'

const SAMPLE_STEP = 4
const LOCAL_MODEL = 'Wan2.2-T2V-A14B-Diffusers'

/** Local fixture + 采样步数 50 → 4. Interpreted by runFormCase. */
const sampleStepLifecycle: FormCase = {
  module: MODULES.video,
  fills: [
    fill(FIELDS.video.modelId, LOCAL_MODEL),
    fill(FIELDS.video.sampleStep, SAMPLE_STEP),
  ],
  expectParams: {
    [FIELDS.video.modelId.id]: LOCAL_MODEL,
    [FIELDS.video.sampleStep.id]: SAMPLE_STEP,
  },
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
