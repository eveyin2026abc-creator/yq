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
 * E2E cases: text_generate against a REAL backend + vite dev server (started
 * by playwright.config webServer on isolated ports/DB).
 *
 * The lifecycle case is a declarative FormCase (support/case.ts) — dp-size is
 * intentionally left blank there: the backend auto-derives it as
 * num_devices // (tp × pp) = 9, satisfying the tp×dp×pp = num_devices
 * validator. The two remaining cases are non-standard flows (form-only spin
 * buttons; deep-link pages) and stay hand-written against support/actions.ts.
 */
import { expect, test } from 'playwright/test'
import {
  clickNumberSpin,
  expectConsoleReady,
  expectJobResultPage,
  expectJobStatusPage,
  expectJobSucceeded,
  formItem,
  openModule,
  readJobId,
  setComboboxField,
  setMultiSelectField,
  setSelectField,
  setSwitchField,
  submitJob,
} from './support/actions'
import { runFormCase, fill, type FormCase } from './support/case'
import { FIELDS, MODULES, SEL, TEXT } from './support/locators'

const NUM_DEVICES = 9

/** 设备数量 1 → 9, full lifecycle — pure config, interpreted by runFormCase. */
const numDevicesLifecycle: FormCase = {
  module: MODULES.text,
  fills: [fill(FIELDS.text.numDevices, NUM_DEVICES)],
  expectParams: { [FIELDS.text.numDevices.id]: NUM_DEVICES },
  result: {
    kind: 'metrics',
    labels: [TEXT.metricBatchSize, TEXT.metricExecutionTime, TEXT.metricPeakMemory],
  },
}

/** The spinbutton input of a number field (value assertions). */
function spinInput(page: Parameters<typeof formItem>[0], field: typeof FIELDS.text.numDevices) {
  return formItem(page, field).locator(SEL.inputNumberInput)
}

test.describe('Console · text_generate lifecycle', () => {
  test('num-devices=9 → run → succeeded result → history', async ({ page }) => {
    // Job runs a real simulation (~10s); every other step is FormCase config.
    await runFormCase(page, numDevicesLifecycle)
  })

  test('select, combobox, multi-select, and switch commit in the form', async ({ page }) => {
    // Form layer only — no simulation. Covers the four setters colleagues use
    // from fill(), including a switch inside a group that starts collapsed.
    await openModule(page, MODULES.text)
    await setSelectField(page, FIELDS.text.logLevel, 'info')
    await setComboboxField(page, FIELDS.text.modelId, 'Qwen/Qwen3-8B')
    await setMultiSelectField(page, FIELDS.text.device, ['ATLAS_350_425T_84G'])
    await setSwitchField(page, FIELDS.text.prefill, false)
    await setSwitchField(page, FIELDS.text.chromeTrace, true)
  })

  test('spin buttons move 设备数量 by ±1 per click', async ({ page }) => {
    // Focused check of the el-input-number "+/−" buttons — form layer only,
    // no simulation job, so it stays cheap and isolates control regressions
    // from runner regressions.
    await expectConsoleReady(page)

    const input = spinInput(page, FIELDS.text.numDevices)
    await expect(input).toHaveValue('1') // schema default

    await clickNumberSpin(page, FIELDS.text.numDevices, 'increase', 3)
    await expect(input).toHaveValue('4')

    await clickNumberSpin(page, FIELDS.text.numDevices, 'decrease', 1)
    await expect(input).toHaveValue('3')
  })

  test('finished job is reachable via /jobs/:id/status and /jobs/:id/result deep links', async ({
    page,
  }) => {
    // Runs a stock-defaults job (num-devices=1 — the cheapest real simulation)
    // purely to obtain a finished job id, then exercises the two deep-link
    // pages: they must render the SAME job outside the console context
    // (share-by-jobId, not by in-page runner state).
    await expectConsoleReady(page)

    await submitJob(page)
    await expectJobSucceeded(page)
    const jobId = await readJobId(page)

    // Status deep link: succeeded tag + matching job id.
    await expectJobStatusPage(page, jobId)

    // Result deep link: page chrome + the text_generate metric tiles re-render
    // from the stored result envelope (not from console memory).
    await expectJobResultPage(page, jobId, MODULES.text.id, [
      TEXT.metricBatchSize,
      TEXT.metricExecutionTime,
      TEXT.metricPeakMemory,
    ])
  })
})
