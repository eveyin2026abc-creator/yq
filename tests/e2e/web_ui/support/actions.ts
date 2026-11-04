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
 * Operation functions — the reusable verbs of the e2e suite. Specs compose
 * these; raw selectors stay in support/locators.ts. Each function is a thin,
 * side-effect-only wrapper over Playwright locators (no hidden state), so a
 * spec reads as a sequence of domain steps.
 */
import { expect, type Locator, type Page, type Request } from 'playwright/test'
import { API, ROUTES } from './routes'
import { LOCALE, MODULES, SEL, TEXT, type FieldDef, type ModuleDef } from './locators'

// --- locator factories (pure: page + def -> Locator) -----------------------

/** Resolve a field's display label in the active UI locale. */
export function fieldLabel(field: FieldDef): string {
  return field.label[LOCALE]
}

/** Escape a literal string for use inside a RegExp. */
function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * The `.el-form-item` wrapping a schema field, anchored by its exact label
 * text. Exact-match regex avoids substring collisions between labels.
 */
export function formItem(page: Page, field: FieldDef): Locator {
  const label = page.locator(SEL.formItemLabel, {
    hasText: new RegExp(`^${escapeRegExp(fieldLabel(field))}$`),
  })
  return page.locator(SEL.formItem).filter({ has: label })
}

/** A console tab by its bilingual title. */
export function tab(page: Page, title: RegExp): Locator {
  return page.locator(SEL.consoleTabs).locator(SEL.tabItem, { hasText: title })
}

/** True for the POST /api/jobs submit request (for payload interception). */
export function isJobSubmitRequest(request: Request): boolean {
  return request.method() === 'POST' && new URL(request.url()).pathname === API.jobSubmit
}

// --- page-level operations --------------------------------------------------

/**
 * Wait until a module's freshly-mounted workspace is usable: the model-id
 * combobox shows the schema default (proves the form-schema round-trip) and
 * the result pane is idle. The generous timeout covers vite's cold transform
 * + schema fetch on first hit.
 */
async function expectModuleFormReady(
  page: Page,
  module: ModuleDef,
  timeoutMs: number,
): Promise<void> {
  await expect(
    formItem(page, module.modelField).getByText(module.defaultModel).first(),
    `form should prefill the default model-id (${module.defaultModel}) from the backend form-schema`,
  ).toBeVisible({ timeout: timeoutMs })

  await expect(page.locator(SEL.stateIdle)).toBeVisible()
}

/**
 * Open the console and wait until it is fully usable on the DEFAULT tab
 * (text_generate): three module tabs rendered, schema-driven form prefilled
 * from the backend, result pane in its idle state.
 */
export async function expectConsoleReady(page: Page, timeoutMs = 90_000): Promise<void> {
  await page.goto(ROUTES.console)
  await expect(tab(page, TEXT.tabTextGenerate)).toBeVisible()
  await expect(tab(page, TEXT.tabVideoGenerate)).toBeVisible()
  await expect(tab(page, TEXT.tabThroughputOptimizer)).toBeVisible()
  await expectModuleFormReady(page, MODULES.text, timeoutMs)
}

/**
 * Open the console and switch to `module`'s tab, waiting until ITS form is
 * ready. Console.vue mounts only the active module's workspace (:key remount),
 * so after the switch every form/result selector resolves within that module.
 */
export async function openModule(
  page: Page,
  module: ModuleDef,
  timeoutMs = 90_000,
): Promise<void> {
  await page.goto(ROUTES.console)
  const target = tab(page, module.tab)
  await expect(target).toBeVisible()
  await target.click()
  await expectModuleFormReady(page, module, timeoutMs)
}

/**
 * Set a `control: 'number'` field (el-input-number) to an exact value.
 * Types into the spinbutton input and commits with Enter — el-input-number
 * applies the change on change/blur, not per keystroke.
 */
export async function setNumberField(page: Page, field: FieldDef, value: number): Promise<void> {
  if (field.control !== 'number') {
    throw new Error(`setNumberField: ${field.id} renders as '${field.control}', not 'number'`)
  }
  const input = formItem(page, field).locator(SEL.inputNumberInput)
  await input.click()
  await input.fill(String(value))
  await input.press('Enter')
  await expect(input).toHaveValue(String(value))
}

/**
 * Set a `control: 'text'` field (el-input) to an exact string. fill() fires
 * the input event el-input's v-model listens on; Enter additionally blurs
 * any popper so later clicks land on the form, not an overlay.
 */
export async function setTextField(page: Page, field: FieldDef, value: string): Promise<void> {
  if (field.control !== 'text') {
    throw new Error(`setTextField: ${field.id} renders as '${field.control}', not 'text'`)
  }
  const input = formItem(page, field).locator(SEL.inputText)
  await input.click()
  await input.fill(value)
  await input.press('Enter')
  await expect(input).toHaveValue(value)
}

/**
 * Click an el-input-number spin button (`+`/`−`) `times` times — exercises
 * the button path specifically (contrast with setNumberField's fill path).
 */
export async function clickNumberSpin(
  page: Page,
  field: FieldDef,
  direction: 'increase' | 'decrease',
  times = 1,
): Promise<void> {
  if (field.control !== 'number') {
    throw new Error(`clickNumberSpin: ${field.id} renders as '${field.control}', not 'number'`)
  }
  const button = formItem(page, field).locator(
    direction === 'increase' ? SEL.inputNumberIncrease : SEL.inputNumberDecrease,
  )
  for (let i = 0; i < times; i += 1) {
    await button.click()
  }
}

/**
 * Click 运行 and capture the POST /api/jobs request. The returned Request
 * lets specs assert the submitted payload (e.g. that a form mutation really
 * committed into params) via `request.postDataJSON()`.
 */
export async function submitJob(page: Page): Promise<Request> {
  const pending = page.waitForRequest(isJobSubmitRequest)
  await page.locator(SEL.runButton).click()
  return pending
}

/**
 * Wait for the current job to reach `succeeded`. If it fails instead, throw
 * immediately with the on-page error text (the alert description is the
 * fastest triage signal — no need to dig through the trace).
 */
export async function expectJobSucceeded(page: Page, timeoutMs = 180_000): Promise<void> {
  // Reach SOME lifecycle state first — a selector regression that leaves the
  // pane idle should fail fast (30s) instead of burning the full job timeout.
  const terminal = `${SEL.stateRunning}, ${SEL.stateSucceeded}, ${SEL.stateFailed}`
  await expect(page.locator(terminal)).toBeVisible({ timeout: 30_000 })

  const failed = page.locator(SEL.stateFailed)
  if (await failed.isVisible().catch(() => false)) {
    const detail = await failed.innerText().catch(() => '(unreadable)')
    throw new Error(`job failed:\n${detail}`)
  }

  await expect(page.locator(SEL.stateSucceeded)).toBeVisible({ timeout: timeoutMs })
}

/**
 * Assert the succeeded result pane: 已完成 tag plus the given metric tiles
 * (anchored label regexes from TEXT).
 */
export async function expectMetricTiles(page: Page, labels: RegExp[]): Promise<void> {
  const result = page.locator(SEL.resultPane)
  await expect(result.getByText(TEXT.stateCompleted)).toBeVisible()
  for (const label of labels) {
    await expect(result.locator(SEL.metricLabel, { hasText: label })).toBeVisible()
  }
}

/**
 * Open History and assert `jobId`'s row shows `moduleId` with the given status
 * text (default: succeeded). The row is located BY JOB ID — not "newest row":
 * spec files run in parallel against the shared backend, where another
 * worker's job can always claim a newer row mid-assertion.
 */
export async function expectHistoryRow(
  page: Page,
  jobId: string,
  moduleId: string,
  statusText: RegExp = TEXT.historyStatusSucceeded,
): Promise<void> {
  await page.goto(ROUTES.history)
  await expect(page.getByRole('heading', { name: TEXT.historyHeading })).toBeVisible()
  const row = page.locator(SEL.historyRow, { hasText: jobId })
  await expect(row, 'History should list exactly one row for this job id').toHaveCount(1)
  await expect(row).toContainText(moduleId)
  await expect(row).toContainText(statusText)
}

/**
 * Assert the throughput optimizer's succeeded result: the mode badge (e.g.
 * 聚合 for benchmark pd_mixed) plus the best-config card carrying the given
 * metric labels. Scoped to `.best-config-card` because the sweep table below
 * reuses column captions like 输出吞吐.
 */
export async function expectBestConfigCard(
  page: Page,
  modeLabel: RegExp,
  labels: RegExp[],
): Promise<void> {
  const result = page.locator(SEL.resultPane)
  await expect(result.getByText(TEXT.stateCompleted)).toBeVisible()
  await expect(result.locator(SEL.modeBadge, { hasText: modeLabel })).toBeVisible()
  const card = result.locator(SEL.bestConfigCard)
  await expect(card).toBeVisible()
  for (const label of labels) {
    await expect(card.locator(SEL.metricLabel, { hasText: label })).toBeVisible()
  }
}

/** Read the finished job's id from the console success strip (32-hex). */
export async function readJobId(page: Page): Promise<string> {
  const el = page.locator(SEL.jobIdMono)
  await expect(el).toBeVisible()
  const jobId = (await el.innerText()).trim()
  expect(jobId, 'success strip should expose the 32-hex job id').toMatch(/^[0-9a-f]{32}$/)
  return jobId
}

/**
 * Deep-link /jobs/:id/status: the header card shows the succeeded status tag
 * and the same job id.
 */
export async function expectJobStatusPage(page: Page, jobId: string): Promise<void> {
  await page.goto(ROUTES.jobStatus(jobId))
  const card = page.locator(SEL.statusMainCard)
  await expect(card.locator(SEL.statusTag)).toHaveText(TEXT.historyStatusSucceeded)
  await expect(card.locator(SEL.jobIdText)).toContainText(jobId)
}

/**
 * Deep-link /jobs/:id/result: page chrome (heading + job/module meta) plus
 * the module's result component rendering the given metric labels.
 */
export async function expectJobResultPage(
  page: Page,
  jobId: string,
  moduleId: string,
  metricLabels: RegExp[],
): Promise<void> {
  await page.goto(ROUTES.jobResult(jobId))
  await expect(page.locator(SEL.pageTitle)).toHaveText(TEXT.jobResultHeading)
  await expect(page.locator(SEL.jobIdText)).toContainText(jobId)
  await expect(page.locator(SEL.moduleIdMeta)).toContainText(moduleId)
  for (const label of metricLabels) {
    await expect(page.locator(SEL.metricLabel, { hasText: label })).toBeVisible()
  }
}
