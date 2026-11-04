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
 * Declarative form-case layer — "cases as data, steps as code".
 *
 * A lifecycle case (open module → mutate form → submit → succeeded → result →
 * history) is fully described by a typed `FormCase` object; `runFormCase` is
 * its interpreter. Specs shrink to one config literal per case, while the
 * assertion menu stays extensible plain TS (no DSL): a new result shape is a
 * new `result` variant here, not a rewrite of every spec.
 *
 * Division of labor:
 *   locators.ts — WHAT to point at (fields, selectors, text, modules);
 *   actions.ts  — the atomic verbs (openModule, setNumberField, submitJob, …);
 *   case.ts     — the factory + generic composition of those verbs (this file);
 *   *.e2e.ts    — case configs; only genuinely non-standard flows (deep links,
 *                 spin buttons) stay hand-written against actions.ts.
 *
 * Type safety: `FillStep` pairs each field with the value type its control
 * takes (FieldDef<'number'> ↔ number), so a string into a spinbutton — or a
 * fill for a control the factory doesn't support yet — is a COMPILE error,
 * not a runtime surprise.
 */
import { expect, type Page } from 'playwright/test'
import {
  expectBestConfigCard,
  expectHistoryRow,
  expectJobSucceeded,
  expectMetricTiles,
  openModule,
  readJobId,
  setNumberField,
  setTextField,
  submitJob,
} from './actions'
import type { FieldDef, ModuleDef } from './locators'

/** Controls that have a registered setter (add rows as specs need them —
 *  deliberately no speculative switch/select/combobox support yet). */
export type FillableControl = 'number' | 'text'

/** The value type each fillable control takes. */
export type FillValue<C extends FillableControl> = C extends 'number' ? number : string

/**
 * One form mutation. Opaque to configs — build it with `fill()` below, whose
 * generic ties the value type to the field's control at COMPILE time.
 * (TS cannot correlate `value` with a discriminant nested inside `field`, so
 * a union-of-shapes FillStep doesn't narrow in the interpreter; the factory
 * moves the guarantee to the config site instead.)
 */
export interface FillStep {
  readonly field: FieldDef<FillableControl>
  readonly value: number | string
}

/**
 * Fill-step factory: `fill(FIELDS.video.sampleStep, 4)` ✓,
 * `fill(FIELDS.video.sampleStep, '4')` ✗ (compile error), and a field whose
 * control has no setter (select/switch/…) fails the generic bound.
 */
export function fill<C extends FillableControl>(
  field: FieldDef<C>,
  value: FillValue<C>,
): FillStep {
  return { field, value }
}

/**
 * Factory dispatch: apply one fill through the setter matching its control.
 * The `fill()` constructor guarantees field/value coherence, so the
 * interpreter only switches on the control; the `never` default makes adding
 * a FillableControl without a setter branch a compile error.
 */
export async function applyFill(page: Page, step: FillStep): Promise<void> {
  const { field, value } = step
  switch (field.control) {
    case 'number':
      return setNumberField(page, field, value as number)
    case 'text':
      return setTextField(page, field, value as string)
    default: {
      const unhandled: never = field.control
      throw new Error(`applyFill: no setter for control '${unhandled}'`)
    }
  }
}

/** What the succeeded result pane must show, per module's result component. */
export type ResultExpectation =
  /** Metric tiles (text/video single-case results). */
  | { kind: 'metrics'; labels: readonly RegExp[] }
  /** Mode badge + best-config card metrics (throughput optimizer). */
  | { kind: 'bestConfig'; mode: RegExp; labels: readonly RegExp[] }

/**
 * A full lifecycle case as pure data. module.id doubles as the expected
 * POST module_id AND the History row's 模块 column value (they are the same
 * id by construction — benchmark jobs submit under their own module id).
 */
export interface FormCase {
  /** Console tab to run on; also the schema-ready gate via openModule. */
  module: ModuleDef
  /** Form mutations, applied in order after the form is ready. */
  fills: readonly FillStep[]
  /**
   * Params entries that must appear verbatim in the intercepted POST
   * payload — proves each fill committed into the v-model, not just the DOM.
   * module_id is always asserted from `module`, so it needs no entry here.
   */
  expectParams?: Record<string, unknown>
  /** Succeeded-result-pane assertion. */
  result: ResultExpectation
}

/**
 * Interpret a FormCase end-to-end against the real stack:
 * open tab → apply fills → submit (payload asserted) → wait succeeded →
 * assert result pane → assert History row. Each step delegates to the
 * corresponding action; this function adds sequencing and the payload
 * assertions only.
 */
export async function runFormCase(page: Page, c: FormCase): Promise<void> {
  // 1. Module tab opens with its schema-driven form prefilled from backend.
  await openModule(page, c.module)

  // 2. Mutate the form via the control→setter factory.
  for (const step of c.fills) {
    await applyFill(page, step)
  }

  // 3. Submit; the intercepted POST proves which runner fired and that every
  //    fill really committed into params.
  const submit = await submitJob(page)
  const body = submit.postDataJSON()
  expect(body.module_id, 'POST /api/jobs module_id').toBe(c.module.id)
  for (const [key, value] of Object.entries(c.expectParams ?? {})) {
    expect(body.params?.[key], `POST /api/jobs params['${key}']`).toEqual(value)
  }

  // 4. Real simulation job must end succeeded (throws fast with the on-page
  //    error text if it failed instead); grab its id for the History check.
  await expectJobSucceeded(page)
  const jobId = await readJobId(page)

  // 5. Result pane: module-specific success view.
  if (c.result.kind === 'metrics') {
    await expectMetricTiles(page, [...c.result.labels])
  } else {
    await expectBestConfigCard(page, c.result.mode, [...c.result.labels])
  }

  // 6. History lists THIS job as succeeded (row located by id — parallel
  //    spec files share the backend, so "newest row" would race).
  await expectHistoryRow(page, jobId, c.module.id)
}
