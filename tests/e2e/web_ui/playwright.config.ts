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
 * Playwright e2e config for the web console.
 *
 * The e2e stack is fully isolated from a developer's running dev stack:
 *   - backend  -> port 8901 (MSMODELING_PORT), fresh per-run SQLite temp dir
 *   - frontend -> vite dev on port 5901, proxying /api to 8901
 * Both are started/stopped by Playwright's `webServer` (see below); no manual
 * setup needed. Override ports via MSMODELING_E2E_BACKEND_PORT /
 * MSMODELING_E2E_FRONTEND_PORT if 8901/5901 are taken.
 *
 * Spec files use the `.e2e.ts` suffix (testMatch below) — a naming convention
 * kept from when the suite lived inside web_ui/frontend (vitest isolation);
 * it also keeps the specs visually distinct from the Python tests around them.
 *
 * Run (from this directory — it is a self-contained npm package; the specs
 * live outside web_ui/frontend and cannot resolve its node_modules):
 *   npm install               # once
 *   npm run test              # headless chromium
 *   npx playwright test --ui  # interactive
 */
import { defineConfig, devices } from 'playwright/test'
import { mkdtempSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const e2eDir = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(e2eDir, '../../..')
const frontendDir = path.join(repoRoot, 'web_ui', 'frontend')

const BACKEND_PORT = Number(process.env.MSMODELING_E2E_BACKEND_PORT ?? 8901)
const FRONTEND_PORT = Number(process.env.MSMODELING_E2E_FRONTEND_PORT ?? 5901)

// Fresh backend state per run: a unique temp SQLite dir so jobs, history and
// the params-hash dedup cache never leak between runs (a cached "succeeded"
// job would make submit return instantly without exercising the runner).
//
// Cleanup model — "sweep on next start", NOT teardown: Playwright runs
// globalTeardown BEFORE stopping the webServers, so the backend is still
// alive (SQLite locked) when teardown executes and deletion cannot succeed.
// Instead, the first config evaluation in the main process removes stale
// msm-e2e-ui-* dirs from previous runs (their processes are long gone), then
// creates this run's dir and pins it into the env. Worker processes re-
// evaluate this module but inherit MSMODELING_E2E_UI_DIR and skip all
// filesystem side effects.
const UI_DIR_PREFIX = 'msm-e2e-ui-'
let uiDir = process.env.MSMODELING_E2E_UI_DIR
if (!uiDir) {
  for (const entry of readdirSync(tmpdir())) {
    if (entry.startsWith(UI_DIR_PREFIX)) {
      rmSync(path.join(tmpdir(), entry), { recursive: true, force: true, maxRetries: 2 })
    }
  }
  uiDir = mkdtempSync(path.join(tmpdir(), UI_DIR_PREFIX))
  process.env.MSMODELING_E2E_UI_DIR = uiDir
}

// The backend must run with the repo venv python (it owns the simulation deps).
const venvPython =
  process.platform === 'win32'
    ? path.join(repoRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(repoRoot, '.venv', 'bin', 'python')

export default defineConfig({
  testDir: '.',
  testMatch: '**/*.e2e.ts',
  // File-level parallelism: the 3 spec files run concurrently against the
  // shared backend; tests within a file stay sequential (the two throughput
  // benchmarks are the heaviest jobs — never run them against each other).
  // Keep workers == MSMODELING_MAX_WORKERS (below) so no file's job queues
  // behind another file's, and assert History rows BY JOB ID (expectHistoryRow)
  // — "newest row" races once files overlap.
  fullyParallel: false,
  workers: 3,
  // Cold vite compile + schema fetch + a ~20s simulation job, with headroom.
  timeout: 240_000,
  expect: { timeout: 15_000 },
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI
    ? [
        ['list'],
        ['html', { open: 'never' }],
        ['json', { outputFile: process.env.MSMODELING_E2E_JSON || 'e2e-results.json' }],
      ]
    : [['list']],
  use: {
    baseURL: `http://127.0.0.1:${FRONTEND_PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      // Backend from the backend dir (its imports are top-level packages
      // rooted there — same contract as `python main.py` in the dev docs).
      command: `"${venvPython}" main.py`,
      cwd: path.join(repoRoot, 'web_ui', 'backend'),
      env: {
        MSMODELING_PORT: String(BACKEND_PORT),
        MSMODELING_UI_DIR: uiDir,
        // One job slot per parallel spec file (workers above). Note each
        // throughput job additionally spawns its own ProcessPoolExecutor —
        // raise this only on boxes that can take the load (see main.py).
        MSMODELING_MAX_WORKERS: '3',
      },
      url: `http://127.0.0.1:${BACKEND_PORT}/api/modules`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      stdout: 'pipe',
      stderr: 'pipe',
    },
    {
      // Vite dev server; `--` forwards the port flags to vite itself.
      command: `npm run dev -- --port ${FRONTEND_PORT} --strictPort`,
      cwd: frontendDir,
      env: {
        MSMODELING_BACKEND_HOST: '127.0.0.1',
        MSMODELING_BACKEND_PORT: String(BACKEND_PORT),
        MSMODELING_FRONTEND_HOST: '127.0.0.1',
      },
      url: `http://127.0.0.1:${FRONTEND_PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      stdout: 'pipe',
      stderr: 'pipe',
    },
  ],
})
