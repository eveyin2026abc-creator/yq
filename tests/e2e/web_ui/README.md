# Web Console E2E (Playwright)

End-to-end specs for the msmodeling web console (`web_ui/`). Each spec drives
a **real** stack: FastAPI backend (repo venv python, isolated SQLite dir) +
vite dev server, both started and torn down automatically by
`playwright.config.ts` (`webServer`). No manual startup needed.

This directory is a **self-contained npm package** (specs live outside
`web_ui/frontend`, so they cannot resolve its `node_modules` — Node walks UP
the tree only). It reuses the app's repo venv for the backend and the global
`ms-playwright` browser cache; only the `playwright` runner itself is
installed here.

## Run

```bash
cd tests/e2e/web_ui
npm install                 # once (~ playwright + typescript)
npm run test                # headless chromium, full stack auto-managed
npm run test:ui             # interactive watch mode
npm run type-check          # tsc over specs + config
npx playwright show-report  # last HTML report (CI reporter)
```

Prereqs:

- repo venv provisioned (`uv sync` at repo root — the backend runs on it);
- `npm install` done in `web_ui/frontend` (the vite dev server + app deps);
- Playwright chromium browser: `npx playwright install chromium` once
  (shared global cache — no re-download if `web_ui/frontend` already uses
  the same playwright minor version).

## Isolation from your dev stack

| Concern  | Dev stack        | E2E stack                                    |
|----------|------------------|----------------------------------------------|
| backend  | port 8000        | port **8901** (`MSMODELING_E2E_BACKEND_PORT`) |
| frontend | port 5173        | port **5901** (`MSMODELING_E2E_FRONTEND_PORT`)|
| SQLite   | `.msmodeling_ui` | fresh `msm-e2e-ui-*` dir under the OS temp, per run |

The fresh-DB-per-run matters: the backend dedups identical submissions by
params hash, so a leftover `succeeded` job would make resubmits return
instantly without exercising the runner. Stale temp dirs are swept at the
START of the next run (Playwright runs teardown hooks before stopping the
webServers, so end-of-run deletion would race a live backend holding SQLite
locks — one leftover dir between runs is by design).

## Layout

```text
tests/e2e/web_ui/
├── support/
│   ├── routes.ts     # path variables: frontend routes + API paths
│   ├── locators.ts   # node variables: structural selectors (SEL), bilingual
│   │                 #   text patterns (TEXT), and the form-field registry
│   │                 #   (FIELDS — id + control component + zh/en label)
│   ├── actions.ts    # operation functions: expectConsoleReady, openModule,
│   │                 #   setNumberField, setTextField, submitJob,
│   │                 #   expectJobSucceeded, …
│   └── case.ts       # declarative layer: FormCase config type + control→
│                     #   setter factory (applyFill) + generic runFormCase
├── console-text-generate.e2e.ts   # specs = FormCase configs + hand-written
├── console-video-generate.e2e.ts  #   non-standard flows (one file per
├── console-throughput-optimizer.e2e.ts # console module tab)
├── playwright.config.ts           # stack bring-up + temp-DB management
└── tsconfig.json                  # standalone type-check: npm run type-check
```

## Conventions

- Spec files end in `.e2e.ts` — `vitest run` (default include `*.spec.ts` /
  `*.test.ts`) must not collect them, and `vue-tsc` build only covers `src/`.
- **No absolute xpaths.** Fields are anchored by label text on `.el-form-item`
  (`el-form-item` does NOT render its `prop` as a DOM attribute — see
  SchemaFormItem.vue), components by app-owned structural classes (`.run-btn`,
  `.state-succeeded`, `.metric-label`) or Element Plus classes. The registry in
  `support/locators.ts` documents which component each selector belongs to, so
  a UI refactor is one edit there.
- `FIELDS` entries must match `web_ui/backend/services/ui_props/*.py`
  (labels, control types) — that module is the form-schema source of truth.
- Match UI text bilingually (`/文本生成|Text Generation/`): zh is the app
  default locale, en keeps specs robust to a locale flip.
- Assert submits through the intercepted `POST /api/jobs` payload
  (`submitJob` returns the Request) — proves the control's v-model committed,
  not just what the DOM shows.
- **3 workers, file-level parallelism** (`fullyParallel: false`): the three
  spec files run concurrently against the shared backend; tests within a file
  stay sequential (the two heavy throughput benchmarks never overlap). Backend
  job slots match the worker count (`MSMODELING_MAX_WORKERS=3` in
  playwright.config) so no file's job queues behind another's. Every job still
  runs a real simulation.
- Parallel-safe assertions: History rows are located **by job id**
  (`expectHistoryRow`) — "newest row" races once files overlap on the shared
  backend/DB. Everything else is per-page or append-only by construction.
- Form-layer behaviors (spin buttons, validation) get job-free specs; keep
  simulation-running specs few — each runs a real job (measured: text_generate
  ~10s, throughput_optimizer ~40s with tp-sizes pinned, video_generate ~45s;
  full suite ~1.2min in parallel, bounded by the throughput file's two
  sequential benchmarks).
