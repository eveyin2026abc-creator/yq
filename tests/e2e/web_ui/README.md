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
# Nightly runs this same command after the pytest waves (CI=1, 900s).
# It skips before npm install when this directory has no *.e2e.ts,
# and when pytest already timed out.
# Pytest counts on the Nightly exit= line stay pytest-only;
# e2e_passed / e2e_failed are appended only when the suite ran.
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
- Chromium needs `libasound.so.2`. Nightly prepends
  `~/.local/playwright-libs/usr/lib/<triplet>` when that extract is present.

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
│   │                 #   setNumberField, setTextField, setSelectField,
│   │                 #   setComboboxField, setMultiSelectField, setSwitchField,
│   │                 #   submitJob, expectJobSucceeded, …
│   └── case.ts       # declarative layer: FormCase config type + control→
│                     #   setter factory (applyFill) + generic runFormCase
├── console-text-generate.e2e.ts
├── console-video-generate.e2e.ts
├── console-throughput-optimizer.e2e.ts
├── *.e2e.ts                       # further specs, one file per console module
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
  Add a field when a spec uses it. `fill()` accepts `number`, `text`,
  `select`, `combobox`, `multi-select`, and `switch`. Pass the schema value;
  set `options` on the field only when the visible option text differs
  (log level `info` is shown as `Info`). Collapsed groups are expanded by the setter.
- Match UI text bilingually (`/文本生成|Text Generation/`): zh is the app
  default locale, en keeps specs robust to a locale flip.
- Assert submits through the intercepted `POST /api/jobs` payload
  (`submitJob` returns the Request) — proves the control's v-model committed,
  not just what the DOM shows.
- **3 workers, file-level parallelism** (`fullyParallel: false`): spec files
  run concurrently against the shared backend; tests within a file stay
  sequential. Put at most one heavy simulation in each file. Backend job
  slots match the worker count (`MSMODELING_MAX_WORKERS=3` in
  playwright.config). A fourth module file must raise `workers` and
  `MSMODELING_MAX_WORKERS` together. Every submitted job still runs a real
  simulation.
- Parallel-safe assertions: History rows are located **by job id**
  (`expectHistoryRow`) — "newest row" races once files overlap on the shared
  backend/DB. Everything else is per-page or append-only by construction.
- Form-layer behaviors (spin buttons, validation) get job-free specs. Keep
  simulation-running specs few — each one runs a real job.
