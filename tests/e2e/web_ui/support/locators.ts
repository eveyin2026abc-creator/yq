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
 * Locator registry — "nodes as variables".
 *
 * Selector strategy (deliberately NOT absolute xpaths):
 *   1. app-owned structural classes (`.run-btn`, `.state-succeeded`,
 *      `.metric-label`) defined in ModuleWorkspace.vue / ResultPane.vue /
 *      TextGenerateResult.vue — stable across layout tweaks;
 *   2. Element Plus component classes (`.el-form-item`, `.el-input-number`)
 *      for library-rendered chrome;
 *   3. the field's LABEL TEXT to pin a specific form item — SchemaFormItem
 *      binds `:prop="field.id"` but el-form-item does NOT render prop as a
 *      DOM attribute (see the comment in SchemaFormItem.vue), so the label
 *      is the only reliable per-field anchor in the DOM.
 *
 * Every entry documents which component renders it, so a UI refactor that
 * swaps e.g. el-input-number for something else points straight at the
 * constants that need updating — one edit here, all specs follow.
 */

/** Structural selectors, keyed by role in the page. */
export const SEL = {
  // --- Console layout (Console.vue / ModuleWorkspace.vue) ---
  consoleTabs: '.console-tabs', // el-tabs header wrapper
  tabItem: '.el-tabs__item', // el-tabs tab
  formPane: '.pane-form', // form section of the workspace
  resultPane: '.pane-result', // result section of the workspace
  runButton: '.run-btn', // el-button(type=primary) — submit job

  // --- Job states (ResultPane.vue) ---
  stateIdle: '.state-idle', // el-empty hint before any submit
  stateRunning: '.state-running', // spinner + 取消/日志 buttons
  stateSucceeded: '.state-succeeded', // el-tag(已完成) + result component
  stateFailed: '.state-failed', // el-alert(任务失败) + error detail

  // --- Result metrics (TextGenerateResult.vue) ---
  metricLabel: '.metric-label', // metric tile caption (批次大小/执行时间/…)
  metricValue: '.metric-value', // metric tile number

  // --- Schema form (SchemaFormItem.vue → Element Plus) ---
  formItem: '.el-form-item', // one rendered schema field
  formItemLabel: '.el-form-item__label', // its caption (label-text anchor)
  inputNumber: '.el-input-number', // control="number" (controls-position=right)
  inputNumberInput: '.el-input-number input[role="spinbutton"]',
  inputNumberIncrease: '.el-input-number__increase', // "+" spin button
  inputNumberDecrease: '.el-input-number__decrease', // "−" spin button
  inputText: '.el-input__inner', // control="text" (el-input)
  select: '.el-select', // control="select" / "combobox"
  multiSelect: '.msm-multi-select', // control="multi-select"
  switch: '.el-switch', // control="switch"

  // --- Throughput result (ThroughputOptimizerResult.vue) ---
  modeBadge: '.mode-badge', // 聚合 / 分离部署 / PD 配比优化 badge
  bestConfigCard: '.best-config-card', // best-config metric card

  // --- Console success strip (ResultPane.vue) ---
  jobIdMono: '.job-id-mono', // finished job's id (32-hex)

  // --- Job status deep-link page (JobStatusHeader.vue) ---
  statusMainCard: '.main-card', // header card of /jobs/:id/status
  statusTag: '.status-tag', // el-tag with the lifecycle status text
  jobIdText: '.job-id', // "Job ID: <id>" line

  // --- Job result deep-link page (JobResult.vue) ---
  pageTitle: '.page-title', // h2 作业结果
  moduleIdMeta: '.module-id', // "模块: <module_id>" line

  // --- History page (History.vue) ---
  historyHeading: 'h2', // 任务历史 heading
  historyRow: '.el-table__row', // el-table job row
} as const

/** Bilingual UI text patterns (zh is the app default locale). */
export const TEXT = {
  tabTextGenerate: /文本生成|Text Generation/,
  tabVideoGenerate: /视频生成|Video Generation/,
  tabThroughputOptimizer: /吞吐优化|Throughput Optimizer/,
  runButton: /运行|Run/,
  stateCompleted: /已完成|Completed/,
  historyHeading: /任务历史|Job History/,
  historyStatusSucceeded: /成功|Succeeded/,
  jobResultHeading: /作业结果|Job Result/,
  // Anchored: unanchored 执行时间 also matches the hint （含编译，非模型执行时间）.
  metricBatchSize: /^批次大小$|^Batch Size$/,
  metricExecutionTime: /^执行时间$|^Execution Time$/,
  metricPeakMemory: /^峰值内存$|^Peak Memory$/,
  metricTotalDeviceMemory: /^设备总内存$|^Total Device Memory$/,
  // VideoGenerateResult.vue summary tiles.
  metricVideoTotalTime: /^总执行时间$|^Total Execution Time$/,
  metricVideoPerfModels: /^性能模型$|^Performance Models$/,
  metricVideoOpTypes: /^算子种类$|^Operator Types$/,
  // ThroughputOptimizerResult.vue best-config card (aggregation mode).
  metricOutputThroughput: /^输出吞吐$|^Output Throughput$/,
  metricInputThroughput: /^输入吞吐$|^Input Throughput$/,
  metricQps: /^QPS$/,
  modeAggregated: /^聚合$|^Aggregated$/,
} as const

/** Element Plus control a schema field renders as (SchemaFormItem.vue). */
export type Control = 'number' | 'text' | 'select' | 'combobox' | 'multi-select' | 'switch'

/**
 * A schema form field: id + rendered control + official bilingual label.
 * The control type parameter lets fill-step configs type-check the value
 * against the control (number field ↔ number value) at compile time;
 * it defaults to the full union for control-agnostic consumers (formItem).
 */
export interface FieldDef<C extends Control = Control> {
  /** Schema field id — equals the CLI param name in the submit payload. */
  id: string
  /** Which Element Plus component renders it (drives the action helper). */
  control: C
  /** Label exactly as defined in backend/services/ui_props/*.py. */
  label: { zh: string; en: string }
}

/** Active UI locale — mirrors DEFAULT_LOCALE in src/composables/useLocale.ts. */
export const LOCALE: 'zh' | 'en' = 'zh'

/**
 * Per-module field registries. Labels/controls MUST match the schema source of
 * truth in web_ui/backend/services/ui_props/ — note the SAME field id can have
 * DIFFERENT labels across modules (num-devices is 设备数量 in text_generate but
 * 设备数 in video_generate; model-id is 模型 ID everywhere except the
 * throughput benchmark form, which says 模型名称). Add entries as specs need
 * them — don't speculative-fill.
 */
export const FIELDS = {
  /** ui_props/text_generate.py */
  text: {
    modelId: {
      id: 'model-id',
      control: 'combobox',
      label: { zh: '模型 ID', en: 'Model ID' },
    },
    device: {
      id: 'device',
      control: 'multi-select',
      label: { zh: '目标设备', en: 'Target Device' },
    },
    numDevices: {
      id: 'num-devices',
      control: 'number',
      label: { zh: '设备数量', en: 'Number of Devices' },
    },
    numQueries: {
      id: 'num-queries',
      control: 'number',
      label: { zh: '并行请求数', en: 'Number of Queries' },
    },
    queryLength: {
      id: 'query-length',
      control: 'number',
      label: { zh: '输入序列长度', en: 'Input Sequence Length' },
    },
    contextLength: {
      id: 'context-length',
      control: 'number',
      label: { zh: '上下文长度', en: 'Context Length' },
    },
    dpSize: {
      id: 'dp-size',
      control: 'number',
      label: { zh: '数据并行', en: 'Data Parallel Size' },
    },
  },
  /** ui_props/video_generate.py */
  video: {
    modelId: {
      id: 'model-id',
      control: 'combobox',
      label: { zh: '模型 ID', en: 'Model ID' },
    },
    numDevices: {
      id: 'num-devices',
      control: 'number',
      label: { zh: '设备数', en: 'World Size' },
    },
    sampleStep: {
      id: 'sample-step',
      control: 'number',
      label: { zh: '采样步数', en: 'Sample Steps' },
    },
  },
  /**
   * ui_props/throughput_optimizer_benchmark.py — the console's 吞吐优化 tab
   * defaults to Benchmark mode (ModuleWorkspace formMode ref), so the visible
   * form is the 14-field benchmark schema, not the ~60-field full one.
   */
  throughput: {
    modelId: {
      id: 'model-id',
      control: 'combobox',
      label: { zh: '模型名称', en: 'Model Name' },
    },
    numDevices: {
      id: 'num-devices',
      control: 'number',
      label: { zh: '设备数量', en: 'Number of Devices' },
    },
    tpSizes: {
      id: 'tp-sizes',
      control: 'text',
      label: { zh: 'TP 搜索尺寸', en: 'TP Search Sizes' },
    },
    ppSizes: {
      id: 'pp-sizes',
      control: 'text',
      label: { zh: 'PP 搜索尺寸', en: 'PP Search Sizes' },
    },
    inputLength: {
      id: 'input-length',
      control: 'number',
      label: { zh: '输入长度', en: 'Input Length' },
    },
    outputLength: {
      id: 'output-length',
      control: 'number',
      label: { zh: '输出长度', en: 'Output Length' },
    },
    prefixCacheHitRate: {
      id: 'prefix-cache-hit-rate',
      control: 'number',
      label: { zh: 'PrefixCache 命中率', en: 'Prefix Cache Hit Rate' },
    },
    ttftLimit: {
      id: 'ttft-limit',
      control: 'number',
      label: { zh: 'TTFT SLO (ms)', en: 'TTFT SLO (ms)' },
    },
    tpotLimit: {
      id: 'tpot-limit',
      control: 'number',
      label: { zh: 'TPOT SLO (ms)', en: 'TPOT SLO (ms)' },
    },
    maxConcurrency: {
      id: 'max-concurrency',
      control: 'number',
      label: { zh: '最大并发', en: 'Maximum Concurrency' },
    },
  },
} as const satisfies Record<string, Record<string, FieldDef>>

/** A console module tab: how to open it and how to know its form loaded. */
export interface ModuleDef {
  /** module_id as POSTed to /api/jobs (and shown in History's 模块 column). */
  id: string
  /** Console tab title (bilingual). */
  tab: RegExp
  /** The module's model-id field (label differs per module). */
  modelField: FieldDef
  /** Default model chip proving the schema round-trip. */
  defaultModel: string
}

export const MODULES = {
  text: {
    id: 'text_generate',
    tab: TEXT.tabTextGenerate,
    modelField: FIELDS.text.modelId,
    defaultModel: 'Qwen/Qwen3-32B',
  },
  video: {
    id: 'video_generate',
    tab: TEXT.tabVideoGenerate,
    modelField: FIELDS.video.modelId,
    defaultModel: 'Wan-AI/Wan2.1-T2V-14B-Diffusers',
  },
  /** 吞吐优化 tab submits under the benchmark schema's module id. */
  throughputBenchmark: {
    id: 'throughput_optimizer_benchmark',
    tab: TEXT.tabThroughputOptimizer,
    modelField: FIELDS.throughput.modelId,
    defaultModel: 'Qwen/Qwen3-32B',
  },
} as const satisfies Record<string, ModuleDef>
