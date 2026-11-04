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
 * Route table — single source for every path an e2e spec navigates to.
 * Mirrors web_ui/frontend/src/router/index.ts.
 */
export const ROUTES = {
  /** Root; redirects to console. */
  root: '/',
  /** Tabbed workspace (text_generate / video_generate / throughput_optimizer). */
  console: '/console',
  /** Job history list. */
  history: '/history',
  /** Docs page. */
  docs: '/docs',
  /** Deep link: per-job status view. */
  jobStatus: (jobId: string) => `/jobs/${jobId}/status`,
  /** Deep link: per-job result view. */
  jobResult: (jobId: string) => `/jobs/${jobId}/result`,
} as const

/** Backend API paths used for request interception assertions. */
export const API = {
  jobSubmit: '/api/jobs', // POST — create job (payload: module_id, form_schema_version, params)
  modules: '/api/modules', // GET — module list
  formSchema: (moduleId: string) => `/api/modules/${moduleId}/form-schema`, // GET
} as const
