# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""Extra unit tests for ``serving_cast/service/optimizer_summary.py`` (helpers + render_* + uncovered paths)."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from serving_cast.service.optimizer_summary import (
    OptimizerSummary,
    _compute_disagg_request_qps,
    _fmt_optional,
    _get_disagg_table_buf,
    _get_pd_ratio_table_buf,
    _positive_float,
    _sorted_rows,
    render_cross_device_comparison,
    render_cross_hardware_disagg_decode,
    render_cross_hardware_disagg_prefill,
    render_cross_hardware_pd_ratio,
)


class TestPositiveFloat(TestCase):
    def test_positive_float_accept_and_reject(self):
        self.assertEqual(_positive_float(1.5), 1.5)
        self.assertIsNone(_positive_float(0))
        self.assertIsNone(_positive_float(None))
        self.assertIsNone(_positive_float("bad"))


class TestFmtOptionalSortedRows(TestCase):
    def test_fmt_optional_formats_or_dash(self):
        self.assertEqual(_fmt_optional(3.14159), "3.14")
        self.assertEqual(_fmt_optional(None), "-")

    def test_sorted_rows_orders_by_metric(self):
        rows = [{"k": 1}, {"k": 3}, {"k": 2}]
        ordered = _sorted_rows(rows, "k")
        self.assertEqual([r["k"] for r in ordered], [3, 2, 1])


class TestComputeDisaggRequestQps(TestCase):
    def test_prefill_formula(self):
        row = pd.Series({"concurrency": 10.0, "ttft": 50.0, "tpot": None})
        self.assertAlmostEqual(_compute_disagg_request_qps(row, 64), 10.0 / 50.0 * 1000.0)

    def test_decode_formula_requires_output_length(self):
        row = pd.Series({"concurrency": 8.0, "ttft": None, "tpot": 2.0})
        self.assertIsNone(_compute_disagg_request_qps(row, None))
        self.assertIsNone(_compute_disagg_request_qps(row, 0))
        self.assertAlmostEqual(
            _compute_disagg_request_qps(row, 4),
            8.0 / (2.0 * 4.0) * 1000.0,
        )

    def test_returns_none_when_both_ttft_and_tpot(self):
        row = pd.Series({"concurrency": 1.0, "ttft": 1.0, "tpot": 1.0})
        self.assertIsNone(_compute_disagg_request_qps(row, 8))


class TestDisaggPdRatioTableBuf(TestCase):
    def test_disagg_prefill_table_title_and_qps_cell(self):
        df = pd.DataFrame(
            {
                "token/s": [88.0],
                "ttft": [110.0],
                "tpot": [pd.NA],
                "concurrency": [22.0],
                "num_devices": [1],
                "parallel": ["tp2"],
                "batch_size": [1],
            }
        )
        buf = _get_disagg_table_buf(df, output_length=None)
        self.assertRegex(buf, r"PD Disaggregated Prefill Configurations:")
        expected_qps = 22.0 / 110.0 * 1000.0
        self.assertIn(f"{expected_qps:.2f}", buf)

    def test_disagg_decode_table_uses_decode_title(self):
        df = pd.DataFrame(
            {
                "token/s": [50.0],
                "ttft": [pd.NA],
                "tpot": [2.5],
                "concurrency": [5.0],
                "num_devices": [1],
                "parallel": ["tp1"],
                "batch_size": [1],
            }
        )
        buf = _get_disagg_table_buf(df, output_length=4)
        self.assertRegex(buf, r"PD Disaggregated Decode Configurations:")
        expected_qps = 5.0 / (2.5 * 4.0) * 1000.0
        self.assertIn(f"{expected_qps:.2f}", buf)


class TestPdRatioTableBuf(TestCase):
    def test_get_pd_ratio_table_buf_contains_banner_and_columns(self):
        df = pd.DataFrame(
            {
                "pd_ratio": [0.5],
                "balanced_qps": [12.34],
                "p_qps": [10.0],
                "d_qps": [20.0],
                "ttft_p": [30.0],
                "tpot_d": [1.5],
                "parallel_p": ["Pa"],
                "parallel_d": ["Da"],
                "num_devices_p": [4],
                "num_devices_d": [4],
                "batch_size_p": [1],
                "batch_size_d": [2],
                "concurrency_p": [3],
                "concurrency_d": [4],
            }
        )
        buf = _get_pd_ratio_table_buf(df)
        self.assertIn("PD Ratio Configurations:", buf)
        self.assertIn("Balanced QPS", buf)


class TestRenderComparisonTables(TestCase):
    def test_render_helpers_empty_lists(self):
        self.assertEqual(render_cross_device_comparison([]), "")
        self.assertEqual(render_cross_hardware_pd_ratio([]), "")
        self.assertEqual(render_cross_hardware_disagg_prefill([]), "")
        self.assertEqual(render_cross_hardware_disagg_decode([]), "")

    def test_render_cross_device_comparison_non_empty(self):
        txt = render_cross_device_comparison(
            [{"device": "D1", "throughput_tps": 99.9, "concurrency": 1, "parallel": "p", "batch_size": 1, "num_devices": 1}]
        )
        self.assertIn("Cross-hardware", txt)
        self.assertIn("D1", txt)

    def test_render_cross_hardware_pd_ratio_shows_banner(self):
        rows = [
            {
                "device": "X",
                "balanced_qps": 1.23,
                "pd_ratio": 0.25,
                "p_qps": 4.0,
                "d_qps": 1.0,
                "ttft_p": 50.0,
                "tpot_d": 2.0,
                "p_instances": 2,
                "d_instances": 1,
                "total_devices": 8,
            }
        ]
        txt = render_cross_hardware_pd_ratio(rows)
        self.assertIn("PD Ratio", txt)
        self.assertIn("num-devices=", txt.lower())

    def test_render_cross_hardware_disagg_prefill_decode(self):
        pref = render_cross_hardware_disagg_prefill(
            [{"device": "P1", "throughput_tps": 80.0, "qps_req_s": None, "ttft_ms": 100.0, "concurrency": 2}]
        )
        self.assertIn("PD Disaggregated Prefill", pref)
        dec = render_cross_hardware_disagg_decode(
            [{"device": "D2", "throughput_tps": 90.0, "qps_req_s": 1.23, "tpot_ms": 20.0, "concurrency": 3}]
        )
        self.assertIn("PD Disaggregated Decode", dec)


class TestOptimizerSummaryBranches(TestCase):
    def test_report_final_result_silent_returns_immediately(self):
        cfg = SimpleNamespace(ttft_limits=10.0, tpot_limits=10.0, output_length=8)
        s = OptimizerSummary(cfg)
        s.set_summary_df(pd.DataFrame({"token/s": [1.0], "ttft": [1.0], "tpot": [1.0], "concurrency": [1]}))
        with patch("builtins.print") as p:
            s.report_final_result(SimpleNamespace(disagg=False, dump_original_results=False), silent=True)
            p.assert_not_called()

    def test_report_final_result_warns_when_no_summary(self):
        cfg = SimpleNamespace(ttft_limits=10.0, tpot_limits=10.0, output_length=8)
        s = OptimizerSummary(cfg)
        with self.assertLogs("serving_cast.service.optimizer_summary", level="WARNING") as log_ctx:
            s.report_final_result(SimpleNamespace(disagg=False, dump_original_results=False))
        self.assertTrue(any("empty or unset" in m for m in log_ctx.output))

    def test_get_agg_disagg_final_out_empty_after_filters(self):
        cfg = SimpleNamespace(ttft_limits=10.0, tpot_limits=10.0, output_length=None)
        s = OptimizerSummary(cfg)
        s.set_summary_df(
            pd.DataFrame(
                {
                    "token/s": [1.0],
                    "ttft": [1000.0],
                    "tpot": [500.0],
                    "concurrency": [1],
                    "num_devices": [1],
                    "parallel": ["x"],
                    "batch_size": [1],
                }
            )
        )
        args = SimpleNamespace(
            model_id="m",
            num_devices=1,
            device="TEST",
            quantize_linear_action="DISABLED",
            quantize_attention_action="DISABLED",
            disagg=False,
        )
        with self.assertLogs("serving_cast.service.optimizer_summary", level="WARNING") as log_ctx:
            out = s._get_agg_disagg_final_out(args)
        self.assertTrue(any("TTFT/TPOT filters" in m for m in log_ctx.output))
        self.assertIn("No configurations satisfy", "\n".join(out))

    def test_collect_comparison_row_via_best_agg_disagg_row(self):
        cfg = SimpleNamespace(ttft_limits=1000.0, tpot_limits=50.0, output_length=32)
        s = OptimizerSummary(cfg)
        s.set_summary_df(
            pd.DataFrame(
                {
                    "token/s": [10.0, 30.0],
                    "ttft": [90.0, 80.0],
                    "tpot": [5.0, 6.0],
                    "concurrency": [1, 2],
                    "num_devices": [8, 8],
                    "parallel": ["tp1", "tp2"],
                    "batch_size": [1, 1],
                }
            )
        )
        row = s.collect_comparison_row("device_a")
        self.assertEqual(row["device"], "device_a")
        self.assertEqual(row["throughput_tps"], 30.0)

    def test_collect_disagg_prefill_decode_guards(self):
        # Prefill collector requires TTFT limit and forbids simultaneous TPOT limit.
        cfg_no_ttft = SimpleNamespace(ttft_limits=None, tpot_limits=None, output_length=None)
        self.assertIsNone(OptimizerSummary(cfg_no_ttft).collect_disagg_prefill_row("d"))

        cfg_ttft_and_tpot = SimpleNamespace(ttft_limits=100.0, tpot_limits=10.0, output_length=None)
        self.assertIsNone(OptimizerSummary(cfg_ttft_and_tpot).collect_disagg_prefill_row("d"))

        # Decode collector requires TPOT limit and forbids TTFT limit simultaneously.
        cfg_decode_but_ttft_set = SimpleNamespace(ttft_limits=100.0, tpot_limits=10.0, output_length=4)
        self.assertIsNone(OptimizerSummary(cfg_decode_but_ttft_set).collect_disagg_decode_row("d"))

    def test_collect_pd_ratio_comparison_row_needs_pd_mode_and_data(self):
        cfg_plain = SimpleNamespace(ttft_limits=100.0, tpot_limits=10.0, num_devices=None)
        s_plain = OptimizerSummary(cfg_plain)
        s_plain.set_summary_df(pd.DataFrame({"x": [1]}))
        self.assertIsNone(s_plain.collect_pd_ratio_comparison_row("d"))
