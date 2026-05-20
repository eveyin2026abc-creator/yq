# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""Unit tests for ``serving_cast/service/optimizer_curve_plots.py`` helpers and plot entry points."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from serving_cast.service import optimizer_curve_plots as ocp


class TestCurvePlotHelpers(TestCase):
    def test_axis_metric_name_strips_unit_suffix(self):
        self.assertEqual(ocp._axis_metric_name("Throughput (token/s)"), "Throughput")
        self.assertEqual(ocp._axis_metric_name("foo"), "foo")

    def test_parallel_label_truncates_long_string(self):
        short = "tp2pp1dp2"
        self.assertEqual(ocp._parallel_label(short), short)
        long_p = "x" * 60
        got = ocp._parallel_label(long_p)
        self.assertEqual(len(got), 48)
        self.assertTrue(got.endswith("..."))

    def test_padded_axis_limits_empty_and_values(self):
        self.assertIsNone(ocp._padded_axis_limits([]))
        lim = ocp._padded_axis_limits([10.0])
        self.assertIsNotNone(lim)
        self.assertEqual(lim[1] - lim[0] > 0, True)
        lim2 = ocp._padded_axis_limits([0.0, 10.0])
        self.assertIsNotNone(lim2)
        self.assertGreaterEqual(lim2[0], 0.0)

    def test_compact_scatter_legend_collapses_double_marker(self):
        label = "parallel_a"
        marker = ocp._TERMINAL_MARKER
        line = f"| {marker}{marker} {label} |"
        compacted = ocp._compact_scatter_legend(line, [label])
        self.assertIn(f"{marker}{label}", compacted)
        self.assertNotIn(f"{marker}{marker} {label}", compacted)

    def test_jitter_overlapping_points_offsets_duplicates(self):
        xs = [1.0, 1.0, 2.0]
        ys = [3.0, 3.0, 4.0]
        out = ocp._jitter_overlapping_points(xs, ys)
        self.assertEqual(len(out), 3)
        self.assertNotEqual(out[0], out[1])

    def test_sorted_curve_subset_sorts_by_batch_and_concurrency(self):
        df = pd.DataFrame(
            {
                "parallel": ["p1", "p1", "p1"],
                "concurrency": [2, 1, 1],
                "batch_size": [2, 1, 2],
                "token/s": [10.0, 20.0, 15.0],
                "tpot": [5.0, 5.0, 5.0],
            }
        )
        sub = ocp._sorted_curve_subset(df, "p1", ["concurrency", "batch_size", "tpot"])
        self.assertEqual(sub.iloc[0]["concurrency"], 1)
        self.assertEqual(sub.iloc[0]["batch_size"], 1)

    def test_memory_filter_drops_non_positive_when_column_present(self):
        df = pd.DataFrame(
            {
                "parallel": ["a", "b"],
                "concurrency": [1, 1],
                "token/s": [1.0, 2.0],
                "tpot": [1.0, 1.0],
                "memory_left_gb": [1.0, -0.1],
            }
        )
        filt = ocp._memory_filter(df.copy())
        self.assertEqual(len(filt), 1)
        self.assertEqual(filt.iloc[0]["parallel"], "a")

    def test_require_columns_raises(self):
        df = pd.DataFrame({"parallel": []})
        with self.assertRaises(ValueError) as ctx:
            ocp._require_columns(df, {"parallel", "tpot"}, "missing")
        self.assertIn("tpot", str(ctx.exception))

    def test_sort_curve_df_empty(self):
        self.assertTrue(ocp._sort_curve_df(pd.DataFrame()).empty)

    def test_prepare_latency_curve_df_drops_na_and_sorts(self):
        df = pd.DataFrame(
            {
                "parallel": ["tp1", "tp1"],
                "concurrency": [1.0, 2.0],
                "token/s": [10.0, float("nan")],
                "tpot": [30.0, 20.0],
            }
        )
        work = ocp._prepare_latency_curve_df(
            df,
            latency_col="tpot",
            missing_message="test_missing",
        )
        self.assertEqual(len(work), 1)
        self.assertAlmostEqual(work.iloc[0]["token/s"], 10.0)


class TestPdTpsAndMerge(TestCase):
    def test_pd_tps_curve_df_computes_token_per_s(self):
        df = pd.DataFrame(
            {
                "parallel_d": ["d1", "d1"],
                "concurrency_d": [100.0, 200.0],
                "tpot_d": [10.0, 20.0],
            }
        )
        out = ocp._pd_tps_curve_df(df)
        self.assertIn("token/s", out.columns)
        self.assertAlmostEqual(out.loc[out["parallel"] == "d1", "token/s"].iloc[0], 10000.0)

    def test_pd_tps_curve_df_drops_non_positive_tpot(self):
        df = pd.DataFrame(
            {
                "parallel_d": ["d1"],
                "concurrency_d": [100.0],
                "tpot_d": [0.0],
            }
        )
        out = ocp._pd_tps_curve_df(df)
        self.assertTrue(out.empty)


class TestPlotEntryPoints(TestCase):
    def test_plot_concurrency_curves_from_optimizer_summaries_empty(self):
        self.assertFalse(
            ocp.plot_concurrency_curves_from_optimizer_summaries(
                [],
                basename_prefix="x",
                ttft_limit=None,
                tpot_limit=None,
            )
        )

    def test_emit_curve_df_empty_returns_false(self):
        with self.assertLogs(ocp.logger, level="WARNING") as logctx:
            ok = ocp._emit_curve_df(
                pd.DataFrame(),
                title_prefix="t",
                skip_label="unittest empty",
            )
        self.assertFalse(ok)
        self.assertTrue(any("no rows after filtering" in m for m in logctx.output))

    @patch.object(ocp, "_emit_terminal_optimizer_curve_ascii")
    def test_plot_concurrency_optimizer_curves_success(self, mock_emit):
        df = pd.DataFrame(
            {
                "parallel": ["tp2pp1dp1"],
                "concurrency": [4.0],
                "batch_size": [1],
                "token/s": [12.34],
                "tpot": [18.0],
            }
        )
        self.assertTrue(
            ocp.plot_concurrency_optimizer_curves(df, basename_prefix="unit_pref", ttft_limit=None, tpot_limit=None)
        )
        mock_emit.assert_called_once()

    @patch.object(ocp, "_emit_terminal_optimizer_curve_ascii")
    def test_plot_concurrency_optimizer_curves_value_error_returns_false(self, mock_emit):
        df = pd.DataFrame({"parallel": []})
        self.assertFalse(
            ocp.plot_concurrency_optimizer_curves(df, basename_prefix="bad", ttft_limit=None, tpot_limit=None)
        )
        mock_emit.assert_not_called()

    def test_first_non_empty_summary_df(self):
        empty = MagicMock()
        empty.get_summary_df.return_value = None
        nonempty = MagicMock()
        nonempty.get_summary_df.return_value = pd.DataFrame({"x": [1]})
        self.assertIsNotNone(ocp._first_non_empty_summary_df([empty, nonempty]))
        only_empty_df = MagicMock()
        only_empty_df.get_summary_df.return_value = pd.DataFrame()
        self.assertIsNone(ocp._first_non_empty_summary_df([empty, only_empty_df]))


class TestRenderCrossHardwareSummary(TestCase):
    @patch.object(ocp, "render_hardware_profile_comparison", return_value="")
    @patch.object(ocp, "render_cross_device_comparison", return_value="")
    def test_render_cross_hardware_summary_skips_single_device(
        self, _mock_render_table, _mock_render_hw
    ):
        args = MagicMock()
        args.disagg = False
        args.enable_optimize_prefill_decode_ratio = False
        rows = ocp.MultiDeviceComparisonRows()
        logger = MagicMock()
        ocp.render_cross_hardware_summary(args, ["only_one"], rows, logger=logger)
        _mock_render_table.assert_not_called()

    @patch("builtins.print")
    @patch.object(ocp, "render_hardware_profile_comparison", return_value="hw")
    @patch.object(ocp, "render_cross_device_comparison", return_value="table")
    def test_render_cross_hardware_summary_prints_when_multi_device(
        self, _mock_render_table, _mock_render_hw, _mock_print
    ):
        args = MagicMock()
        args.disagg = False
        args.enable_optimize_prefill_decode_ratio = False
        rows = ocp.MultiDeviceComparisonRows(aggregation=[{"device": "a"}])
        logger = MagicMock()
        ocp.render_cross_hardware_summary(args, ["d1", "d2"], rows, logger=logger)
        self.assertTrue(_mock_print.called)
