#!/usr/bin/env python3

import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server


class NightlyBoardTests(unittest.TestCase):
    def test_day_status_pass(self):
        runs = [{"branch": "master", "label": "master", "started": "2026-11-02 01:00", "exit_code": 0, "failed": 0, "errors": 0}]
        self.assertEqual(server.day_status(runs), "pass")

    def test_day_status_failure(self):
        runs = [{"branch": "master", "label": "master", "started": "2026-11-02 01:00", "exit_code": 3, "failed": 2, "errors": 0}]
        self.assertEqual(server.day_status(runs), "fail")

    def test_day_status_includes_every_branch(self):
        runs = [
            {"branch": "master", "label": "master", "started": "2026-11-02 05:30", "exit_code": 0, "failed": 0, "errors": 0},
            {"branch": "26.2.0", "label": "26.2.0", "started": "2026-11-02 05:40", "exit_code": 1, "failed": 1, "errors": 0},
        ]
        self.assertEqual(server.day_status(runs), "fail")
        self.assertEqual(
            server.day_branches(runs),
            [
                {"branch": "master", "status": "pass", "failed": 0},
                {"branch": "26.2.0", "status": "fail", "failed": 1},
            ],
        )

    def test_archived_day_stays_when_its_log_is_gone(self):
        archived = {
            "date": "2026-11-13",
            "runs": [{
                "log": "nightly-master-20261113-0530.log",
                "branch": "master",
                "label": "master",
                "date": "2026-11-13",
                "started": "2026-11-13 05:30",
                "exit_code": 1,
                "failed": 1,
                "errors": 0,
                "issue_numbers": [648],
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            day = Path(tmp)
            (day / "2026-11-13.json").write_text(json.dumps(archived), encoding="utf-8")
            with mock.patch.object(server, "archive_dir", return_value=day), mock.patch.object(
                server, "LOG_DIR", day / "missing-logs"
            ):
                runs = server.load_runs()
        self.assertEqual([run["date"] for run in runs], ["2026-11-13"])
        self.assertEqual(runs[0]["issue_numbers"], [648])
        self.assertIn("2026-11-13", server.month_payload(2026, 11, runs)["days"])

    def test_log_label_is_the_branch_name(self):
        self.assertEqual(server.LOG_NAME.match("nightly-26.1.0-20261116-0530.log").group("label"), "26.1.0")
        self.assertEqual(server.LOG_NAME.match("nightly-develop-20261116-0530.log").group("label"), "develop")

    @mock.patch.object(server, "issue_state")
    @mock.patch.object(server, "load_runs")
    def test_issue_sync_failure_is_explicit(self, load_runs, issue_state):
        load_runs.return_value = [{"date": "2026-11-02", "issue_numbers": [586]}]
        issue_state.return_value = {"number": 586, "closed": None, "error": "offline"}
        payload = server.open_issues_payload()
        self.assertFalse(payload["sync_ok"])
        self.assertEqual(payload["sync_errors"][0]["number"], 586)

    @mock.patch.object(server, "issue_state")
    def test_previous_day_issue_is_marked_as_repeat(self, issue_state):
        issue_state.return_value = {"number": 586, "closed": False}
        run = {"date": "2026-10-31", "issue_numbers": [586]}
        result = server.attach_issues(run, {586: "2026-10-30"})
        self.assertEqual(result["issues"][0]["repeat_kind"], "previous_day")
        self.assertEqual(result["issues"][0]["first_seen_date"], "2026-10-30")

    def test_same_failed_node_is_a_repeat_and_keeps_its_issue(self):
        node = "tests/benchmark/ops/perf_database/test_empirical_metrics.py::TestModelRunnerProfilingMetrics::test_consecutive_non_pipeline_runs_isolate_empirical_records"
        runs = [
            {"date": "2026-10-21", "branch": "master", "label": "master", "started": "2026-10-21 02:40", "issue_numbers": [500], "failures": [{"node_id": node, "commit": "2509db59", "error": "engram"}]},
            {"date": "2026-11-13", "branch": "master", "label": "master", "started": "2026-11-13 05:30", "issue_numbers": [648], "failures": [{"node_id": node, "commit": "c81710cc", "error": "get_visual"}]},
            {"date": "2026-11-14", "branch": "master", "label": "master", "started": "2026-11-14 05:30", "issue_numbers": [], "failures": [{"node_id": node, "commit": "c81710cc", "error": "get_visual"}]},
            {"date": "2026-11-16", "branch": "master", "label": "master", "started": "2026-11-16 05:30", "issue_numbers": [], "failures": [{"node_id": node, "commit": "c81710cc", "error": "get_visual"}]},
        ]
        with mock.patch.object(server, "issue_state", return_value={"number": 648, "closed": False, "html_url": "https://example/648", "assignees": ["stormchasingg"], "title": "get_visual"}):
            day14 = server.day_payload("2026-11-14", runs)["runs"][0]
            day16 = server.day_payload("2026-11-16", runs)["runs"][0]
        self.assertEqual(day14["failures"][0]["repeat_kind"], "previous_day")
        self.assertEqual(day14["failures"][0]["first_seen_date"], "2026-11-13")
        self.assertEqual(day14["issues"][0]["number"], 648)
        self.assertEqual(day14["issues"][0]["repeat_kind"], "previous_day")
        self.assertEqual(day16["failures"][0]["repeat_kind"], "historical")
        self.assertEqual(day16["failures"][0]["first_seen_date"], "2026-11-13")
        self.assertEqual(day16["issues"][0]["number"], 648)
        self.assertEqual(day16["issues"][0]["first_seen_date"], "2026-11-13")

    def test_new_failed_node_stays_untracked(self):
        runs = [
            {"date": "2026-11-13", "branch": "master", "label": "master", "started": "2026-11-13 05:30", "issue_numbers": [648], "failures": [{"node_id": "tests/a.py::test_old"}]},
            {"date": "2026-11-14", "branch": "master", "label": "master", "started": "2026-11-14 05:30", "issue_numbers": [], "failures": [{"node_id": "tests/b.py::test_new"}]},
        ]
        day = server.day_payload("2026-11-14", runs)["runs"][0]
        self.assertNotIn("repeat_kind", day["failures"][0])
        self.assertEqual(day["issues"], [])
        self.assertEqual(day["loop"], "未建 issue")

    def test_same_failure_on_another_branch_is_not_a_repeat(self):
        node = "tests/example.py::test_case"
        runs = [
            {"date": "2026-11-13", "branch": "master", "label": "master", "started": "2026-11-13 05:30", "issue_numbers": [648], "failures": [{"node_id": node, "commit": "c81710cc"}]},
            {"date": "2026-11-14", "branch": "26.2.0", "label": "26.2.0", "started": "2026-11-14 05:30", "issue_numbers": [], "failures": [{"node_id": node, "commit": "c81710cc"}]},
        ]
        day = server.day_payload("2026-11-14", runs)["runs"][0]
        self.assertNotIn("repeat_kind", day["failures"][0])
        self.assertEqual(day["issues"], [])
        self.assertEqual(day["branch"], "26.2.0")


if __name__ == "__main__":
    unittest.main()
