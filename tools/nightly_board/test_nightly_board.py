#!/usr/bin/env python3

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

    @mock.patch.object(server, "issue_state")
    @mock.patch.object(server, "load_runs")
    def test_issue_sync_failure_is_explicit(self, load_runs, issue_state):
        load_runs.return_value = [{"date": "2026-11-02", "issue_numbers": [586]}]
        issue_state.return_value = {"number": 586, "closed": None, "error": "offline"}
        payload = server.open_issues_payload()
        self.assertFalse(payload["sync_ok"])
        self.assertEqual(payload["sync_errors"][0]["number"], 586)


if __name__ == "__main__":
    unittest.main()
