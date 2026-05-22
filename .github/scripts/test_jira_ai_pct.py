#!/usr/bin/env python3
"""
test_jira_ai_pct.py — local test harness for jira_ai_pct.py

Modes:
    python3 test_jira_ai_pct.py           # run all unit tests
    python3 test_jira_ai_pct.py --dryrun  # integration dry-run: real git-ai, mocked Jira writes

Dry-run requires env vars:
    PR_TITLE, PR_BASE_SHA, PR_HEAD_SHA
    JIRA_BASE_URL, JIRA_USER, JIRA_TOKEN  (reads Jira, skips writes)
"""
import importlib
import json
import os
import sys
import traceback
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── load the module under test ────────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import jira_ai_pct as m

# ── helpers ───────────────────────────────────────────────────────────────────
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


# ── unit tests ────────────────────────────────────────────────────────────────

class TestExtractJiraId(unittest.TestCase):
    def test_space_separator(self):
        self.assertEqual(m.extract_jira_id("ABC-123 do something"), "ABC-123")

    def test_colon_separator(self):
        self.assertEqual(m.extract_jira_id("ABC-123: do something"), "ABC-123")

    def test_dash_separator(self):
        self.assertEqual(m.extract_jira_id("ABC-123 - do something"), "ABC-123")

    def test_multi_digit_project(self):
        self.assertEqual(m.extract_jira_id("MYPROJ-9999 fix bug"), "MYPROJ-9999")

    def test_no_key(self):
        self.assertIsNone(m.extract_jira_id("no jira key here"))

    def test_lowercase_not_matched(self):
        self.assertIsNone(m.extract_jira_id("abc-123 lower"))

    def test_leading_space_stripped(self):
        self.assertEqual(m.extract_jira_id("  XY-7 title"), "XY-7")

    def test_no_separator_after_key(self):
        self.assertIsNone(m.extract_jira_id("ABC-123title"))


class TestGitAiStats(unittest.TestCase):
    def _run(self, stdout, returncode=0):
        proc = MagicMock()
        proc.stdout = stdout
        proc.returncode = returncode
        with patch("subprocess.run", return_value=proc) as sp:
            result = m.git_ai_stats("abc..def")
            sp.assert_called_once_with(
                ["git-ai", "stats", "abc..def", "--json"],
                capture_output=True, text=True,
            )
        return result

    def _stats_payload(self, ai=42, added=100, deleted=10):
        return {
            "authorship_stats": {"total_commits": 9, "commits_with_authorship": 4},
            "range_stats": {
                "ai_additions": ai,
                "git_diff_added_lines": added,
                "git_diff_deleted_lines": deleted,
            },
        }

    def test_valid_json(self):
        payload = self._stats_payload()
        self.assertEqual(self._run(json.dumps(payload)), payload)

    def test_empty_output(self):
        self.assertEqual(self._run(""), {})

    def test_invalid_json(self):
        self.assertEqual(self._run("not json at all"), {})

    def test_partial_json(self):
        payload = {"range_stats": {"ai_additions": 5}}
        self.assertEqual(self._run(json.dumps(payload)), payload)


class TestJiraGetFields(unittest.TestCase):
    def _mock_response(self, body: dict, status: int = 200):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = json.dumps(body).encode()
        resp.status = status
        return resp

    def _call(self, fields_body):
        resp = self._mock_response({"fields": fields_body})
        with patch("urllib.request.urlopen", return_value=resp):
            return m.jira_get_fields("https://x.atlassian.net", "ABC-1",
                                      ["26954", "26955", "26956"], "u", "t")

    def test_all_fields_present(self):
        result = self._call({
            "customfield_26954": "75.5",
            "customfield_26955": "200",
            "customfield_26956": "50",
        })
        self.assertEqual(result, {"26954": 75.5, "26955": 200.0, "26956": 50.0})

    def test_missing_field_returns_none(self):
        result = self._call({"customfield_26954": "50.0"})
        self.assertIsNone(result["26955"])
        self.assertIsNone(result["26956"])

    def test_null_field_returns_none(self):
        result = self._call({"customfield_26954": None})
        self.assertIsNone(result["26954"])

    def test_http_error_returns_none(self):
        import urllib.error
        err = urllib.error.HTTPError("url", 404, "Not Found", {}, StringIO())
        err.read = lambda: b"not found"
        with patch("urllib.request.urlopen", side_effect=err):
            result = m.jira_get_fields("https://x", "ABC-1", ["26954"], "u", "t")
        self.assertIsNone(result["26954"])


class TestJiraSetFields(unittest.TestCase):
    def test_success(self):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = b""
        with patch("urllib.request.urlopen", return_value=resp):
            m.jira_set_fields("https://x", "ABC-1", {"26954": 80.0}, "u", "t")

    def test_http_error_raises(self):
        import urllib.error
        err = urllib.error.HTTPError("url", 400, "Bad Request", {}, StringIO())
        err.read = lambda: b"bad"
        with patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(urllib.error.HTTPError):
                m.jira_set_fields("https://x", "ABC-1", {"26954": 80.0}, "u", "t")


class TestMain(unittest.TestCase):
    BASE_ENV = {
        "PR_TITLE":      "ABC-42: implement feature",
        "PR_BASE_SHA":   "aaa",
        "PR_HEAD_SHA":   "bbb",
        "JIRA_BASE_URL": "https://x.atlassian.net",
        "JIRA_USER":     "user@example.com",
        "JIRA_TOKEN":    "tok",
    }

    def _run_main(self, env, stats, existing_fields):
        """Run main() with mocked subprocess and HTTP calls."""
        with patch.dict("os.environ", env, clear=True), \
             patch.object(m, "git_ai_stats", return_value=stats) as mock_stats, \
             patch.object(m, "jira_get_fields", return_value=existing_fields) as mock_get, \
             patch.object(m, "jira_set_fields") as mock_set:
            m.main()
        return mock_stats, mock_get, mock_set

    def test_no_jira_key_skips(self):
        env = {**self.BASE_ENV, "PR_TITLE": "chore: no key here"}
        with patch.dict("os.environ", env, clear=True), \
             patch.object(m, "git_ai_stats") as mock_stats:
            with self.assertRaises(SystemExit) as ctx:
                m.main()
            self.assertEqual(ctx.exception.code, 0)
            mock_stats.assert_not_called()

    def _make_stats(self, ai, added, deleted):
        return {"range_stats": {"ai_additions": ai, "git_diff_added_lines": added, "git_diff_deleted_lines": deleted}}

    def test_no_diff_lines_skips(self):
        stats = self._make_stats(0, 0, 0)
        with patch.dict("os.environ", self.BASE_ENV, clear=True), \
             patch.object(m, "git_ai_stats", return_value=stats), \
             patch.object(m, "jira_set_fields") as mock_set:
            with self.assertRaises(SystemExit) as ctx:
                m.main()
            self.assertEqual(ctx.exception.code, 0)
            mock_set.assert_not_called()

    def test_first_time_sets_value(self):
        stats = self._make_stats(80, 100, 20)
        existing = {"26954": None, "26955": None, "26956": None}
        _, _, mock_set = self._run_main(self.BASE_ENV, stats, existing)
        mock_set.assert_called_once()
        updates = mock_set.call_args[0][2]
        self.assertEqual(updates["26954"], 80.0)   # 80/100 * 100
        self.assertEqual(updates["26955"], 100.0)  # new adds
        self.assertEqual(updates["26956"], 20.0)   # new dels

    def test_averages_existing_ai_pct(self):
        stats = self._make_stats(100, 100, 0)
        existing = {"26954": 60.0, "26955": 50.0, "26956": 10.0}
        _, _, mock_set = self._run_main(self.BASE_ENV, stats, existing)
        updates = mock_set.call_args[0][2]
        self.assertEqual(updates["26954"], 80.0)   # avg(60, 100)
        self.assertEqual(updates["26955"], 150.0)  # 50 + 100
        self.assertEqual(updates["26956"], 10.0)   # 10 + 0

    def test_stats_called_with_correct_range(self):
        stats = self._make_stats(50, 100, 5)
        existing = {"26954": None, "26955": None, "26956": None}
        mock_stats, _, _ = self._run_main(self.BASE_ENV, stats, existing)
        mock_stats.assert_called_once_with("aaa..bbb")


# ── dry-run integration mode ──────────────────────────────────────────────────

def dryrun():
    """Run end-to-end with real git-ai and real Jira reads, but mock writes."""
    required = ["PR_TITLE", "PR_BASE_SHA", "PR_HEAD_SHA", "JIRA_USER", "JIRA_TOKEN"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"Missing env vars for dry-run: {', '.join(missing)}")
        sys.exit(1)

    print("=== DRY-RUN mode: Jira writes are suppressed ===\n")

    written = {}

    def fake_set(base_url, issue_key, updates, user, token):
        written["issue_key"] = issue_key
        written["updates"] = updates
        print(f"[DRY-RUN] Would write to {issue_key}: {updates}")

    with patch.object(m, "jira_set_fields", side_effect=fake_set):
        try:
            m.main()
        except SystemExit as e:
            print(f"Script exited with code {e.code}")
            return

    if written:
        print("\nDry-run complete — no Jira fields were actually modified.")
    else:
        print("\nDry-run complete — nothing to write (script exited early).")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--dryrun" in sys.argv:
        dryrun()
    else:
        # run unittest with verbose output
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
