#!/usr/bin/env python3
"""
jira_ai_pct.py — read AI attribution from git notes on a merged PR's commits
and write the averaged ai_total% to a custom Jira field.

Usage (called by GitHub Actions):
    python3 .github/scripts/jira_ai_pct.py

Required environment variables:
    PR_TITLE               — pull request title (Jira ID extracted from here)
    PR_BASE_SHA            — merge base commit SHA
    PR_HEAD_SHA            — PR head commit SHA
    JIRA_BASE_URL          — e.g. https://yourcompany.atlassian.net
    JIRA_USER              — Jira user email
    JIRA_TOKEN             — Jira API token
"""
import base64
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.request


# ── git-ai helpers ────────────────────────────────────────────────────────────

def git_ai_stats(commit_range: str) -> dict:
    result = subprocess.run(
        ["git-ai", "stats", commit_range, "--json"],
        capture_output=True, text=True,
    )
    try:
        print(result.stdout)
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stderr)
        return {}


# ── Jira helpers ──────────────────────────────────────────────────────────────

def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    for ca in ("/etc/ssl/cert.pem", "/usr/local/etc/openssl/cert.pem",
               "/opt/homebrew/etc/openssl@3/cert.pem"):
        if os.path.isfile(ca):
            ctx.load_verify_locations(ca)
            return ctx
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


_SSL = _ssl_context()


def _jira_headers(user: str, token: str) -> dict:
    creds = base64.b64encode(f"{user}:{token}".encode()).decode()
    return {"Authorization": f"Basic {creds}", "Accept": "application/json"}


def jira_get_fields(base_url: str, issue_key: str, field_ids: list[str],
                    user: str, token: str) -> dict[str, float | None]:
    """Fetch multiple custom fields in one API call. Returns {field_id: value}."""
    fields_param = ",".join(f"customfield_{fid}" for fid in field_ids)
    url = f"{base_url}/rest/api/3/issue/{issue_key}?fields={fields_param}"
    print(f"Fetching Jira {url} for {issue_key}: {field_ids}")
    req = urllib.request.Request(url, headers=_jira_headers(user, token))
    try:
        with urllib.request.urlopen(req, context=_SSL) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"Jira GET failed ({e.code}): {e.read().decode()}", file=sys.stderr)
        return {fid: None for fid in field_ids}
    fields = data.get("fields", {})
    result = {}
    for fid in field_ids:
        val = fields.get(f"customfield_{fid}")
        try:
            result[fid] = float(val) if val is not None else None
        except (TypeError, ValueError):
            result[fid] = None
    return result


def jira_set_fields(base_url: str, issue_key: str, updates: dict[str, float],
                    user: str, token: str) -> None:
    """Write multiple custom fields in one API call. updates is {field_id: value}."""
    url = f"{base_url}/rest/api/3/issue/{issue_key}"
    payload = json.dumps({"fields": {f"customfield_{fid}": v for fid, v in updates.items()}}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={**_jira_headers(user, token), "Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, context=_SSL) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        print(f"Jira PUT failed ({e.code}): {e.read().decode()}", file=sys.stderr)
        raise


# ── Jira ID extraction ────────────────────────────────────────────────────────

def extract_jira_id(pr_title: str) -> str | None:
    """Match 'NN-123 title', 'NN-123: title', 'NN-123 - title' at start of title."""
    m = re.match(r"^([A-Z]+-\d+)[\s:\-]", pr_title.strip())
    return m.group(1) if m else None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pr_title      = os.environ["PR_TITLE"]
    base_sha      = os.environ["PR_BASE_SHA"]
    head_sha      = os.environ["PR_HEAD_SHA"]
    jira_url      = os.environ.get("JIRA_BASE_URL", "https://neogov.jira.com").rstrip("/")
    jira_user     = os.environ["JIRA_USER"]
    jira_token    = os.environ["JIRA_TOKEN"]
    field_ai_pct  = "26954"
    field_added   = "26955"
    field_deleted = "26956"

    jira_key = extract_jira_id(pr_title)
    if not jira_key:
        print(f"No Jira issue key found in PR title: '{pr_title}'. Skipping.")
        sys.exit(0)
    print(f"Jira issue: {jira_key}")

    commit_range = f"{base_sha}..{head_sha}"
    print(f"Running git-ai stats {commit_range}")
    d = git_ai_stats(commit_range)
    
    rs         = d.get("range_stats", {})
    total_ai   = rs.get("ai_additions", 0)
    diff_adds  = rs.get("git_diff_added_lines", 0)
    diff_dels  = rs.get("git_diff_deleted_lines", 0)

    if diff_adds == 0:
        print("No git-ai attribution found for this PR. Skipping.")
        sys.exit(0)

    new_pct = round(total_ai / diff_adds * 100, 1)
    print(f"PR AI attribution: {total_ai} AI lines / {diff_adds} added → {new_pct}%")
    print(f"PR lines: +{diff_adds} added, -{diff_dels} deleted")

    # fetch all three existing field values in one API call
    existing = jira_get_fields(jira_url, jira_key,
                               [field_ai_pct, field_added, field_deleted],
                               jira_user, jira_token)

    # AI %: average old and new
    old_pct = existing[field_ai_pct]
    if old_pct is not None:
        ai_pct_value = round((old_pct + new_pct) / 2, 1)
        print(f"AI %: existing={old_pct}  new={new_pct}  →  averaged={ai_pct_value}")
    else:
        ai_pct_value = new_pct
        print(f"AI %: not previously set  →  {ai_pct_value}")

    # lines added/deleted: accumulate
    added_value   = (existing[field_added]   or 0) + diff_adds
    deleted_value = (existing[field_deleted] or 0) + diff_dels
    print(f"Lines added:   existing={existing[field_added] or 0}  +{diff_adds}  →  {added_value}")
    print(f"Lines deleted: existing={existing[field_deleted] or 0}  +{diff_dels}  →  {deleted_value}")

    jira_set_fields(jira_url, jira_key, {
        field_ai_pct:  ai_pct_value,
        field_added:   added_value,
        field_deleted: deleted_value,
    }, jira_user, jira_token)
    print(f"Updated {jira_key}: AI%={ai_pct_value}, lines_added={added_value}, lines_deleted={deleted_value}")


if __name__ == "__main__":
    main()
