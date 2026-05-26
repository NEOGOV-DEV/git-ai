use crate::authorship::range_authorship::{range_authorship, RangeAuthorshipStats};
use crate::authorship::ignore::effective_ignore_patterns;
use crate::error::GitAiError;
use crate::git::repository::{CommitRange, Repository};
use serde::Deserialize;
use std::collections::HashMap;

pub struct JiraAttribution {
    pub ai_pct: f64,
    pub lines_added: u32,
    pub lines_deleted: u32,
}

pub struct JiraOptions {
    pub base_url: String,
    pub field_ai_pct: String,
    pub field_lines_added: String,
    pub field_lines_deleted: String,
}

pub fn extract_jira_id(pr_title: &str) -> Option<String> {
    let title = pr_title.trim();
    let end = title.find(|c: char| !c.is_ascii_uppercase() && !c.is_ascii_digit() && c != '-')?;
    let candidate = &title[..end];
    // Must match LETTERS-DIGITS
    let dash = candidate.rfind('-')?;
    let prefix = &candidate[..dash];
    let suffix = &candidate[dash + 1..];
    if prefix.chars().all(|c| c.is_ascii_uppercase())
        && !prefix.is_empty()
        && suffix.chars().all(|c| c.is_ascii_digit())
        && !suffix.is_empty()
    {
        let rest = &title[end..];
        if rest.starts_with([' ', ':', '-']) {
            return Some(candidate.to_string());
        }
    }
    None
}

pub fn compute_attribution(
    repo: &Repository,
    base_sha: &str,
    head_sha: &str,
) -> Result<Option<JiraAttribution>, GitAiError> {
    let range = CommitRange::new_infer_refname(
        repo,
        base_sha.to_string(),
        head_sha.to_string(),
        None,
    )?;

    let ignore = effective_ignore_patterns(repo, &[], &[]);
    let stats: RangeAuthorshipStats = range_authorship(range, false, &ignore, None)?;

    let rs = &stats.range_stats;
    let added = rs.git_diff_added_lines;
    let deleted = rs.git_diff_deleted_lines;

    if added == 0 {
        return Ok(None);
    }

    let ai_pct = (rs.ai_additions as f64 / added as f64 * 100.0 * 10.0).round() / 10.0;
    Ok(Some(JiraAttribution { ai_pct, lines_added: added, lines_deleted: deleted }))
}

#[derive(Debug, Deserialize)]
struct JiraIssueResponse {
    fields: HashMap<String, serde_json::Value>,
}

fn jira_auth_header(user: &str, token: &str) -> String {
    let raw = format!("{}:{}", user, token);
    let encoded = base64_encode(raw.as_bytes());
    format!("Basic {}", encoded)
}

fn base64_encode(input: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((input.len() + 2) / 3 * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let combined = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[((combined >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((combined >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(CHARS[((combined >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(CHARS[(combined & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn get_existing_field(
    base_url: &str,
    issue_key: &str,
    field_ids: &[&str],
    auth: &str,
) -> Result<HashMap<String, Option<f64>>, GitAiError> {
    let fields_param: String = field_ids
        .iter()
        .map(|id| format!("customfield_{}", id))
        .collect::<Vec<_>>()
        .join(",");
    let url = format!(
        "{}/rest/api/3/issue/{}?fields={}",
        base_url.trim_end_matches('/'),
        issue_key,
        fields_param
    );

    let response = minreq::get(&url)
        .with_header("Authorization", auth)
        .with_header("Accept", "application/json")
        .with_timeout(30)
        .send()
        .map_err(|e| GitAiError::Generic(format!("Jira GET request failed: {}", e)))?;

    if response.status_code < 200 || response.status_code >= 300 {
        return Err(GitAiError::Generic(format!(
            "Jira GET failed ({}): {}",
            response.status_code,
            response.as_str().unwrap_or("(no body)")
        )));
    }

    let body: JiraIssueResponse = serde_json::from_slice(response.as_bytes())
        .map_err(|e| GitAiError::Generic(format!("Failed to parse Jira response: {}", e)))?;

    let mut result = HashMap::new();
    for id in field_ids {
        let key = format!("customfield_{}", id);
        let value = body
            .fields
            .get(&key)
            .and_then(|v| v.as_f64());
        result.insert(id.to_string(), value);
    }
    Ok(result)
}

fn put_fields(
    base_url: &str,
    issue_key: &str,
    fields: &[(String, f64)],
    auth: &str,
) -> Result<(), GitAiError> {
    let url = format!(
        "{}/rest/api/3/issue/{}",
        base_url.trim_end_matches('/'),
        issue_key
    );

    let mut field_map = serde_json::Map::new();
    for (id, value) in fields {
        let key = format!("customfield_{}", id);
        field_map.insert(key, serde_json::json!(value));
    }
    let payload = serde_json::json!({ "fields": field_map });
    let body = serde_json::to_vec(&payload)
        .map_err(|e| GitAiError::Generic(format!("Failed to serialize Jira payload: {}", e)))?;

    let response = minreq::put(&url)
        .with_header("Authorization", auth)
        .with_header("Content-Type", "application/json")
        .with_header("Accept", "application/json")
        .with_body(body)
        .with_timeout(30)
        .send()
        .map_err(|e| GitAiError::Generic(format!("Jira PUT request failed: {}", e)))?;

    if response.status_code < 200 || response.status_code >= 300 {
        return Err(GitAiError::Generic(format!(
            "Jira PUT failed ({}): {}",
            response.status_code,
            response.as_str().unwrap_or("(no body)")
        )));
    }
    Ok(())
}

/// Merged field values ready to write back to Jira.
pub struct MergedFields {
    pub ai_pct: f64,
    pub lines_added: f64,
    pub lines_deleted: f64,
}

/// Pure merge logic: average AI %, accumulate line counts.
pub fn merge_fields(
    existing_ai_pct: Option<f64>,
    existing_added: Option<f64>,
    existing_deleted: Option<f64>,
    attribution: &JiraAttribution,
) -> MergedFields {
    let ai_pct = if let Some(old) = existing_ai_pct {
        ((old + attribution.ai_pct) / 2.0 * 10.0).round() / 10.0
    } else {
        attribution.ai_pct
    };
    MergedFields {
        ai_pct,
        lines_added: existing_added.unwrap_or(0.0) + attribution.lines_added as f64,
        lines_deleted: existing_deleted.unwrap_or(0.0) + attribution.lines_deleted as f64,
    }
}

pub fn send_to_jira(
    opts: &JiraOptions,
    issue_key: &str,
    attribution: &JiraAttribution,
    jira_user: &str,
    jira_token: &str,
) -> Result<(), GitAiError> {
    let auth = jira_auth_header(jira_user, jira_token);

    let field_ids = [
        opts.field_ai_pct.as_str(),
        opts.field_lines_added.as_str(),
        opts.field_lines_deleted.as_str(),
    ];

    let existing = get_existing_field(&opts.base_url, issue_key, &field_ids, &auth)?;

    let merged = merge_fields(
        existing.get(opts.field_ai_pct.as_str()).copied().flatten(),
        existing.get(opts.field_lines_added.as_str()).copied().flatten(),
        existing.get(opts.field_lines_deleted.as_str()).copied().flatten(),
        attribution,
    );

    println!(
        "AI %: {:.1}, lines added: {:.0}, lines deleted: {:.0}",
        merged.ai_pct, merged.lines_added, merged.lines_deleted
    );

    put_fields(
        &opts.base_url,
        issue_key,
        &[
            (opts.field_ai_pct.clone(), merged.ai_pct),
            (opts.field_lines_added.clone(), merged.lines_added),
            (opts.field_lines_deleted.clone(), merged.lines_deleted),
        ],
        &auth,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_jira_id ───────────────────────────────────────────────────────

    #[test]
    fn jira_id_space_separator() {
        assert_eq!(extract_jira_id("ABC-123 add feature"), Some("ABC-123".into()));
    }

    #[test]
    fn jira_id_colon_separator() {
        assert_eq!(extract_jira_id("ABC-123: add feature"), Some("ABC-123".into()));
    }

    #[test]
    fn jira_id_dash_separator() {
        assert_eq!(extract_jira_id("ABC-123 - add feature"), Some("ABC-123".into()));
    }

    #[test]
    fn jira_id_leading_whitespace() {
        assert_eq!(extract_jira_id("  DM-456: fix bug"), Some("DM-456".into()));
    }

    #[test]
    fn jira_id_multi_letter_prefix() {
        assert_eq!(extract_jira_id("NEOGOV-789 update config"), Some("NEOGOV-789".into()));
    }

    #[test]
    fn jira_id_not_at_start() {
        assert_eq!(extract_jira_id("Fix ABC-123 bug"), None);
    }

    #[test]
    fn jira_id_no_separator_after_key() {
        // "ABC-123title" has no separator so should not match
        assert_eq!(extract_jira_id("ABC-123title"), None);
    }

    #[test]
    fn jira_id_lowercase_prefix() {
        assert_eq!(extract_jira_id("abc-123: fix"), None);
    }

    #[test]
    fn jira_id_empty() {
        assert_eq!(extract_jira_id(""), None);
    }

    #[test]
    fn jira_id_no_digits() {
        assert_eq!(extract_jira_id("ABC- fix"), None);
    }

    #[test]
    fn jira_id_no_prefix() {
        assert_eq!(extract_jira_id("-123: fix"), None);
    }

    // ── base64_encode ─────────────────────────────────────────────────────────

    #[test]
    fn base64_empty() {
        assert_eq!(base64_encode(b""), "");
    }

    #[test]
    fn base64_one_byte() {
        assert_eq!(base64_encode(b"M"), "TQ==");
    }

    #[test]
    fn base64_two_bytes() {
        assert_eq!(base64_encode(b"Ma"), "TWE=");
    }

    #[test]
    fn base64_three_bytes_no_padding() {
        assert_eq!(base64_encode(b"Man"), "TWFu");
    }

    #[test]
    fn base64_credentials() {
        // standard test vector: "user:token"
        assert_eq!(base64_encode(b"user:token"), "dXNlcjp0b2tlbg==");
    }

    // ── merge_fields ──────────────────────────────────────────────────────────

    fn attr(ai_pct: f64, added: u32, deleted: u32) -> JiraAttribution {
        JiraAttribution { ai_pct, lines_added: added, lines_deleted: deleted }
    }

    #[test]
    fn merge_no_existing_values() {
        let m = merge_fields(None, None, None, &attr(80.0, 100, 20));
        assert_eq!(m.ai_pct, 80.0);
        assert_eq!(m.lines_added, 100.0);
        assert_eq!(m.lines_deleted, 20.0);
    }

    #[test]
    fn merge_averages_ai_pct() {
        // (60 + 80) / 2 = 70.0
        let m = merge_fields(Some(60.0), None, None, &attr(80.0, 0, 0));
        assert_eq!(m.ai_pct, 70.0);
    }

    #[test]
    fn merge_ai_pct_rounds_to_one_decimal() {
        // (33.3 + 66.6) in f64 = 99.899..., /2 = 49.949..., rounds to 49.9
        let m = merge_fields(Some(33.3), None, None, &attr(66.6, 0, 0));
        assert_eq!(m.ai_pct, 49.9);
    }

    #[test]
    fn merge_accumulates_lines_added() {
        let m = merge_fields(None, Some(500.0), None, &attr(0.0, 250, 0));
        assert_eq!(m.lines_added, 750.0);
    }

    #[test]
    fn merge_accumulates_lines_deleted() {
        let m = merge_fields(None, None, Some(100.0), &attr(0.0, 0, 50));
        assert_eq!(m.lines_deleted, 150.0);
    }

    #[test]
    fn merge_all_fields_existing() {
        let m = merge_fields(Some(40.0), Some(200.0), Some(50.0), &attr(60.0, 100, 30));
        assert_eq!(m.ai_pct, 50.0);
        assert_eq!(m.lines_added, 300.0);
        assert_eq!(m.lines_deleted, 80.0);
    }
}
