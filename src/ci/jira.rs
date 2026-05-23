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

    // AI %: average with existing value if present
    let old_pct = existing.get(opts.field_ai_pct.as_str()).copied().flatten();
    let ai_pct_value = if let Some(old) = old_pct {
        let avg = (old + attribution.ai_pct) / 2.0;
        (avg * 10.0).round() / 10.0
    } else {
        attribution.ai_pct
    };

    // Lines: accumulate
    let added_value = existing
        .get(opts.field_lines_added.as_str())
        .copied()
        .flatten()
        .unwrap_or(0.0)
        + attribution.lines_added as f64;
    let deleted_value = existing
        .get(opts.field_lines_deleted.as_str())
        .copied()
        .flatten()
        .unwrap_or(0.0)
        + attribution.lines_deleted as f64;

    println!(
        "AI %: {:.1} (was {:?}), lines added: {:.0}, lines deleted: {:.0}",
        ai_pct_value, old_pct, added_value, deleted_value
    );

    put_fields(
        &opts.base_url,
        issue_key,
        &[
            (opts.field_ai_pct.clone(), ai_pct_value),
            (opts.field_lines_added.clone(), added_value),
            (opts.field_lines_deleted.clone(), deleted_value),
        ],
        &auth,
    )?;

    Ok(())
}
