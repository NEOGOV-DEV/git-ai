use crate::authorship::range_authorship::range_authorship;
use crate::git::repository::{CommitRange, find_repository_in_path};
use crate::http;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::json;

fn basic_auth_header(user: &str, token: &str) -> String {
    let encoded = base64::engine::general_purpose::STANDARD.encode(format!("{user}:{token}"));
    format!("Basic {encoded}")
}

fn extract_jira_id(pr_title: &str) -> Option<String> {
    let re = regex::Regex::new(r"^([A-Z]+-\d+)[\s\-|]").ok()?;
    let caps = re.captures(pr_title.trim())?;
    Some(caps[1].to_string())
}

#[derive(Debug, Deserialize)]
struct JiraIssueFields {
    fields: std::collections::HashMap<String, serde_json::Value>,
}

fn jira_get_fields(
    agent: &ureq::Agent,
    base_url: &str,
    issue_key: &str,
    field_ids: &[&str],
    auth: &str,
) -> Result<std::collections::HashMap<String, Option<f64>>, String> {
    let fields_param = field_ids
        .iter()
        .map(|id| format!("customfield_{id}"))
        .collect::<Vec<_>>()
        .join(",");
    let url = format!("{base_url}/rest/api/3/issue/{issue_key}?fields={fields_param}");
    println!("Fetching Jira {url} for {issue_key}");

    let response = http::send(
        agent
            .get(&url)
            .set("Authorization", auth)
            .set("Accept", "application/json"),
    )
    .map_err(|e| format!("Jira GET transport error: {e}"))?;

    if response.status_code != 200 {
        return Err(format!(
            "Jira GET failed ({}): {}",
            response.status_code,
            response.as_str().unwrap_or("(unreadable)")
        ));
    }

    let body: JiraIssueFields = serde_json::from_str(
        response
            .as_str()
            .map_err(|e| format!("UTF-8 error: {e}"))?,
    )
    .map_err(|e| format!("JSON parse error: {e}"))?;

    let mut result = std::collections::HashMap::new();
    for id in field_ids {
        let key = format!("customfield_{id}");
        let val = body.fields.get(&key).and_then(|v| v.as_f64());
        result.insert(id.to_string(), val);
    }
    Ok(result)
}

fn jira_set_fields(
    agent: &ureq::Agent,
    base_url: &str,
    issue_key: &str,
    updates: &std::collections::HashMap<String, f64>,
    auth: &str,
) -> Result<(), String> {
    let url = format!("{base_url}/rest/api/3/issue/{issue_key}");
    let fields: serde_json::Map<String, serde_json::Value> = updates
        .iter()
        .map(|(id, val)| (format!("customfield_{id}"), json!(val)))
        .collect();
    let payload = json!({ "fields": fields }).to_string();

    let response = http::send_with_body(
        agent
            .put(&url)
            .set("Authorization", auth)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json"),
        &payload,
    )
    .map_err(|e| format!("Jira PUT transport error: {e}"))?;

    if response.status_code / 100 != 2 {
        return Err(format!(
            "Jira PUT failed ({}): {}",
            response.status_code,
            response.as_str().unwrap_or("(unreadable)")
        ));
    }
    Ok(())
}

pub fn run_jira(args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        let mut i = 0usize;
        while i < args.len() {
            if args[i] == name {
                if i + 1 < args.len() {
                    return Some(args[i + 1].clone());
                } else {
                    eprintln!("Missing value for flag {name}");
                    std::process::exit(1);
                }
            }
            i += 1;
        }
        None
    };

    let require = |name: &str| -> String {
        match flag(name) {
            Some(v) => v,
            None => {
                eprintln!("{name} is required");
                print_jira_help_and_exit();
            }
        }
    };

    let pr_title = require("--pr-title");
    let base_sha = require("--base-sha");
    let head_sha = require("--head-sha");
    let jira_base_url = require("--jira-base-url").trim_end_matches('/').to_string();
    let field_ai_pct = require("--field-ai-pct");
    let field_lines_added = require("--field-lines-added");
    let field_lines_deleted = require("--field-lines-deleted");

    let jira_user = match std::env::var("JIRA_USER") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            eprintln!("JIRA_USER environment variable is required");
            std::process::exit(1);
        }
    };
    let jira_token = match std::env::var("JIRA_TOKEN") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            eprintln!("JIRA_TOKEN environment variable is required");
            std::process::exit(1);
        }
    };

    let jira_key = match extract_jira_id(&pr_title) {
        Some(k) => k,
        None => {
            println!("No Jira issue key found in PR title: '{pr_title}'. Skipping.");
            std::process::exit(0);
        }
    };
    println!("Jira issue: {jira_key}");

    // Compute range stats
    let repo = match find_repository_in_path(".") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open repository: {e}");
            std::process::exit(1);
        }
    };

    let commit_range = match CommitRange::new_infer_refname(
        &repo,
        base_sha.clone(),
        head_sha.clone(),
        None,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to build commit range {base_sha}..{head_sha}: {e}");
            std::process::exit(1);
        }
    };

    println!("Running git-ai stats {base_sha}..{head_sha}");
    let range_stats = match range_authorship(commit_range, false, &[], None) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to compute range stats: {e}");
            std::process::exit(1);
        }
    };

    let rs = &range_stats.range_stats;
    let diff_adds = rs.git_diff_added_lines;
    let diff_dels = rs.git_diff_deleted_lines;
    let total_ai = rs.ai_additions;

    if diff_adds == 0 {
        println!("No git-ai attribution found for this PR. Skipping.");
        std::process::exit(0);
    }

    let new_pct = (total_ai as f64 / diff_adds as f64 * 100.0 * 10.0).round() / 10.0;
    println!("PR AI attribution: {total_ai} AI lines / {diff_adds} added → {new_pct}%");
    println!("PR lines: +{diff_adds} added, -{diff_dels} deleted");

    let auth = basic_auth_header(&jira_user, &jira_token);
    let agent = http::build_agent(Some(30));

    let field_ids = [
        field_ai_pct.as_str(),
        field_lines_added.as_str(),
        field_lines_deleted.as_str(),
    ];
    let existing = match jira_get_fields(&agent, &jira_base_url, &jira_key, &field_ids, &auth) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to fetch Jira fields: {e}");
            std::process::exit(1);
        }
    };

    let old_pct = existing.get(&field_ai_pct).copied().flatten();
    let ai_pct_value = if let Some(old) = old_pct {
        let avg = ((old + new_pct) / 2.0 * 10.0).round() / 10.0;
        println!("AI %: existing={old}  new={new_pct}  →  averaged={avg}");
        avg
    } else {
        println!("AI %: not previously set  →  {new_pct}");
        new_pct
    };

    let added_value =
        existing.get(&field_lines_added).copied().flatten().unwrap_or(0.0) + diff_adds as f64;
    let deleted_value =
        existing.get(&field_lines_deleted).copied().flatten().unwrap_or(0.0) + diff_dels as f64;

    let old_added = existing
        .get(&field_lines_added)
        .copied()
        .flatten()
        .unwrap_or(0.0);
    let old_deleted = existing
        .get(&field_lines_deleted)
        .copied()
        .flatten()
        .unwrap_or(0.0);
    println!("Lines added:   existing={old_added}  +{diff_adds}  →  {added_value}");
    println!("Lines deleted: existing={old_deleted}  +{diff_dels}  →  {deleted_value}");

    let mut updates = std::collections::HashMap::new();
    updates.insert(field_ai_pct.clone(), ai_pct_value);
    updates.insert(field_lines_added.clone(), added_value);
    updates.insert(field_lines_deleted.clone(), deleted_value);

    if let Err(e) = jira_set_fields(&agent, &jira_base_url, &jira_key, &updates, &auth) {
        eprintln!("Failed to update Jira fields: {e}");
        std::process::exit(1);
    }
    println!(
        "Updated {jira_key}: AI%={ai_pct_value}, lines_added={added_value}, lines_deleted={deleted_value}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── basic_auth_header ────────────────────────────────────────────────────

    #[test]
    fn test_basic_auth_header_encodes_correctly() {
        let header = basic_auth_header("user@example.com", "mytoken");
        let expected = base64::engine::general_purpose::STANDARD
            .encode("user@example.com:mytoken");
        assert_eq!(header, format!("Basic {expected}"));
    }

    #[test]
    fn test_basic_auth_header_empty_credentials() {
        let header = basic_auth_header("", "");
        let expected = base64::engine::general_purpose::STANDARD.encode(":");
        assert_eq!(header, format!("Basic {expected}"));
    }

    // ── extract_jira_id ──────────────────────────────────────────────────────

    #[test]
    fn test_extract_jira_id_space_separator() {
        assert_eq!(
            extract_jira_id("ABC-123 my change"),
            Some("ABC-123".to_string())
        );
    }

    #[test]
    fn test_extract_jira_id_dash_separator() {
        assert_eq!(
            extract_jira_id("ABC-123-my-change"),
            Some("ABC-123".to_string())
        );
    }

    #[test]
    fn test_extract_jira_id_pipe_separator() {
        assert_eq!(
            extract_jira_id("ABC-123| some title"),
            Some("ABC-123".to_string())
        );
    }

    #[test]
    fn test_extract_jira_id_leading_whitespace() {
        assert_eq!(
            extract_jira_id("  PROJ-42 refactor auth"),
            Some("PROJ-42".to_string())
        );
    }

    #[test]
    fn test_extract_jira_id_multi_letter_project() {
        assert_eq!(
            extract_jira_id("MYPROJECT-9999 large number"),
            Some("MYPROJECT-9999".to_string())
        );
    }

    #[test]
    fn test_extract_jira_id_no_separator_returns_none() {
        // Key present but not followed by [\s\-|]
        assert_eq!(extract_jira_id("ABC-123"), None);
    }

    #[test]
    fn test_extract_jira_id_colon_separator_returns_none() {
        // Colon is not in the character class
        assert_eq!(extract_jira_id("ABC-123: some title"), None);
    }

    #[test]
    fn test_extract_jira_id_lowercase_returns_none() {
        assert_eq!(extract_jira_id("abc-123 some title"), None);
    }

    #[test]
    fn test_extract_jira_id_not_at_start_returns_none() {
        assert_eq!(extract_jira_id("feat: ABC-123 something"), None);
    }

    #[test]
    fn test_extract_jira_id_no_jira_key_returns_none() {
        assert_eq!(extract_jira_id("just a plain PR title"), None);
    }

    #[test]
    fn test_extract_jira_id_empty_string_returns_none() {
        assert_eq!(extract_jira_id(""), None);
    }

    // ── JiraIssueFields deserialization ──────────────────────────────────────

    #[test]
    fn test_jira_issue_fields_deserializes_numeric_fields() {
        let json = r#"{"fields":{"customfield_10001":42.5,"customfield_10002":100.0}}"#;
        let parsed: JiraIssueFields = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.fields.get("customfield_10001").and_then(|v| v.as_f64()),
            Some(42.5)
        );
        assert_eq!(
            parsed.fields.get("customfield_10002").and_then(|v| v.as_f64()),
            Some(100.0)
        );
    }

    #[test]
    fn test_jira_issue_fields_handles_null_field() {
        let json = r#"{"fields":{"customfield_10001":null}}"#;
        let parsed: JiraIssueFields = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.fields.get("customfield_10001").and_then(|v| v.as_f64()),
            None
        );
    }

    #[test]
    fn test_jira_issue_fields_missing_key_returns_none() {
        let json = r#"{"fields":{}}"#;
        let parsed: JiraIssueFields = serde_json::from_str(json).unwrap();
        assert!(parsed.fields.get("customfield_99999").is_none());
    }

    // ── jira_get_fields (HTTP) ───────────────────────────────────────────────

    fn plain_agent() -> ureq::Agent {
        ureq::AgentBuilder::new().build()
    }

    #[test]
    fn test_jira_get_fields_success() {
        let mut server = mockito::Server::new();
        let body = serde_json::json!({
            "fields": {
                "customfield_10001": 75.0,
                "customfield_10002": 200.0,
                "customfield_10003": null
            }
        })
        .to_string();
        let _mock = server
            .mock("GET", mockito::Matcher::Regex(r"^/rest/api/3/issue/".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&body)
            .create();

        let agent = plain_agent();
        let result = jira_get_fields(
            &agent,
            &server.url(),
            "TEST-1",
            &["10001", "10002", "10003"],
            "Basic dGVzdA==",
        );
        let fields = result.unwrap();
        assert_eq!(fields.get("10001").copied().flatten(), Some(75.0));
        assert_eq!(fields.get("10002").copied().flatten(), Some(200.0));
        assert_eq!(fields.get("10003").copied().flatten(), None);
    }

    #[test]
    fn test_jira_get_fields_non_200_returns_error() {
        let mut server = mockito::Server::new();
        let _mock = server
            .mock("GET", mockito::Matcher::Regex(r"^/rest/api/3/issue/".to_string()))
            .with_status(404)
            .with_body("Not Found")
            .create();

        let agent = plain_agent();
        let result = jira_get_fields(
            &agent,
            &server.url(),
            "TEST-1",
            &["10001"],
            "Basic dGVzdA==",
        );
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("404"), "expected 404 in: {msg}");
    }

    #[test]
    fn test_jira_get_fields_missing_field_id_mapped_to_none() {
        let mut server = mockito::Server::new();
        // Response only has customfield_10001; customfield_10002 is absent
        let body = serde_json::json!({
            "fields": { "customfield_10001": 50.0 }
        })
        .to_string();
        let _mock = server
            .mock("GET", mockito::Matcher::Regex(r"^/rest/api/3/issue/".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&body)
            .create();

        let agent = plain_agent();
        let result = jira_get_fields(
            &agent,
            &server.url(),
            "TEST-1",
            &["10001", "10002"],
            "Basic dGVzdA==",
        )
        .unwrap();
        assert_eq!(result.get("10001").copied().flatten(), Some(50.0));
        assert_eq!(result.get("10002").copied().flatten(), None);
    }

    // ── jira_set_fields (HTTP) ───────────────────────────────────────────────

    #[test]
    fn test_jira_set_fields_success_on_204() {
        let mut server = mockito::Server::new();
        let _mock = server
            .mock("PUT", mockito::Matcher::Regex(r"^/rest/api/3/issue/".to_string()))
            .with_status(204)
            .create();

        let agent = plain_agent();
        let mut updates = std::collections::HashMap::new();
        updates.insert("10001".to_string(), 88.0_f64);

        let result = jira_set_fields(
            &agent,
            &server.url(),
            "TEST-1",
            &updates,
            "Basic dGVzdA==",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_jira_set_fields_success_on_200() {
        let mut server = mockito::Server::new();
        let _mock = server
            .mock("PUT", mockito::Matcher::Regex(r"^/rest/api/3/issue/".to_string()))
            .with_status(200)
            .create();

        let agent = plain_agent();
        let mut updates = std::collections::HashMap::new();
        updates.insert("10001".to_string(), 50.0_f64);

        let result = jira_set_fields(
            &agent,
            &server.url(),
            "TEST-1",
            &updates,
            "Basic dGVzdA==",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_jira_set_fields_non_2xx_returns_error() {
        let mut server = mockito::Server::new();
        let _mock = server
            .mock("PUT", mockito::Matcher::Regex(r"^/rest/api/3/issue/".to_string()))
            .with_status(403)
            .with_body("Forbidden")
            .create();

        let agent = plain_agent();
        let mut updates = std::collections::HashMap::new();
        updates.insert("10001".to_string(), 10.0_f64);

        let result = jira_set_fields(
            &agent,
            &server.url(),
            "TEST-1",
            &updates,
            "Basic dGVzdA==",
        );
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("403"), "expected 403 in: {msg}");
    }

    #[test]
    fn test_jira_set_fields_sends_customfield_keys() {
        let mut server = mockito::Server::new();
        let _mock = server
            .mock("PUT", mockito::Matcher::Regex(r"^/rest/api/3/issue/".to_string()))
            .with_status(204)
            .match_body(mockito::Matcher::PartialJsonString(
                r#"{"fields":{"customfield_10001":42.0}}"#.to_string(),
            ))
            .create();

        let agent = plain_agent();
        let mut updates = std::collections::HashMap::new();
        updates.insert("10001".to_string(), 42.0_f64);

        let result = jira_set_fields(
            &agent,
            &server.url(),
            "TEST-1",
            &updates,
            "Basic dGVzdA==",
        );
        assert!(result.is_ok());
    }
}

pub fn print_jira_help_and_exit() -> ! {
    eprintln!("git-ai ci jira - Send PR AI attribution stats to a Jira ticket");
    eprintln!();
    eprintln!("Usage: git-ai ci jira --pr-title <title> --base-sha <sha> --head-sha <sha>");
    eprintln!("                       --jira-base-url <url>");
    eprintln!("                       --field-ai-pct <id> --field-lines-added <id> --field-lines-deleted <id>");
    eprintln!();
    eprintln!("Environment variables:");
    eprintln!("  JIRA_USER    Jira user email");
    eprintln!("  JIRA_TOKEN   Jira API token");
    eprintln!();
    eprintln!("The Jira issue key is extracted from the PR title (e.g. 'ABC-123: my change').");
    eprintln!("AI % is averaged with the existing value; lines added/deleted are accumulated.");
    std::process::exit(1);
}
