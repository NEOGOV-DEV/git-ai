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
