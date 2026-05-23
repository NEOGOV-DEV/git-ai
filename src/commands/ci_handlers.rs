use crate::ci::ci_context::{CiContext, CiEvent, CiRunOptions, CiRunResult};
use crate::ci::github::{
    get_github_ci_context, install_github_ci_workflow, install_github_push_metrics_workflow,
    run_github_push_metrics,
};
use crate::ci::gitlab::{get_gitlab_ci_context, print_gitlab_ci_yaml};
use crate::ci::jira::{JiraOptions, compute_attribution, extract_jira_id, send_to_jira};
use crate::git::repository::find_repository_in_path;
use crate::utils::debug_log;

/// Print a human-readable message for a CiRunResult
fn print_ci_result(result: &CiRunResult, prefix: &str) {
    match result {
        CiRunResult::AuthorshipRewritten { .. } => {
            println!("{}: authorship rewritten successfully", prefix);
        }
        CiRunResult::AlreadyExists { .. } => {
            println!("{}: authorship already exists", prefix);
        }
        CiRunResult::SkippedSimpleMerge => {
            println!("{}: skipped simple merge (authorship preserved)", prefix);
        }
        CiRunResult::SkippedFastForward => {
            println!("{}: skipped fast-forward merge", prefix);
        }
        CiRunResult::NoAuthorshipAvailable => {
            println!(
                "{}: no AI authorship to track (pre-git-ai commits or human-only code)",
                prefix
            );
        }
    }
}

pub fn handle_ci(args: &[String]) {
    if args.is_empty() {
        print_ci_help_and_exit();
    }

    match args[0].as_str() {
        "github" => {
            handle_ci_github(&args[1..]);
        }
        "gitlab" => {
            handle_ci_gitlab(&args[1..]);
        }
        "local" => {
            handle_ci_local(&args[1..]);
        }
        "jira" => {
            handle_ci_jira(&args[1..]);
        }
        _ => {
            eprintln!("Unknown ci subcommand: {}", args[0]);
            print_ci_help_and_exit();
        }
    }
}

fn handle_ci_github(args: &[String]) {
    if args.is_empty() {
        print_ci_github_help_and_exit();
    }
    // Subcommands: install | (default: run in CI context)
    match args[0].as_str() {
        "run" => {
            let no_cleanup = args[1..].iter().any(|a| a == "--no-cleanup");
            let ci_context = get_github_ci_context();
            match ci_context {
                Ok(Some(ci_context)) => {
                    debug_log(&format!("GitHub CI context: {:?}", ci_context));
                    match ci_context.run() {
                        Ok(result) => {
                            debug_log(&format!("GitHub CI result: {:?}", result));
                            print_ci_result(&result, "GitHub CI");
                        }
                        Err(e) => {
                            eprintln!("Error running GitHub CI context: {}", e);
                            std::process::exit(1);
                        }
                    }
                    if !no_cleanup {
                        if let Err(e) = ci_context.teardown() {
                            eprintln!("Error tearing down GitHub CI context: {}", e);
                            std::process::exit(1);
                        }
                        debug_log("GitHub CI context teared down");
                    } else {
                        debug_log("Skipping teardown (--no-cleanup)");
                    }
                    std::process::exit(0);
                }
                Err(e) => {
                    eprintln!("Failed to get GitHub CI context: {}", e);
                    std::process::exit(1);
                }
                Ok(None) => {
                    eprintln!("No GitHub CI context found");
                    std::process::exit(1);
                }
            }
        }
        "install" => match install_github_ci_workflow() {
            Ok(path) => {
                println!("Installed GitHub Actions workflow to {}", path.display());
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("Failed to install GitHub CI workflow: {}", e);
                std::process::exit(1);
            }
        },
        "metrics" => {
            let sub_args = &args[1..];
            if sub_args.first().is_some_and(|s| s == "install") {
                match install_github_push_metrics_workflow() {
                    Ok(path) => {
                        println!("Installed push-metrics workflow to {}", path.display());
                        std::process::exit(0);
                    }
                    Err(e) => {
                        eprintln!("Failed to install push-metrics workflow: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                match run_github_push_metrics(sub_args) {
                    Ok(count) => {
                        println!("Pushed metrics for {} commit(s).", count);
                        std::process::exit(0);
                    }
                    Err(e) => {
                        eprintln!("Push metrics failed: {}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
        other => {
            eprintln!("Unknown ci github subcommand: {}", other);
            print_ci_help_and_exit();
        }
    }
}

fn handle_ci_gitlab(args: &[String]) {
    if args.is_empty() {
        print_ci_gitlab_help_and_exit();
    }
    // Subcommands: install | run
    match args[0].as_str() {
        "run" => {
            let no_cleanup = args[1..].iter().any(|a| a == "--no-cleanup");
            let ci_context = get_gitlab_ci_context();
            match ci_context {
                Ok(Some(ci_context)) => {
                    debug_log(&format!("GitLab CI context: {:?}", ci_context));
                    match ci_context.run() {
                        Ok(result) => {
                            debug_log(&format!("GitLab CI result: {:?}", result));
                            print_ci_result(&result, "GitLab CI");
                        }
                        Err(e) => {
                            eprintln!("Error running GitLab CI context: {}", e);
                            std::process::exit(1);
                        }
                    }
                    if !no_cleanup {
                        if let Err(e) = ci_context.teardown() {
                            eprintln!("Error tearing down GitLab CI context: {}", e);
                            std::process::exit(1);
                        }
                        debug_log("GitLab CI context teared down");
                    } else {
                        debug_log("Skipping teardown (--no-cleanup)");
                    }
                    std::process::exit(0);
                }
                Err(e) => {
                    eprintln!("Failed to get GitLab CI context: {}", e);
                    std::process::exit(1);
                }
                Ok(None) => {
                    // No matching MR found - this is not an error, just nothing to do
                    std::process::exit(0);
                }
            }
        }
        "install" => {
            print_gitlab_ci_yaml();
            std::process::exit(0);
        }
        other => {
            eprintln!("Unknown ci gitlab subcommand: {}", other);
            print_ci_help_and_exit();
        }
    }
}

fn handle_ci_local(args: &[String]) {
    if args.is_empty() {
        print_ci_local_help_and_exit();
    }

    let event = args[0].as_str();
    let event_args: &[String] = &args[1..];
    let has_bool_flag = |name: &str| event_args.iter().any(|arg| arg == name);

    // Simple flag parser over remaining args: --key value
    let flag = |name: &str| -> Option<String> {
        let mut i = 0usize;
        while i < event_args.len() {
            if event_args[i] == name {
                if i + 1 < event_args.len() {
                    return Some(event_args[i + 1].clone());
                } else {
                    eprintln!("Missing value for flag {}", name);
                    std::process::exit(1);
                }
            }
            i += 1;
        }
        None
    };

    // Open current repo
    let repo = match find_repository_in_path(".") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open repository in current directory: {}", e);
            std::process::exit(1);
        }
    };

    match event {
        "merge" => {
            let skip_fetch_all = has_bool_flag("--skip-fetch");
            let skip_fetch_notes = skip_fetch_all || has_bool_flag("--skip-fetch-notes");
            let skip_fetch_base = skip_fetch_all || has_bool_flag("--skip-fetch-base");

            // Required inputs for merge
            let merge_commit_sha = match flag("--merge-commit-sha") {
                Some(v) => v,
                None => {
                    eprintln!("--merge-commit-sha is required");
                    std::process::exit(1);
                }
            };

            let base_ref = match flag("--base-ref") {
                Some(v) => v,
                None => {
                    eprintln!("--base-ref is required (e.g., main)");
                    std::process::exit(1);
                }
            };

            // All flags required for merge
            let head_ref = match flag("--head-ref") {
                Some(v) => v,
                None => {
                    eprintln!("--head-ref is required");
                    std::process::exit(1);
                }
            };

            let head_sha = match flag("--head-sha") {
                Some(v) => v,
                None => {
                    eprintln!("--head-sha is required");
                    std::process::exit(1);
                }
            };

            let base_sha = match flag("--base-sha") {
                Some(v) => v,
                None => {
                    eprintln!("--base-sha is required");
                    std::process::exit(1);
                }
            };

            let ctx = CiContext {
                repo,
                event: CiEvent::Merge {
                    merge_commit_sha,
                    head_ref,
                    head_sha,
                    base_ref,
                    base_sha,
                },
                // Not used for local runs; teardown not invoked
                temp_dir: std::path::PathBuf::from("."),
            };

            debug_log(&format!("Local CI context: {:?}", ctx));
            match ctx.run_with_options(CiRunOptions {
                skip_fetch_notes,
                skip_fetch_base,
            }) {
                Ok(result) => {
                    debug_log(&format!("Local CI result: {:?}", result));
                    print_ci_result(&result, "Local CI (merge)");
                }
                Err(e) => {
                    eprintln!("Error running local CI: {}", e);
                    std::process::exit(1);
                }
            }
            std::process::exit(0);
        }
        other => {
            eprintln!("Unknown local CI event: {}", other);
            print_ci_local_help_and_exit();
        }
    }
}

fn handle_ci_jira(args: &[String]) {
    // Simple flag parser
    let flag = |name: &str| -> Option<String> {
        let mut i = 0;
        while i < args.len() {
            if args[i] == name {
                if i + 1 < args.len() {
                    return Some(args[i + 1].clone());
                } else {
                    eprintln!("Missing value for flag {}", name);
                    std::process::exit(1);
                }
            }
            i += 1;
        }
        None
    };

    let require_flag = |name: &str| -> String {
        flag(name).unwrap_or_else(|| {
            eprintln!("{} is required", name);
            print_ci_jira_help_and_exit();
        })
    };

    let pr_title = require_flag("--pr-title");
    let base_sha = require_flag("--base-sha");
    let head_sha = require_flag("--head-sha");
    let jira_base_url = require_flag("--jira-base-url");
    let field_ai_pct = require_flag("--field-ai-pct");
    let field_lines_added = require_flag("--field-lines-added");
    let field_lines_deleted = require_flag("--field-lines-deleted");

    let jira_user = std::env::var("JIRA_USER").unwrap_or_else(|_| {
        eprintln!("JIRA_USER environment variable is required");
        std::process::exit(1);
    });
    let jira_token = std::env::var("JIRA_TOKEN").unwrap_or_else(|_| {
        eprintln!("JIRA_TOKEN environment variable is required");
        std::process::exit(1);
    });

    let jira_key = match extract_jira_id(&pr_title) {
        Some(key) => key,
        None => {
            println!("No Jira issue key found in PR title: '{}'. Skipping.", pr_title);
            std::process::exit(0);
        }
    };
    println!("Jira issue: {}", jira_key);

    let repo = match find_repository_in_path(".") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open repository: {}", e);
            std::process::exit(1);
        }
    };

    println!("Computing AI attribution for {}..{}", base_sha, head_sha);
    let attribution = match compute_attribution(&repo, &base_sha, &head_sha) {
        Ok(Some(a)) => a,
        Ok(None) => {
            println!("No lines added in this PR range. Skipping.");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Failed to compute attribution: {}", e);
            std::process::exit(1);
        }
    };

    println!(
        "PR stats: AI={:.1}%, lines added={}, lines deleted={}",
        attribution.ai_pct, attribution.lines_added, attribution.lines_deleted
    );

    let opts = JiraOptions {
        base_url: jira_base_url,
        field_ai_pct,
        field_lines_added,
        field_lines_deleted,
    };

    match send_to_jira(&opts, &jira_key, &attribution, &jira_user, &jira_token) {
        Ok(()) => {
            println!("Updated Jira issue {} successfully.", jira_key);
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Failed to update Jira: {}", e);
            std::process::exit(1);
        }
    }
}

fn print_ci_jira_help_and_exit() -> ! {
    eprintln!("git-ai ci jira - Send AI attribution stats to a Jira issue on PR closure");
    eprintln!();
    eprintln!("Usage: git-ai ci jira [flags]");
    eprintln!();
    eprintln!("Required flags:");
    eprintln!("  --pr-title <title>          PR title (Jira ID extracted from here)");
    eprintln!("  --base-sha <sha>            Merge base commit SHA");
    eprintln!("  --head-sha <sha>            PR head / merge commit SHA");
    eprintln!("  --jira-base-url <url>       Jira instance base URL");
    eprintln!("  --field-ai-pct <id>         Custom field ID for AI percentage");
    eprintln!("  --field-lines-added <id>    Custom field ID for total lines added");
    eprintln!("  --field-lines-deleted <id>  Custom field ID for total lines deleted");
    eprintln!();
    eprintln!("Required environment variables:");
    eprintln!("  JIRA_USER    Jira user email");
    eprintln!("  JIRA_TOKEN   Jira API token");
    std::process::exit(1);
}

fn print_ci_help_and_exit() -> ! {
    eprintln!("git-ai ci - Continuous integration utilities");
    eprintln!();
    eprintln!("Usage: git-ai ci <subcommand> [args...]");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  github           GitHub CI");
    eprintln!("    run [--no-cleanup]  Run GitHub CI in current repo");
    eprintln!("    install        Install/update workflow in current repo");
    eprintln!("  gitlab           GitLab CI");
    eprintln!("    run [--no-cleanup]  Run GitLab CI in current repo");
    eprintln!("    install        Print YAML snippet to add to .gitlab-ci.yml");
    eprintln!("  local            Run CI locally by event name and flags");
    eprintln!("                   Usage: git-ai ci local <event> [flags]");
    eprintln!("                   Events:");
    eprintln!(
        "                     merge  --merge-commit-sha <sha> --base-ref <ref> --head-ref <ref> --head-sha <sha> --base-sha <sha>"
    );
    eprintln!(
        "                            [--skip-fetch-notes] [--skip-fetch-base] [--skip-fetch]"
    );
    eprintln!("  jira             Send AI attribution stats to Jira on PR closure");
    eprintln!("                   Usage: git-ai ci jira --pr-title <title> --base-sha <sha> --head-sha <sha>");
    eprintln!("                          --jira-base-url <url> --field-ai-pct <id>");
    eprintln!("                          --field-lines-added <id> --field-lines-deleted <id>");
    eprintln!("                   Env:   JIRA_USER, JIRA_TOKEN");
    std::process::exit(1);
}

fn print_ci_local_help_and_exit() -> ! {
    eprintln!("git-ai ci local - Run CI locally by event name and flags");
    eprintln!();
    eprintln!("Usage: git-ai ci local <event> [flags]");
    eprintln!();
    eprintln!("Events:");
    eprintln!(
        "  merge  --merge-commit-sha <sha> --base-ref <ref> --head-ref <ref> --head-sha <sha> --base-sha <sha>"
    );
    eprintln!("         [--skip-fetch-notes] [--skip-fetch-base] [--skip-fetch]");
    std::process::exit(1);
}

fn print_ci_github_help_and_exit() -> ! {
    eprintln!("git-ai ci github - GitHub CI utilities");
    eprintln!();
    eprintln!("Usage: git-ai ci github <subcommand> [args...]");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  run [--no-cleanup]   Run GitHub CI in current repo");
    eprintln!("                       --no-cleanup  Skip teardown after run");
    eprintln!("  install              Install/update PR-authorship workflow in current repo");
    eprintln!("  metrics              Push AI stats for commits in a push to OTel");
    eprintln!(
        "                       --before <sha>        SHA before push (default: GITHUB_BEFORE)"
    );
    eprintln!("                       --after <sha>         SHA after push  (default: GITHUB_SHA)");
    eprintln!(
        "                       --branch <name>       Branch name     (default: GITHUB_REF_NAME)"
    );
    eprintln!(
        "                       --repo-url <url>      Repo URL        (default: GITHUB_SERVER_URL/GITHUB_REPOSITORY)"
    );
    eprintln!(
        "                       --otel-endpoint <url> OTel base URL   (default: GIT_AI_OTEL_ENDPOINT / config)"
    );
    eprintln!(
        "  metrics install      Install push-metrics workflow to .github/workflows/git-ai-metrics.yaml"
    );
    std::process::exit(1);
}

fn print_ci_gitlab_help_and_exit() -> ! {
    eprintln!("git-ai ci gitlab - GitLab CI utilities");
    eprintln!();
    eprintln!("Usage: git-ai ci gitlab <subcommand> [args...]");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  run [--no-cleanup]   Run GitLab CI in current repo");
    eprintln!("                       --no-cleanup  Skip teardown after run");
    eprintln!("  install              Print YAML snippet to add to .gitlab-ci.yml");
    std::process::exit(1);
}
