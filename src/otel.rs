//! OpenTelemetry metrics exporter.
//!
//! Sends git-ai metrics to an OTLP HTTP/JSON endpoint (e.g. an OTel Collector
//! forwarding to Prometheus/Grafana).
//!
//! Two metrics are emitted per flush:
//!   - `git_ai_lines_generated_total`  – lines added in AI checkpoint events
//!   - `git_ai_lines_accepted_total`   – AI lines accepted at commit time
//!
//! Both are reported as delta sums so each flush represents only the events in
//! that batch.  Labels: `agent`, `model`, `repo_url`.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

use crate::metrics::MetricEvent;

// ── OTLP/HTTP JSON types ─────────────────────────────────────────────────────

#[derive(Serialize)]
struct OtlpPayload {
    #[serde(rename = "resourceMetrics")]
    resource_metrics: Vec<ResourceMetrics>,
}

#[derive(Serialize)]
struct ResourceMetrics {
    resource: Resource,
    #[serde(rename = "scopeMetrics")]
    scope_metrics: Vec<ScopeMetrics>,
}

#[derive(Serialize)]
struct Resource {
    attributes: Vec<KeyValue>,
}

#[derive(Serialize)]
struct ScopeMetrics {
    scope: InstrumentationScope,
    metrics: Vec<Metric>,
}

#[derive(Serialize)]
struct InstrumentationScope {
    name: String,
    version: String,
}

#[derive(Serialize)]
struct Metric {
    name: String,
    description: String,
    unit: String,
    sum: Sum,
}

#[derive(Serialize)]
struct Sum {
    #[serde(rename = "dataPoints")]
    data_points: Vec<NumberDataPoint>,
    /// 1 = DELTA, 2 = CUMULATIVE
    #[serde(rename = "aggregationTemporality")]
    aggregation_temporality: u8,
    #[serde(rename = "isMonotonic")]
    is_monotonic: bool,
}

#[derive(Serialize)]
struct NumberDataPoint {
    attributes: Vec<KeyValue>,
    #[serde(rename = "startTimeUnixNano")]
    start_time_unix_nano: String,
    #[serde(rename = "timeUnixNano")]
    time_unix_nano: String,
    #[serde(rename = "asInt")]
    as_int: i64,
}

#[derive(Serialize, Clone)]
struct KeyValue {
    key: String,
    value: AnyValue,
}

#[derive(Serialize, Clone)]
struct AnyValue {
    #[serde(rename = "stringValue")]
    string_value: String,
}

// ── Label key type ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MetricLabels {
    agent: String,
    model: String,
    repo_url: String,
    user_email: String,
}

impl MetricLabels {
    fn to_kv(&self) -> Vec<KeyValue> {
        vec![
            kv("agent", &self.agent),
            kv("model", &self.model),
            kv("repo_url", &self.repo_url),
            kv("user_email", &self.user_email),
        ]
    }
}

/// Extract the email address from a git author string.
///
/// Handles formats:
/// - `"Name <email>"` → `"email"`
/// - `"<email>"`       → `"email"`
/// - `"email"`         → `"email"` (returned as-is)
fn extract_email(author: &str) -> &str {
    if let Some(start) = author.find('<') {
        if let Some(end) = author[start..].find('>') {
            return &author[start + 1..start + end];
        }
    }
    author
}

fn kv(key: &str, value: &str) -> KeyValue {
    KeyValue {
        key: key.to_string(),
        value: AnyValue {
            string_value: value.to_string(),
        },
    }
}

// ── Event parsing helpers ─────────────────────────────────────────────────────

/// Extract a string from a sparse array by numeric position key.
fn sparse_str(map: &HashMap<String, Value>, pos: usize) -> String {
    map.get(&pos.to_string())
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Extract a u32 from a sparse array by numeric position key.
fn sparse_u32(map: &HashMap<String, Value>, pos: usize) -> u32 {
    map.get(&pos.to_string())
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

/// Extract a Vec<u32> from a sparse array by numeric position key.
fn sparse_vec_u32(map: &HashMap<String, Value>, pos: usize) -> Vec<u32> {
    map.get(&pos.to_string())
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default()
}

/// Extract a Vec<String> from a sparse array by numeric position key.
fn sparse_vec_str(map: &HashMap<String, Value>, pos: usize) -> Vec<String> {
    map.get(&pos.to_string())
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

// ── Metric aggregation ────────────────────────────────────────────────────────

/// Aggregate `lines_generated` (from AI Checkpoint events, event_id=4),
/// `lines_accepted` (from Committed events, event_id=1), and
/// `lines_human` (human_additions from Committed events, event_id=1)
/// keyed by label set.
fn aggregate(
    events: &[MetricEvent],
) -> (
    HashMap<MetricLabels, i64>,
    HashMap<MetricLabels, i64>,
    HashMap<MetricLabels, i64>,
) {
    let mut generated: HashMap<MetricLabels, i64> = HashMap::new();
    let mut accepted: HashMap<MetricLabels, i64> = HashMap::new();
    let mut human: HashMap<MetricLabels, i64> = HashMap::new();

    for event in events {
        match event.event_id {
            // Checkpoint (4): one event per file per AI checkpoint
            4 => {
                let kind = sparse_str(&event.values, 1 /* KIND */);
                if kind == "human" {
                    continue;
                }
                let lines_added = sparse_u32(&event.values, 3 /* LINES_ADDED */);
                if lines_added == 0 {
                    continue;
                }
                let agent = sparse_str(&event.attrs, 20 /* TOOL */);
                let model = sparse_str(&event.attrs, 21 /* MODEL */);
                let repo_url = sparse_str(&event.attrs, 1 /* REPO_URL */);
                let author = sparse_str(&event.attrs, 2 /* AUTHOR */);
                let user_email = extract_email(&author).to_string();
                let labels = MetricLabels {
                    agent,
                    model,
                    repo_url,
                    user_email,
                };
                *generated.entry(labels).or_default() += lines_added as i64;
            }

            // Committed (1): parallel arrays tool_model_pairs / ai_accepted + scalar human_additions
            1 => {
                let repo_url = sparse_str(&event.attrs, 1 /* REPO_URL */);
                let author = sparse_str(&event.attrs, 2 /* AUTHOR */);
                let user_email = extract_email(&author).to_string();

                // AI accepted lines: per-tool breakdown via parallel arrays
                let tool_model_pairs = sparse_vec_str(&event.values, 3 /* TOOL_MODEL_PAIRS */);
                let ai_accepted_vec = sparse_vec_u32(&event.values, 6 /* AI_ACCEPTED */);

                // index 0 is always the "all" aggregate – skip it
                for (pair, &count) in tool_model_pairs.iter().zip(ai_accepted_vec.iter()).skip(1) {
                    if count == 0 {
                        continue;
                    }
                    let (agent, model) = pair
                        .split_once(':')
                        .map(|(a, m)| (a.to_string(), m.to_string()))
                        .unwrap_or_else(|| (pair.clone(), String::new()));
                    let labels = MetricLabels {
                        agent,
                        model,
                        repo_url: repo_url.clone(),
                        user_email: user_email.clone(),
                    };
                    *accepted.entry(labels).or_default() += count as i64;
                }

                // Human additions: scalar field, no agent/model breakdown
                let human_additions = sparse_u32(&event.values, 0 /* HUMAN_ADDITIONS */);
                if human_additions > 0 {
                    let labels = MetricLabels {
                        agent: String::new(),
                        model: String::new(),
                        repo_url,
                        user_email,
                    };
                    *human.entry(labels).or_default() += human_additions as i64;
                }
            }

            _ => {}
        }
    }

    (generated, accepted, human)
}

// ── OTLP payload builder ──────────────────────────────────────────────────────

fn build_payload(
    generated: &HashMap<MetricLabels, i64>,
    accepted: &HashMap<MetricLabels, i64>,
    human: &HashMap<MetricLabels, i64>,
    now_nano: u64,
    start_nano: u64,
) -> OtlpPayload {
    let start_str = start_nano.to_string();
    let now_str = now_nano.to_string();

    let mut metrics = Vec::new();

    // git_ai_lines_generated_total
    if !generated.is_empty() {
        let data_points: Vec<NumberDataPoint> = generated
            .iter()
            .map(|(labels, &value)| NumberDataPoint {
                attributes: labels.to_kv(),
                start_time_unix_nano: start_str.clone(),
                time_unix_nano: now_str.clone(),
                as_int: value,
            })
            .collect();

        metrics.push(Metric {
            name: "git_ai_lines_generated_total".to_string(),
            description: "Lines of code written by an AI agent".to_string(),
            unit: "lines".to_string(),
            sum: Sum {
                data_points,
                aggregation_temporality: 1, // DELTA
                is_monotonic: true,
            },
        });
    }

    // git_ai_lines_accepted_total
    if !accepted.is_empty() {
        let data_points: Vec<NumberDataPoint> = accepted
            .iter()
            .map(|(labels, &value)| NumberDataPoint {
                attributes: labels.to_kv(),
                start_time_unix_nano: start_str.clone(),
                time_unix_nano: now_str.clone(),
                as_int: value,
            })
            .collect();

        metrics.push(Metric {
            name: "git_ai_lines_accepted_total".to_string(),
            description: "AI-written lines committed (accepted) by the developer".to_string(),
            unit: "lines".to_string(),
            sum: Sum {
                data_points,
                aggregation_temporality: 1, // DELTA
                is_monotonic: true,
            },
        });
    }

    // git_ai_lines_human_total
    if !human.is_empty() {
        let data_points: Vec<NumberDataPoint> = human
            .iter()
            .map(|(labels, &value)| NumberDataPoint {
                attributes: labels.to_kv(),
                start_time_unix_nano: start_str.clone(),
                time_unix_nano: now_str.clone(),
                as_int: value,
            })
            .collect();

        metrics.push(Metric {
            name: "git_ai_lines_human_total".to_string(),
            description: "Lines of code written by humans and committed".to_string(),
            unit: "lines".to_string(),
            sum: Sum {
                data_points,
                aggregation_temporality: 1, // DELTA
                is_monotonic: true,
            },
        });
    }

    OtlpPayload {
        resource_metrics: vec![ResourceMetrics {
            resource: Resource {
                attributes: vec![kv("service.name", "git-ai")],
            },
            scope_metrics: vec![ScopeMetrics {
                scope: InstrumentationScope {
                    name: "git-ai".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                },
                metrics,
            }],
        }],
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Send metric events to the configured OTel endpoint.
///
/// `endpoint` should be the base URL of the OTel Collector, e.g.
/// `http://localhost:4318`.  The path `/v1/metrics` is appended automatically.
///
/// `bearer_token` is an optional Authorization: Bearer token for the endpoint.
///
/// Returns `Ok(())` on HTTP 2xx, `Err(message)` otherwise.
pub fn send_to_otel(
    endpoint: &str,
    bearer_token: Option<&str>,
    events: &[MetricEvent],
) -> Result<(), String> {
    let (generated, accepted, human) = aggregate(events);

    if generated.is_empty() && accepted.is_empty() && human.is_empty() {
        return Ok(());
    }

    let now_nano = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    // Use a 1-second window as start time for delta reporting
    let start_nano = now_nano.saturating_sub(1_000_000_000);

    let payload = build_payload(&generated, &accepted, &human, now_nano, start_nano);

    let json = serde_json::to_string(&payload)
        .map_err(|e| format!("otel: failed to serialize payload: {}", e))?;

    let url = format!("{}/v1/metrics", endpoint.trim_end_matches('/'));

    let mut request = minreq::post(&url).with_header("Content-Type", "application/json");

    if let Some(token) = bearer_token {
        request = request.with_header("Authorization", format!("Bearer {}", token));
    }

    let response = request
        .with_body(json)
        .send()
        .map_err(|e| format!("otel: HTTP request to {} failed: {}", url, e))?;

    let status = response.status_code;
    if (200..300).contains(&status) {
        Ok(())
    } else {
        Err(format!("otel: endpoint {} returned HTTP {}", url, status))
    }
}
