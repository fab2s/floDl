//! Timeline JSON parsing (`load_timeline` + helpers).

use std::path::Path;

use super::{Event, EventKind, GpuSample, Sample, Timeline};

/// Load a timeline from a JSON file.
pub fn load_timeline(path: &Path) -> Result<Timeline, String> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let val: serde_json::Value = serde_json::from_str(&data)
        .map_err(|e| format!("invalid JSON in {}: {e}", path.display()))?;

    let samples = parse_samples(&val["samples"])?;
    let events = parse_events(&val["events"])?;

    Ok(Timeline { samples, events })
}

fn parse_samples(val: &serde_json::Value) -> Result<Vec<Sample>, String> {
    let arr = val.as_array().ok_or("samples is not an array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let t = item["t"].as_u64().unwrap_or(0);
        let gpus = if let Some(gpu_arr) = item["gpus"].as_array() {
            gpu_arr
                .iter()
                .map(|g| GpuSample {
                    device: g["d"].as_u64().unwrap_or(0) as u8,
                    util: g["u"].as_u64().unwrap_or(0) as u8,
                    vram_allocated: g["va"].as_u64().unwrap_or(0),
                    vram_used: g["vu"].as_u64().unwrap_or(0),
                    vram_total: g["vt"].as_u64().unwrap_or(0),
                })
                .collect()
        } else {
            Vec::new()
        };
        out.push(Sample { t, gpus });
    }
    Ok(out)
}

fn parse_events(val: &serde_json::Value) -> Result<Vec<Event>, String> {
    let arr = val.as_array().ok_or("events is not an array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let t = item["t"].as_u64().unwrap_or(0);
        let kind = match item["k"].as_str().unwrap_or("") {
            "epoch_start" => EventKind::EpochStart {
                epoch: item["epoch"].as_u64().unwrap_or(0) as usize,
            },
            "epoch_end" => EventKind::EpochEnd {
                epoch: item["epoch"].as_u64().unwrap_or(0) as usize,
                loss: item["loss"].as_f64().unwrap_or(0.0),
                lr: item["lr"].as_f64().unwrap_or(f64::NAN),
            },
            "sync_start" => EventKind::SyncStart,
            "sync_end" => EventKind::SyncEnd {
                ms: item["ms"].as_f64().unwrap_or(0.0),
            },
            "cpu_avg_start" => EventKind::CpuAvgStart,
            "cpu_avg_end" => EventKind::CpuAvgEnd {
                ms: item["ms"].as_f64().unwrap_or(0.0),
            },
            "anchor" => EventKind::Anchor {
                from: item["from"].as_u64().unwrap_or(0) as usize,
                to: item["to"].as_u64().unwrap_or(0) as usize,
            },
            "throttle" => EventKind::Throttle {
                rank: item["rank"].as_u64().unwrap_or(0) as usize,
            },
            "div" => {
                let deltas = item["deltas"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|v| v.as_f64().unwrap_or(0.0))
                            .collect::<Vec<f64>>()
                    })
                    .unwrap_or_default();
                let pre_norms = item["pre_norms"].as_array().map(|a| {
                    a.iter()
                        .map(|v| v.as_f64().unwrap_or(0.0))
                        .collect::<Vec<f64>>()
                });
                EventKind::Divergence {
                    d_raw: item["d"].as_f64().unwrap_or(0.0),
                    lambda_raw: item["lambda"].as_f64(),
                    lambda_ema: item["lambda_ema"].as_f64(),
                    k_used: item["k_used"].as_u64().unwrap_or(0) as usize,
                    k_max: item["k_max"].as_u64().unwrap_or(0) as usize,
                    step: item["step"].as_u64().unwrap_or(0) as usize,
                    deltas,
                    post_norm: item["post_norm"].as_f64(),
                    pre_norms,
                    epoch: item["epoch"].as_u64().map(|v| v as usize),
                }
            }
            "div_epoch" => EventKind::DivergenceEpoch {
                epoch: item["epoch"].as_u64().unwrap_or(0) as usize,
                sync_count: item["syncs"].as_u64().unwrap_or(0) as usize,
                d_min: item["d_min"].as_f64().unwrap_or(0.0),
                d_max: item["d_max"].as_f64().unwrap_or(0.0),
                d_mean: item["d_mean"].as_f64().unwrap_or(0.0),
                lambda_min: item["lambda_min"].as_f64(),
                lambda_max: item["lambda_max"].as_f64(),
                lambda_mean: item["lambda_mean"].as_f64(),
                lambda_ema_at_epoch_end: item["lambda_ema_end"].as_f64(),
                d_at_epoch_end: item["d_end"].as_f64().unwrap_or(0.0),
                k_at_epoch_end: item["k_end"].as_u64().unwrap_or(0) as usize,
            },
            _ => continue, // skip unknown
        };
        out.push(Event { t, kind });
    }
    Ok(out)
}
