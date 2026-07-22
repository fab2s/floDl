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
            "meta_nudge" => EventKind::MetaNudge {
                factor: item["factor"].as_f64().unwrap_or(0.0),
                from: item["from"].as_u64().unwrap_or(0) as usize,
                to: item["to"].as_u64().unwrap_or(0) as usize,
            },
            // "div" / "div_epoch" (MSF passive-observation records) fall
            // through to the skip: the JSON detail stays available to
            // research tooling, the report no longer consumes it.
            _ => continue, // skip unknown
        };
        out.push(Event { t, kind });
    }
    Ok(out)
}
