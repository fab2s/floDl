//! Rendering a verdict: the human report and its JSON twin.
//!
//! Both are projections of one [`ProbeReport`], so the machine form can
//! never describe a different box than the printed one — which matters
//! because the cluster path parses this JSON back out of an SSH pipe.

use std::fmt::Write;

use crate::libtorch::detect;
use crate::util::system;

use super::ProbeReport;

pub(super) fn print_report(r: &ProbeReport) {
    println!("floDl Probe — {}", r.host);
    println!("{}", "=".repeat(40));
    println!();

    println!("GPUs ({}):", r.gpus.len());
    for g in &r.gpus {
        println!(
            "  [{}] {} — {}, {} MB",
            g.index,
            g.short_name(),
            g.arch_label(),
            g.total_memory_mb
        );
    }
    println!();

    println!("libtorch:");
    match &r.libtorch.info {
        Some(info) => {
            println!("  path  : {}", info.path);
            if let Some(t) = &info.torch_version {
                println!("  torch : {}", t);
            }
            // Same reason as `fdl diagnose`: `cuda=` is a CUDA toolkit
            // version, absent (`none`) on both ROCm and CPU builds, so
            // the vendor comes from the variant path instead. The JSON
            // arm below keeps emitting the raw `cuda` field -- it is
            // cluster wire format that remote hosts are parsed back out
            // of, so its shape is not a display decision.
            match detect::variant_vendor(&info.path) {
                Some(v) => println!("  vendor: {}", v),
                None => println!("  vendor: CPU-only"),
            }
            if let Some(c) = info.cuda_version.as_deref().filter(|c| *c != "none") {
                println!("  cuda  : {}", c);
            }
            if let Some(a) = &info.archs {
                println!("  archs : {}", a);
            }
            if !r.libtorch.archs_match.is_empty() {
                let ok = r.libtorch.archs_match.iter().filter(|(_, b)| *b).count();
                println!(
                    "  match : {}/{} GPUs covered",
                    ok,
                    r.libtorch.archs_match.len()
                );
            }
            println!(
                "  valid : {}",
                if r.libtorch.valid_dir { "yes" } else { "no" }
            );
        }
        None => println!("  (not configured)"),
    }
    println!();

    println!("Shared data path:");
    if r.data_path.skipped {
        println!("  (skipped via --skip-mount)");
    } else {
        println!("  path     : {}", r.data_path.path.display());
        println!("  exists   : {}", yn(r.data_path.exists));
        println!("  readable : {}", yn(r.data_path.readable));
        if let Some(t) = &r.data_path.fs_type {
            println!("  fs       : {}", t);
        }
    }
    println!();

    println!("NCCL:");
    if let Some(svc) = &r.nccl.via_docker {
        println!("  via Docker image `{}` (host check skipped)", svc);
    } else {
        match &r.nccl.library_path {
            Some(p) => {
                println!("  found    : {}", p.display());
                if r.nccl.all_found.len() > 1 {
                    println!(
                        "  others   : {} more (check for version skew)",
                        r.nccl.all_found.len() - 1
                    );
                }
            }
            None => println!("  (no libnccl.so* discovered)"),
        }
    }
    println!();

    print_verdict_lines(&r.issues, &r.warnings);
}

/// Render the three-tier verdict + numbered errors/warnings.
fn print_verdict_lines(issues: &[String], warnings: &[String]) {
    let n_err = issues.len();
    let n_warn = warnings.len();
    let line = match (n_err, n_warn) {
        (0, 0) => "verdict: READY".to_string(),
        (0, m) => format!("verdict: READY ({m} warning{})", plural(m)),
        (n, 0) => format!("verdict: ISSUES ({n} error{})", plural(n)),
        (n, m) => format!(
            "verdict: ISSUES ({n} error{}, {m} warning{})",
            plural(n),
            plural(m)
        ),
    };
    println!("{line}");
    if !issues.is_empty() {
        println!("errors:");
        for (i, msg) in issues.iter().enumerate() {
            println!("  {}. {}", i + 1, msg);
        }
    }
    if !warnings.is_empty() {
        println!("warnings:");
        for (i, msg) in warnings.iter().enumerate() {
            println!("  {}. {}", i + 1, msg);
        }
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn yn(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

// ---------------------------------------------------------------------------
// JSON output (`fdl deploy` + CI consume this shape)
// ---------------------------------------------------------------------------

pub(super) fn print_json(r: &ProbeReport) {
    println!("{}", report_to_json_object(r));
}

pub(super) fn report_to_json_object(r: &ProbeReport) -> String {
    let mut b = String::with_capacity(2048);
    b.push('{');
    let _ = write!(b, "\"host\":\"{}\"", system::escape_json(&r.host));

    // GPUs
    b.push_str(",\"gpus\":[");
    for (i, g) in r.gpus.iter().enumerate() {
        if i > 0 {
            b.push(',');
        }
        let _ = write!(
            b,
            "{{\"index\":{},\"name\":\"{}\",\"vendor\":\"{}\",\"arch\":\"{}\",\"sm\":\"{}\",\"vram_mb\":{}}}",
            g.index,
            system::escape_json(&g.name),
            g.vendor.as_str(),
            g.arch_label(),
            // Legacy NVIDIA-only key: an older `fdl` on the controller
            // side reads this one. Empty on a non-NVIDIA device, which
            // such a reader would have mis-handled anyway.
            g.sm_version().unwrap_or_default(),
            g.total_memory_mb
        );
    }
    b.push(']');

    // libtorch
    b.push_str(",\"libtorch\":");
    match &r.libtorch.info {
        Some(info) => {
            let _ = write!(
                b,
                "{{\"path\":\"{}\",\"valid_dir\":{}",
                system::escape_json(&info.path),
                r.libtorch.valid_dir
            );
            if let Some(v) = &info.torch_version {
                let _ = write!(b, ",\"torch\":\"{}\"", system::escape_json(v));
            }
            if let Some(c) = &info.cuda_version {
                let _ = write!(b, ",\"cuda\":\"{}\"", system::escape_json(c));
            }
            if let Some(a) = &info.archs {
                let _ = write!(b, ",\"archs\":\"{}\"", system::escape_json(a));
            }
            b.push_str(",\"archs_match\":[");
            for (i, (gpu, ok)) in r.libtorch.archs_match.iter().enumerate() {
                if i > 0 {
                    b.push(',');
                }
                let _ = write!(b, "{{\"gpu\":{},\"covered\":{}}}", gpu, ok);
            }
            b.push(']');
            b.push('}');
        }
        None => b.push_str("null"),
    }

    // Shared data path
    b.push_str(",\"data_path\":");
    if r.data_path.skipped {
        b.push_str("null");
    } else {
        let _ = write!(
            b,
            "{{\"path\":\"{}\",\"exists\":{},\"readable\":{}",
            system::escape_json(&r.data_path.path.display().to_string()),
            r.data_path.exists,
            r.data_path.readable
        );
        if let Some(t) = &r.data_path.fs_type {
            let _ = write!(b, ",\"fs_type\":\"{}\"", system::escape_json(t));
        }
        b.push('}');
    }

    // NCCL — always emit an object now (even when host scan was
    // skipped via Docker), so consumers can read `via_docker` without
    // null-checking.
    b.push_str(",\"nccl\":");
    if r.nccl.library_path.is_none() && r.nccl.via_docker.is_none() {
        b.push_str("null");
    } else {
        b.push('{');
        let mut first = true;
        if let Some(p) = &r.nccl.library_path {
            let _ = write!(
                b,
                "\"library_path\":\"{}\",\"count\":{}",
                system::escape_json(&p.display().to_string()),
                r.nccl.all_found.len()
            );
            first = false;
        }
        if let Some(svc) = &r.nccl.via_docker {
            if !first {
                b.push(',');
            }
            let _ = write!(b, "\"via_docker\":\"{}\"", system::escape_json(svc));
        }
        b.push('}');
    }

    // Issues (errors) + warnings + verdict.
    b.push_str(",\"issues\":[");
    for (i, msg) in r.issues.iter().enumerate() {
        if i > 0 {
            b.push(',');
        }
        let _ = write!(b, "\"{}\"", system::escape_json(msg));
    }
    b.push(']');
    b.push_str(",\"warnings\":[");
    for (i, msg) in r.warnings.iter().enumerate() {
        if i > 0 {
            b.push(',');
        }
        let _ = write!(b, "\"{}\"", system::escape_json(msg));
    }
    b.push(']');
    let _ = write!(b, ",\"ready\":{}", r.green());
    b.push('}');
    b
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
