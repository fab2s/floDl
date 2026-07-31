//! `fdl diagnose` -- system and GPU diagnostics.
//!
//! Thin formatting layer over `util::system` and `libtorch::detect`.

use std::fmt::Write;
use std::path::Path;

use crate::context::Context;
use crate::libtorch::detect;
use crate::util::system;

pub fn run(json: bool) {
    let ctx = Context::resolve();
    let root = &ctx.root;
    if json {
        print_json(root, &ctx);
    } else {
        print_report(root, &ctx);
    }
}

// ---------------------------------------------------------------------------
// Human-readable report
// ---------------------------------------------------------------------------

fn print_report(root: &Path, ctx: &Context) {
    println!("floDl Diagnostics");
    println!("=================");
    println!();

    // Context
    println!("Context:       {}", ctx.label());
    println!();

    // System
    println!("System");
    let cpu = system::cpu_model().unwrap_or_else(|| "Unknown".into());
    let threads = system::cpu_threads();
    let ram_gb = system::ram_total_gb();
    println!("  CPU:         {} ({} threads, {}GB RAM)", cpu, threads, ram_gb);
    if let Some(os) = system::os_version() {
        println!("  OS:          {}", os);
    }
    if system::is_inside_docker() {
        println!("  Docker:      yes (running inside container)");
    } else {
        match system::docker_version() {
            Some(v) => println!("  Docker:      {}", v),
            None => println!("  Docker:      not found"),
        }
    }
    println!();

    // CUDA / GPU
    println!("CUDA");
    let devices = system::detect_gpus();
    if !devices.is_empty() {
        if let Some(driver) = system::nvidia_driver_version() {
            println!("  Driver:      {}", driver);
        }
        println!("  Devices:     {}", devices.len());
        for d in &devices {
            let vram_gb = d.total_memory_mb / 1024;
            println!(
                "  [{}] {} -- {}, {}GB VRAM",
                d.index,
                d.name,
                d.arch_label(),
                vram_gb
            );
        }
    } else {
        println!("  No CUDA devices available");
    }
    println!();

    // libtorch
    println!("libtorch");
    match detect::read_active(root) {
        Some(info) => {
            println!("  Active:      {}", info.path);
            if let Some(v) = &info.torch_version {
                println!("  Version:     {}", v);
            }
            if let Some(c) = &info.cuda_version {
                println!("  CUDA:        {}", c);
            }
            if let Some(a) = &info.archs {
                println!("  Archs:       {}", a);
            }
            if let Some(s) = &info.source {
                println!("  Source:      {}", s);
            }
        }
        None => {
            println!("  No active variant (run `fdl setup`)");
        }
    }

    let variants = detect::list_variants(root);
    if !variants.is_empty() {
        println!("  Variants:    {}", variants.join(", "));
    }
    println!();

    // Compatibility
    if !devices.is_empty() {
        println!("Compatibility");
        if let Some(info) = detect::read_active(root) {
            let archs = info.archs.as_deref().unwrap_or("");
            let mut all_ok = true;
            for d in &devices {
                if d.covered_by(archs) {
                    println!(
                        "  GPU {} ({}, {}):  OK",
                        d.index,
                        d.short_name(),
                        d.arch_label()
                    );
                } else {
                    all_ok = false;
                    println!(
                        "  GPU {} ({}, {}):  MISSING -- arch {} not in [{}]",
                        d.index,
                        d.short_name(),
                        d.arch_label(),
                        // The archs= spelling, not the display one: this
                        // names the token the user must add to the list.
                        d.arch.archs_token(),
                        archs
                    );
                }
            }
            if all_ok {
                println!();
                println!("  All GPUs compatible with active libtorch.");
            }
        } else {
            println!("  Cannot check -- no active libtorch variant.");
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

fn print_json(root: &Path, ctx: &Context) {
    let mut b = String::with_capacity(2048);
    b.push('{');

    // Context
    let _ = write!(
        b,
        "\"context\":{{\"mode\":\"{}\",\"root\":\"{}\"}}",
        if ctx.is_project { "project" } else { "global" },
        system::escape_json(&ctx.root.display().to_string())
    );

    // System
    let cpu = system::cpu_model().unwrap_or_else(|| "Unknown".into());
    let _ = write!(
        b,
        ",\"system\":{{\"cpu\":\"{}\",\"threads\":{},\"ram_gb\":{}",
        system::escape_json(&cpu),
        system::cpu_threads(),
        system::ram_total_gb()
    );
    if let Some(os) = system::os_version() {
        let _ = write!(b, ",\"os\":\"{}\"", system::escape_json(&os));
    }
    if system::is_inside_docker() {
        b.push_str(",\"docker\":\"container\"");
    } else if let Some(docker) = system::docker_version() {
        let _ = write!(b, ",\"docker\":\"{}\"", system::escape_json(&docker));
    }
    b.push('}');

    // GPUs
    let devices = system::detect_gpus();
    let archs = detect::read_active(root)
        .and_then(|info| info.archs)
        .unwrap_or_default();
    b.push_str(",\"gpus\":[");
    for (i, d) in devices.iter().enumerate() {
        if i > 0 {
            b.push(',');
        }
        let compatible = d.covered_by(&archs);
        // `sm` is the legacy NVIDIA-only key, kept so an older reader
        // does not lose the field; `vendor` + `arch` are the
        // vendor-plural pair every new consumer should read.
        let _ = write!(
            b,
            "{{\"index\":{},\"name\":\"{}\",\"vendor\":\"{}\",\"arch\":\"{}\",\"sm\":\"{}\",\"vram_bytes\":{},\"arch_compatible\":{}}}",
            d.index,
            system::escape_json(&d.name),
            d.vendor.as_str(),
            d.arch_label(),
            d.sm_version().unwrap_or_default(),
            d.vram_bytes(),
            compatible
        );
    }
    b.push(']');

    // libtorch
    b.push_str(",\"libtorch\":");
    match detect::read_active(root) {
        Some(info) => {
            let _ = write!(b, "{{\"path\":\"{}\"", system::escape_json(&info.path));
            if let Some(v) = &info.torch_version {
                let _ = write!(b, ",\"version\":\"{}\"", system::escape_json(v));
            }
            if let Some(c) = &info.cuda_version {
                let _ = write!(b, ",\"cuda\":\"{}\"", system::escape_json(c));
            }
            if let Some(a) = &info.archs {
                let _ = write!(b, ",\"archs\":\"{}\"", system::escape_json(a));
            }
            if let Some(s) = &info.source {
                let _ = write!(b, ",\"source\":\"{}\"", system::escape_json(s));
            }
            b.push('}');
        }
        None => b.push_str("null"),
    }

    b.push('}');
    println!("{}", b);
}
