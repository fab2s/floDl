//! Tests for `fdl probe`.
//!
//! Beside the module, the shape this crate already uses elsewhere. Two
//! standing rules that these tests exist to keep: never assert a
//! platform's own rendering of a message (assert that the caller routes
//! through the decider, and pin spellings in the decider's own tests),
//! and never exec a file the test just wrote.

use std::path::Path;

use flodl_hw::{GpuArch, GpuVendor};

use super::checks::{check_data_path, check_nccl, gpu_toolkit_warning, unescape_mount};
use super::cluster::parse_remote_json;
use super::report::report_to_json_object;
use super::*;
use crate::config::ClusterWorker;
use crate::libtorch::detect;

// --- GPU toolkit headers -------------------------------------------

#[test]
fn toolkit_warning_names_every_missing_header_and_its_package() {
    // The REAL requirements table, not a hand-picked subset: probe
    // reporting clean while the build fails on the eighth header is
    // exactly the drift this check exists to prevent. (Assumes the
    // test host has no /usr/include/hip — true of the dev and cuda
    // containers.)
    let root = PathBuf::from("/nonexistent/flodl-probe-test/rocm");
    let w = gpu_toolkit_warning(
        "precompiled/rocm70",
        &root,
        "ROCM_PATH",
        crate::util::requirements::ROCM_HEADERS,
        None,
        "rocm",
    )
    .expect("absent toolkit must warn");
    for (header, _) in crate::util::requirements::ROCM_HEADERS {
        assert!(w.contains(header), "missing header {header}: {w}");
    }
    assert!(w.contains("precompiled/rocm70"), "{w}");
    assert!(w.contains("ROCM_PATH"), "{w}");
    // The install line is whatever THIS platform's is: apt names,
    // dnf names, brew with a caveat, or a WSL2 pointer that names no
    // package at all. Asserting one family's spelling is how a green
    // ubuntu run shipped a warning that failed on rocky, macOS and
    // windows at once; the spellings themselves are pinned where they
    // are decided, in `requirements::install_hint`.
    let packages = crate::util::requirements::packages_for(
        &crate::util::requirements::ROCM_HEADERS
            .iter()
            .collect::<Vec<_>>(),
    );
    let hint = crate::util::requirements::install_hint(&packages);
    assert!(w.contains(&hint), "install line not `{hint}`: {w}");
}

#[test]
fn toolkit_warning_says_the_container_path_is_unaffected() {
    // Severity rationale, pinned: flodl's default workflow builds in
    // the dev container, where host headers are irrelevant. If this
    // sentence goes, the warning starts reading like a broken host.
    // The metapackage override is NVIDIA's line: cuda-toolkit covers
    // the set, where the per-header names carry version placeholders.
    let root = PathBuf::from("/nonexistent/flodl-probe-test/cuda");
    let w = gpu_toolkit_warning(
        "precompiled/cu128",
        &root,
        "CUDA_HOME",
        &[("cuda_runtime.h", "cuda-cudart-dev-<M>-<m>")],
        Some("cuda-toolkit libnccl-dev"),
        "cuda",
    )
    .unwrap();
    assert!(w.contains("dev container is unaffected"), "{w}");
    assert!(w.contains("--features cuda"), "{w}");
    let hint = crate::util::requirements::install_hint(&[
        "cuda-toolkit".to_string(),
        "libnccl-dev".to_string(),
    ]);
    assert!(w.contains(&hint), "metapackage line not `{hint}`: {w}");
    assert!(
        !w.contains("<M>-<m>"),
        "placeholders must not reach the user: {w}"
    );
}

#[test]
fn toolkit_present_warns_nothing_and_partial_reports_only_the_gap() {
    // A real include/ layout, because the requirements checker looks
    // under <root>/include (and the system dirs) exactly as the
    // compiler will.
    let root = std::env::temp_dir().join(format!("fdl-probe-toolkit-{}", std::process::id()));
    std::fs::create_dir_all(root.join("include/hip")).unwrap();
    std::fs::write(root.join("include/hip/hip_runtime.h"), "//").unwrap();

    assert!(
        gpu_toolkit_warning(
            "precompiled/rocm70",
            &root,
            "ROCM_PATH",
            &[("hip/hip_runtime.h", "hip-dev")],
            None,
            "rocm",
        )
        .is_none(),
        "a present header must not warn"
    );
    let w = gpu_toolkit_warning(
        "precompiled/rocm70",
        &root,
        "ROCM_PATH",
        &[
            ("hip/hip_runtime.h", "hip-dev"),
            ("rccl/rccl.h", "rccl-dev"),
        ],
        None,
        "rocm",
    )
    .expect("one missing header is still a warning");
    assert!(w.contains("rccl/rccl.h"), "{w}");
    assert!(
        !w.contains("hip_runtime"),
        "must not list the header it found: {w}"
    );
    assert!(!w.contains("hip-dev"), "nor the package it owns: {w}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn cpu_variant_wants_no_toolkit() {
    // `variant_vendor` returns None for a CPU build, which is the
    // gate that keeps this whole check silent on CPU-only hosts.
    assert!(detect::variant_vendor("precompiled/cpu").is_none());
    assert!(detect::variant_vendor("precompiled/cpu-linux-aarch64").is_none());
    // And the vendors that DO imply a toolkit still resolve.
    assert_eq!(
        detect::variant_vendor("precompiled/rocm70"),
        Some(GpuVendor::Amd)
    );
    assert_eq!(
        detect::variant_vendor("precompiled/cu128"),
        Some(GpuVendor::Nvidia)
    );
}

#[test]
fn data_path_check_skipped_when_flag_set() {
    let mut issues = Vec::new();
    let mut warnings = Vec::new();
    let status = check_data_path(
        PathBuf::from("/nonexistent"),
        true,
        false,
        &mut issues,
        &mut warnings,
    );
    assert!(status.skipped);
    assert!(
        issues.is_empty(),
        "skip_mount must suppress missing-path issue"
    );
    assert!(
        warnings.is_empty(),
        "skip_mount must suppress missing-path warning"
    );
}

#[test]
fn data_path_check_explicit_missing_is_error() {
    let mut issues = Vec::new();
    let mut warnings = Vec::new();
    let status = check_data_path(
        PathBuf::from("/this/should/never/exist/flodl-probe-test"),
        false,
        true, // explicit
        &mut issues,
        &mut warnings,
    );
    assert!(!status.exists);
    assert!(!status.readable);
    assert_eq!(issues.len(), 1, "explicit missing path → error");
    assert!(warnings.is_empty());
}

#[test]
fn data_path_check_default_missing_is_warning() {
    let mut issues = Vec::new();
    let mut warnings = Vec::new();
    let status = check_data_path(
        PathBuf::from("/this/should/never/exist/flodl-probe-test"),
        false,
        false, // convention default — not explicit
        &mut issues,
        &mut warnings,
    );
    assert!(!status.exists);
    assert!(issues.is_empty(), "default missing path must NOT error");
    assert_eq!(warnings.len(), 1, "default missing path → warning");
}

#[test]
fn data_path_check_reports_readable_tmp() {
    let mut issues = Vec::new();
    let mut warnings = Vec::new();
    // `env::temp_dir()`, not a literal "/tmp": the assertion is that a
    // path which exists is *reported* as existing, and hardcoding a
    // POSIX path made this fail on Windows for a reason that had
    // nothing to do with check_data_path (which was right to call a
    // missing path missing).
    let status = check_data_path(
        std::env::temp_dir(),
        false,
        false,
        &mut issues,
        &mut warnings,
    );
    // The temp dir is readable on any host that can run this test; if
    // not we'd see it in `issues` and the test would surface the
    // surprise.
    assert!(status.exists);
    assert!(status.readable);
    assert!(issues.is_empty(), "issues = {:?}", issues);
    assert!(warnings.is_empty(), "warnings = {:?}", warnings);
}

#[test]
fn nccl_via_docker_skips_host_scan() {
    let mut issues = Vec::new();
    let status = check_nccl(Some("cuda".into()), &mut issues);
    assert!(
        issues.is_empty(),
        "docker-served NCCL must not produce errors"
    );
    assert!(status.library_path.is_none());
    assert!(status.all_found.is_empty());
    assert_eq!(status.via_docker.as_deref(), Some("cuda"));
}

#[test]
fn verdict_format_three_tier() {
    // No errors, no warnings → READY.
    let r0 = ProbeReport {
        host: "h".into(),
        gpus: vec![],
        libtorch: LibtorchStatus {
            info: None,
            valid_dir: false,
            archs_match: vec![],
        },
        data_path: DataPathStatus {
            path: PathBuf::new(),
            exists: false,
            readable: false,
            fs_type: None,
            skipped: true,
        },
        nccl: NcclStatus {
            library_path: None,
            all_found: vec![],
            via_docker: None,
        },
        issues: vec![],
        warnings: vec![],
    };
    assert!(r0.green());

    // Warning-only is still green (exit 0).
    let r1 = ProbeReport {
        warnings: vec!["w".into()],
        ..clone_report(&r0)
    };
    assert!(r1.green());

    // Error flips green to false.
    let r2 = ProbeReport {
        issues: vec!["e".into()],
        ..clone_report(&r0)
    };
    assert!(!r2.green());
}

// Local clone helper — ProbeReport intentionally not Clone (Vec<GpuInfo>
// has its own ownership).
fn clone_report(r: &ProbeReport) -> ProbeReport {
    ProbeReport {
        host: r.host.clone(),
        gpus: vec![],
        libtorch: LibtorchStatus {
            info: None,
            valid_dir: r.libtorch.valid_dir,
            archs_match: vec![],
        },
        data_path: DataPathStatus {
            path: r.data_path.path.clone(),
            exists: r.data_path.exists,
            readable: r.data_path.readable,
            fs_type: r.data_path.fs_type.clone(),
            skipped: r.data_path.skipped,
        },
        nccl: NcclStatus {
            library_path: r.nccl.library_path.clone(),
            all_found: r.nccl.all_found.clone(),
            via_docker: r.nccl.via_docker.clone(),
        },
        issues: r.issues.clone(),
        warnings: r.warnings.clone(),
    }
}

#[test]
fn json_emits_warnings_array() {
    let r = ProbeReport {
        host: "h".into(),
        gpus: vec![],
        libtorch: LibtorchStatus {
            info: None,
            valid_dir: false,
            archs_match: vec![],
        },
        data_path: DataPathStatus {
            path: PathBuf::new(),
            exists: false,
            readable: false,
            fs_type: None,
            skipped: true,
        },
        nccl: NcclStatus {
            library_path: None,
            all_found: vec![],
            via_docker: Some("cuda".into()),
        },
        issues: vec![],
        warnings: vec!["data-path missing".into()],
    };
    let j = report_to_json_object(&r);
    let v: serde_json::Value = serde_json::from_str(&j).expect("emit valid JSON");
    assert!(v["ready"].as_bool().unwrap());
    let warns = v["warnings"].as_array().expect("warnings: []");
    assert_eq!(warns.len(), 1);
    assert_eq!(v["nccl"]["via_docker"].as_str(), Some("cuda"));
}

#[test]
fn json_survives_control_chars_in_names_and_paths() {
    // A tab / CR in a GPU name or mount path previously produced
    // invalid JSON that broke cluster probe fan-in.
    let r = ProbeReport {
        host: "h\tost".into(),
        gpus: vec![GpuInfo {
            index: 0,
            vendor: GpuVendor::Nvidia,
            name: "Weird\tGPU \"X\"\r\n".into(),
            arch: GpuArch::Sm { major: 8, minor: 6 },
            total_memory_mb: 1024,
        }],
        libtorch: LibtorchStatus {
            info: None,
            valid_dir: false,
            archs_match: vec![],
        },
        data_path: DataPathStatus {
            path: PathBuf::from("/mnt/na\ts"),
            exists: true,
            readable: true,
            fs_type: Some("virtio\u{1}fs".into()),
            skipped: false,
        },
        nccl: NcclStatus {
            library_path: None,
            all_found: vec![],
            via_docker: None,
        },
        issues: vec!["line1\nline2\ttabbed".into()],
        warnings: vec![],
    };
    let j = report_to_json_object(&r);
    let v: serde_json::Value = serde_json::from_str(&j).expect("emit valid JSON");
    assert_eq!(v["gpus"][0]["name"].as_str(), Some("Weird\tGPU \"X\"\r\n"));
    assert_eq!(v["data_path"]["fs_type"].as_str(), Some("virtio\u{1}fs"));
    assert_eq!(v["issues"][0].as_str(), Some("line1\nline2\ttabbed"));
}

#[test]
fn parse_remote_json_flags_schema_skew() {
    // A remote fdl speaking a different probe schema must surface as
    // version skew, not parse as a healthy zero-GPU host.
    let worker: ClusterWorker = serde_yaml_ng::from_str(
        "host: pascal\nlocal_devices: [0]\nnccl_socket_ifname: lo\npath: /opt/flodl",
    )
    .expect("minimal worker");
    let report = parse_remote_json(r#"{"something":"else"}"#, &worker).expect("valid JSON parses");
    assert!(
        report.issues.iter().any(|i| i.contains("version skew")),
        "issues: {:?}",
        report.issues
    );
}

/// Minimal worker fixture for the wire tests below.
fn wire_test_worker() -> ClusterWorker {
    serde_yaml_ng::from_str(
        "host: pascal\nlocal_devices: [0]\nnccl_socket_ifname: lo\npath: /opt/flodl",
    )
    .expect("minimal worker")
}

#[test]
fn gpu_wire_round_trips_both_vendors() {
    // The probe JSON is a real wire: `fdl @cluster probe` SSHes and
    // parses what the remote `fdl probe --json` emitted. Emit and
    // parse must therefore agree for every vendor, or a remote AMD
    // host reads back as something else.
    let r = ProbeReport {
        host: "h".into(),
        gpus: vec![
            GpuInfo {
                index: 0,
                vendor: GpuVendor::Nvidia,
                name: "NVIDIA GeForce RTX 5060 Ti".into(),
                arch: GpuArch::Sm {
                    major: 12,
                    minor: 0,
                },
                total_memory_mb: 16311,
            },
            GpuInfo {
                index: 1,
                vendor: GpuVendor::Amd,
                name: "AMD Radeon RX 6800".into(),
                arch: GpuArch::Gfx("gfx1030".into()),
                total_memory_mb: 16384,
            },
        ],
        libtorch: LibtorchStatus {
            info: None,
            valid_dir: false,
            archs_match: vec![],
        },
        data_path: DataPathStatus {
            path: PathBuf::from("/d"),
            exists: true,
            readable: true,
            fs_type: None,
            skipped: false,
        },
        nccl: NcclStatus {
            library_path: None,
            all_found: vec![],
            via_docker: None,
        },
        issues: vec![],
        warnings: vec![],
    };
    let back = parse_remote_json(&report_to_json_object(&r), &wire_test_worker())
        .expect("emitted JSON parses");
    assert_eq!(back.gpus.len(), 2, "warnings: {:?}", back.warnings);
    assert_eq!(
        back.gpus[0].arch,
        GpuArch::Sm {
            major: 12,
            minor: 0
        }
    );
    assert_eq!(back.gpus[0].vendor, GpuVendor::Nvidia);
    assert_eq!(back.gpus[1].arch, GpuArch::Gfx("gfx1030".into()));
    assert_eq!(back.gpus[1].vendor, GpuVendor::Amd);
    assert_eq!(back.gpus[1].total_memory_mb, 16384);
}

#[test]
fn gpu_wire_reads_a_legacy_sm_only_remote() {
    // An older `fdl` on the remote emits `sm` and no `vendor`/`arch`.
    // It only ever ran on NVIDIA, so that is the right assumption.
    let json = r#"{"host":"p","gpus":[{"index":0,"name":"A100","sm":"sm_80","vram_mb":81920}]}"#;
    let back = parse_remote_json(json, &wire_test_worker()).expect("parses");
    assert_eq!(back.gpus.len(), 1);
    assert_eq!(back.gpus[0].vendor, GpuVendor::Nvidia);
    assert_eq!(back.gpus[0].arch, GpuArch::Sm { major: 8, minor: 0 });
}

#[test]
fn gpu_wire_warns_rather_than_inventing_an_arch() {
    // An unrecognized arch must not fall through to a default: a
    // bogus arch compares as incompatible with every libtorch
    // variant, which reads as a hardware problem the user does not
    // have.
    let json =
        r#"{"host":"p","gpus":[{"index":0,"name":"X","vendor":"amd","arch":"wat","vram_mb":8}]}"#;
    let back = parse_remote_json(json, &wire_test_worker()).expect("parses");
    assert!(back.gpus.is_empty());
    assert!(
        back.warnings.iter().any(|w| w.contains("unrecognized")),
        "warnings: {:?}",
        back.warnings
    );
}

#[test]
fn fs_type_detected_for_root() {
    let t = detect_fs_type(Path::new("/"));
    // / is mounted on every Linux box; detection should not fail.
    // Skip on non-Linux (CI matrix) — /proc/mounts unavailable.
    if std::path::Path::new("/proc/mounts").exists() {
        assert!(t.is_some(), "expected fs_type for /");
    }
}

#[test]
fn mounted_at_answers_only_for_a_real_mount_point() {
    if !std::path::Path::new("/proc/mounts").exists() {
        return;
    }
    // `/` is a mount point on every Linux box.
    assert!(mounted_at(Path::new("/")).is_some());
    // A path INSIDE a mount is not the mount point — this is the
    // whole distinction from `detect_fs_type`, and the one that
    // decides "mount it" from "already mounted".
    let inside = std::env::temp_dir().join("fdl-not-a-mount-point");
    assert!(mounted_at(&inside).is_none());
    assert!(detect_fs_type(&inside).is_some(), "but it has an fs type");
}

#[test]
fn mount_fields_come_back_unescaped() {
    assert_eq!(unescape_mount("exa:/flodl\\040data"), "exa:/flodl data");
    assert_eq!(unescape_mount("plain:/flodl/data"), "plain:/flodl/data");
    // A trailing backslash, or one that is not a full octal escape,
    // is passed through rather than eating the rest of the field.
    assert_eq!(unescape_mount("odd\\"), "odd\\");
    assert_eq!(unescape_mount("odd\\9x"), "odd\\9x");
}
