//! The NVIDIA backend: `nvidia-smi`.

use std::path::Path;
use std::process::Command;

use crate::report::{GpuSurvey, NoteKind};
use crate::vendor::{GpuArch, GpuVendor};
use crate::GpuInfo;

/// Cheap, subprocess-free check for "is there an NVIDIA driver here".
///
/// Gates the `nvidia-smi` spawn so a pure-AMD or CPU-only box pays a
/// couple of `stat` calls instead of a process launch. `/dev/nvidiactl`
/// is the control node every NVIDIA driver creates;
/// `/proc/driver/nvidia` covers containers that map the proc entry but
/// not the device nodes.
fn driver_present() -> bool {
    Path::new("/dev/nvidiactl").exists() || Path::new("/proc/driver/nvidia").exists()
}

/// Probe NVIDIA devices, appending to `out`.
///
/// Records a note rather than returning an error: a survey is a report
/// on a whole machine, and one vendor's absence is not a failure of the
/// sweep.
pub(crate) fn probe(out: &mut GpuSurvey) {
    // On non-Linux there are no device nodes to check, so fall through
    // to the subprocess and let its absence answer.
    if cfg!(target_os = "linux") && !driver_present() {
        return;
    }

    let output = match Command::new("nvidia-smi")
        // `name` is queried LAST: it is the only field that can contain
        // the `", "` separator, so with it last `splitn(4, ", ")` keeps
        // the whole name (commas and all) in the final cell. No CSV
        // parser needed, since the first three cells are comma-free by
        // construction.
        .args([
            "--query-gpu=index,compute_cap,memory.total,name",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        // nvidia-smi present but errored: a driver or permission
        // problem, or an old bundle that doesn't support the
        // `compute_cap` query field. Distinct from "not installed",
        // and the difference is what the user needs told.
        Ok(o) => {
            out.note(
                GpuVendor::Nvidia,
                NoteKind::ToolFailed,
                format!(
                    "`nvidia-smi` exited {}: {}. An NVIDIA driver is loaded but the \
                     tool cannot enumerate devices, so this box reports 0 NVIDIA GPUs.",
                    o.status,
                    String::from_utf8_lossy(&o.stderr).trim(),
                ),
            );
            return;
        }
        Err(_) => {
            // Driver nodes exist (we got past the gate) but the CLI is
            // missing. Real on minimal containers that map /dev/nvidia*
            // without installing the utilities.
            if cfg!(target_os = "linux") {
                out.note(
                    GpuVendor::Nvidia,
                    NoteKind::HardwareUnusable,
                    "an NVIDIA driver is present but `nvidia-smi` is not on PATH, so \
                     GPUs cannot be enumerated. Install the NVIDIA utilities package \
                     (or, in Docker, use a CUDA base image / the NVIDIA container \
                     toolkit)."
                        .to_string(),
                );
            }
            return;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        match parse_csv_row(line) {
            Some(g) => out.devices.push(g),
            // A malformed row is a GPU silently dropped. Leave a trace.
            None => out.note(
                GpuVendor::Nvidia,
                NoteKind::Unparsable,
                format!("could not parse an `nvidia-smi` row, GPU skipped: {line:?}"),
            ),
        }
    }
}

/// Parse one `index, compute_cap, memory.total, name` CSV row
/// (`--format=csv,noheader,nounits`). `None` on any bad field.
fn parse_csv_row(line: &str) -> Option<GpuInfo> {
    let parts: Vec<&str> = line.splitn(4, ", ").collect();
    if parts.len() < 4 {
        return None;
    }
    Some(GpuInfo {
        index: parts[0].trim().parse().ok()?,
        vendor: GpuVendor::Nvidia,
        arch: GpuArch::parse(GpuVendor::Nvidia, parts[1])?,
        total_memory_mb: parts[2].trim().parse().ok()?,
        name: parts[3].trim().to_string(),
    })
}

/// NVIDIA driver version string, or `None` when `nvidia-smi` is absent
/// or errors.
pub fn nvidia_driver_version() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout);
    Some(s.lines().next()?.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_well_formed_row() {
        // Column order: index, compute_cap, memory.total, name.
        let g = parse_csv_row("1, 6.1, 6078, NVIDIA GeForce GTX 1060 6GB").unwrap();
        assert_eq!(g.index, 1);
        assert_eq!(g.name, "NVIDIA GeForce GTX 1060 6GB");
        assert_eq!(g.arch, GpuArch::Sm { major: 6, minor: 1 });
        assert_eq!(g.vendor, GpuVendor::Nvidia);
        assert_eq!(g.total_memory_mb, 6078);
    }

    #[test]
    fn keeps_a_comma_inside_the_name() {
        // `name` last means an embedded ", " stays in the final cell
        // (splitn(4) stops after 3 separators) rather than truncating.
        let g = parse_csv_row("0, 8.0, 81920, NVIDIA A100, 80GB").unwrap();
        assert_eq!(g.name, "NVIDIA A100, 80GB");
        assert_eq!(g.total_memory_mb, 81920);
    }

    #[test]
    fn rejects_malformed_rows() {
        assert!(parse_csv_row("0, 8.9, three").is_none());
        assert!(parse_csv_row("x, 8.9, 24564, name").is_none());
        assert!(parse_csv_row("0, notacap, 24564, name").is_none());
    }
}
