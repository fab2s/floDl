//! Cross-platform system detection (CPU, RAM, OS, Docker, GPU).

#[cfg(target_os = "linux")]
use std::fs;
use std::path::Path;
use std::process::Command;

// ---------------------------------------------------------------------------
// GPU detection via nvidia-smi
// ---------------------------------------------------------------------------

pub struct GpuInfo {
    pub index: u8,
    pub name: String,
    pub sm_major: u32,
    pub sm_minor: u32,
    pub total_memory_mb: u64,
}

impl GpuInfo {
    pub fn sm_version(&self) -> String {
        format!("sm_{}{}", self.sm_major, self.sm_minor)
    }

    pub fn vram_bytes(&self) -> u64 {
        self.total_memory_mb * 1024 * 1024
    }

    pub fn short_name(&self) -> String {
        self.name.replace("NVIDIA ", "").replace("GeForce ", "")
    }
}

pub fn detect_gpus() -> Vec<GpuInfo> {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(4, ", ").collect();
            if parts.len() < 4 {
                return None;
            }
            let index: u8 = parts[0].trim().parse().ok()?;
            let name = parts[1].trim().to_string();
            let cap_parts: Vec<&str> = parts[2].trim().split('.').collect();
            let sm_major: u32 = cap_parts.first()?.parse().ok()?;
            let sm_minor: u32 = cap_parts.get(1)?.parse().ok()?;
            let total_memory_mb: u64 = parts[3].trim().parse().ok()?;
            Some(GpuInfo {
                index,
                name,
                sm_major,
                sm_minor,
                total_memory_mb,
            })
        })
        .collect()
}

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

// ---------------------------------------------------------------------------
// CPU
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
pub fn cpu_model() -> Option<String> {
    let info = fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in info.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            if let Some(val) = rest.split(':').nth(1) {
                return Some(val.trim().to_string());
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
pub fn cpu_model() -> Option<String> {
    let out = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

#[cfg(target_os = "windows")]
pub fn cpu_model() -> Option<String> {
    let out = Command::new("wmic")
        .args(["cpu", "get", "Name", "/value"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    for line in s.lines() {
        if let Some(val) = line.strip_prefix("Name=") {
            let v = val.trim();
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
pub fn cpu_threads() -> usize {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .map(|s| s.lines().filter(|l| l.starts_with("processor")).count())
        .unwrap_or(1)
}

#[cfg(target_os = "macos")]
pub fn cpu_threads() -> usize {
    Command::new("sysctl")
        .args(["-n", "hw.logicalcpu"])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse()
                .ok()
        })
        .unwrap_or(1)
}

#[cfg(target_os = "windows")]
pub fn cpu_threads() -> usize {
    std::env::var("NUMBER_OF_PROCESSORS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

// ---------------------------------------------------------------------------
// RAM
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
pub fn ram_total_gb() -> u64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                    return Some(kb / (1024 * 1024));
                }
            }
            None
        })
        .unwrap_or(0)
}

#[cfg(target_os = "macos")]
pub fn ram_total_gb() -> u64 {
    Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| {
            let bytes: u64 = String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse()
                .ok()?;
            Some(bytes / (1024 * 1024 * 1024))
        })
        .unwrap_or(0)
}

#[cfg(target_os = "windows")]
pub fn ram_total_gb() -> u64 {
    Command::new("wmic")
        .args(["os", "get", "TotalVisibleMemorySize", "/value"])
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout);
            for line in s.lines() {
                if let Some(val) = line.strip_prefix("TotalVisibleMemorySize=") {
                    let kb: u64 = val.trim().parse().ok()?;
                    return Some(kb / (1024 * 1024));
                }
            }
            None
        })
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// OS
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
pub fn os_version() -> Option<String> {
    let uname = Command::new("uname").arg("-r").output().ok()?;
    let kernel = String::from_utf8_lossy(&uname.stdout).trim().to_string();
    let wsl = if kernel.contains("WSL") || kernel.contains("microsoft") {
        " (WSL2)"
    } else {
        ""
    };
    Some(format!("Linux {}{}", kernel, wsl))
}

#[cfg(target_os = "macos")]
pub fn os_version() -> Option<String> {
    let out = Command::new("sw_vers")
        .args(["-productVersion"])
        .output()
        .ok()?;
    let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if ver.is_empty() {
        return None;
    }
    let arch = Command::new("uname")
        .arg("-m")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    if arch.is_empty() {
        Some(format!("macOS {}", ver))
    } else {
        Some(format!("macOS {} ({})", ver, arch))
    }
}

#[cfg(target_os = "windows")]
pub fn os_version() -> Option<String> {
    let out = Command::new("cmd")
        .args(["/C", "ver"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

// ---------------------------------------------------------------------------
// Docker
// ---------------------------------------------------------------------------

pub fn is_inside_docker() -> bool {
    Path::new("/.dockerenv").exists()
}

pub fn docker_version() -> Option<String> {
    let out = Command::new("docker").arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    s.split("version ")
        .nth(1)
        .and_then(|v| v.split(',').next())
        .map(|v| v.trim().to_string())
}

/// Check whether cargo is available on the host.
#[allow(dead_code)]
pub fn has_cargo() -> bool {
    Command::new("cargo")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether a command exists on PATH.
#[allow(dead_code)]
pub fn has_command(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

/// Platform string for download URLs (e.g. "linux-x86_64", "macos-arm64").
#[allow(dead_code)]
pub fn platform_tag() -> Option<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    match (os, arch) {
        ("linux", "x86_64") => Some("linux-x86_64".into()),
        ("macos", "aarch64") => Some("macos-arm64".into()),
        ("windows", "x86_64") => Some("windows-x86_64".into()),
        _ => None,
    }
}

/// Escape a string for embedding in a JSON string literal. Complete per
/// RFC 8259: backslash, quote, and every control char below 0x20. The
/// hand-rolled predecessors missed `\t` / `\r` — one control character in
/// a GPU name or mount path produced invalid JSON, which broke cluster
/// probe fan-in.
pub fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("\\u{:04x}", c as u32),
                );
            }
            c => out.push(c),
        }
    }
    out
}

/// Convert a `;`-separated CUDA capability list into a variant directory
/// name: `"6.1;12.0"` -> `"sm61-sm120"`, `"12.0"` -> `"sm120"`. Shared by
/// the libtorch and NCCL source builders so their variant paths cannot drift.
pub fn arch_dir_name(archs: &str) -> String {
    archs
        .split(';')
        .map(|cap| {
            let clean = cap.replace('.', "");
            format!("sm{}", clean)
        })
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(test)]
mod tests {
    use super::{arch_dir_name, escape_json};

    #[test]
    fn escape_json_passes_through_plain_ascii() {
        assert_eq!(escape_json("hello world 123"), "hello world 123");
    }

    #[test]
    fn escape_json_escapes_quotes_and_backslashes() {
        // A Windows-style path with quotes is the realistic hazard for the
        // probe/diagnose JSON output this feeds.
        assert_eq!(escape_json(r#"C:\a\b"#), r#"C:\\a\\b"#);
        assert_eq!(escape_json(r#"say "hi""#), r#"say \"hi\""#);
    }

    #[test]
    fn escape_json_escapes_named_control_chars() {
        assert_eq!(escape_json("a\nb\rc\td"), "a\\nb\\rc\\td");
        assert_eq!(escape_json("\u{08}\u{0C}"), "\\b\\f");
    }

    #[test]
    fn escape_json_uescapes_other_control_chars() {
        // < 0x20 with no short form -> \uXXXX (lowercase, 4 hex digits).
        assert_eq!(escape_json("\u{01}"), "\\u0001");
        assert_eq!(escape_json("\u{1f}"), "\\u001f");
    }

    #[test]
    fn escape_json_leaves_non_ascii_unescaped() {
        // >= 0x20 passes through verbatim, including multibyte UTF-8.
        assert_eq!(escape_json("café — 日本"), "café — 日本");
    }

    #[test]
    fn arch_dir_name_single() {
        assert_eq!(arch_dir_name("12.0"), "sm120");
    }

    #[test]
    fn arch_dir_name_multi() {
        assert_eq!(arch_dir_name("6.1;12.0"), "sm61-sm120");
    }

    #[test]
    fn arch_dir_name_strips_all_dots() {
        // A three-component cap and a two-digit minor both flatten correctly.
        assert_eq!(arch_dir_name("7.5"), "sm75");
        assert_eq!(arch_dir_name("8.0;8.6;9.0"), "sm80-sm86-sm90");
    }
}
