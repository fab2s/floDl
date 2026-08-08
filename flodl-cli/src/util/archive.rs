//! Archive extraction (zip files).
//!
//! Shells out to `unzip` (Linux/macOS) or `PowerShell Expand-Archive` (Windows).

use std::path::Path;
use std::process::{Command, Stdio};

/// Extract a zip file to `dest_dir`.
///
/// Creates `dest_dir` if it doesn't exist.
pub fn extract_zip(zip_path: &Path, dest_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest_dir)
        .map_err(|e| format!("cannot create {}: {}", dest_dir.display(), e))?;

    let zip_str = zip_path
        .to_str()
        .ok_or_else(|| "zip path is not valid UTF-8".to_string())?;
    let dest_str = dest_dir
        .to_str()
        .ok_or_else(|| "destination path is not valid UTF-8".to_string())?;

    let status = if cfg!(target_os = "windows") {
        Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Expand-Archive -Force -Path '{}' -DestinationPath '{}'",
                    zip_str, dest_str
                ),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .status()
    } else {
        // Check for unzip
        if Command::new("unzip")
            .arg("--help")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            return Err("unzip is required but not installed.\n\
                 \n\
                 \x20 Ubuntu/Debian:  sudo apt install unzip\n\
                 \x20 Fedora/RHEL:    sudo dnf install unzip\n\
                 \x20 macOS:          available by default"
                .into());
        }
        Command::new("unzip")
            .args(["-q", "-o", zip_str, "-d", dest_str])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .status()
    };

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!(
            "extraction failed (exit code {})\n  Archive: {}",
            s.code().unwrap_or(-1),
            zip_path.display()
        )),
        Err(e) => Err(format!("failed to run extraction command: {}", e)),
    }
}
