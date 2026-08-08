//! HTTP file downloads via curl or wget.
//!
//! Shells out to curl/wget because TLS in pure std is impractical without
//! crates, and these tools are ubiquitous (curl ships with Windows 10+).

use std::path::Path;
use std::process::{Command, Stdio};

use super::system::has_command;

/// Detected download tool.
enum Downloader {
    Curl,
    Wget,
}

fn detect_downloader() -> Result<Downloader, String> {
    if has_command("curl") {
        Ok(Downloader::Curl)
    } else if has_command("wget") {
        Ok(Downloader::Wget)
    } else {
        Err("Neither curl nor wget is installed.\n\
             Install one of them:\n\
             \n\
             \x20 Ubuntu/Debian:  sudo apt install curl\n\
             \x20 Fedora/RHEL:    sudo dnf install curl\n\
             \x20 macOS:          available by default\n\
             \x20 Windows 10+:    curl.exe is built-in"
            .into())
    }
}

/// Download a file from `url` to `dest`, showing progress on the terminal.
///
/// Overwrites `dest` if it exists. Creates parent directories.
pub fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    let dl = detect_downloader()?;

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create directory {}: {}", parent.display(), e))?;
    }

    let dest_str = dest
        .to_str()
        .ok_or_else(|| "destination path is not valid UTF-8".to_string())?;

    let status = match dl {
        Downloader::Curl => Command::new("curl")
            .args(["-L", "--progress-bar", "-o", dest_str, url])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status(),
        Downloader::Wget => Command::new("wget")
            .args(["-q", "--show-progress", "-O", dest_str, url])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status(),
    };

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!(
            "download failed (exit code {})\n  URL: {}",
            s.code().unwrap_or(-1),
            url
        )),
        Err(e) => Err(format!("failed to run download command: {}", e)),
    }
}
