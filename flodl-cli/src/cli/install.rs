//! `fdl install` (and update) surface — install/update fdl globally + version checks.

use std::env;
use std::process::ExitCode;

use flodl_cli::{cli_error, util};


// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

pub(crate) fn cmd_install(check_only: bool, dev: bool) -> ExitCode {
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    let current_version = env!("CARGO_PKG_VERSION");

    let self_path = match env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            cli_error!("cannot determine own binary path: {e}");
            return ExitCode::FAILURE;
        }
    };

    let home = match env::var_os("HOME").or_else(|| env::var_os("USERPROFILE")) {
        Some(h) => PathBuf::from(h),
        None => {
            cli_error!("cannot determine home directory");
            return ExitCode::FAILURE;
        }
    };

    let bin_dir = home.join(".local/bin");
    let dest = bin_dir.join("fdl");

    // --check: compare versions
    if check_only {
        let latest = fetch_latest_github_tag();
        println!("Installed: {current_version}");
        // Check if current install is a symlink (dev mode)
        if dest.is_symlink() {
            if let Ok(target) = std::fs::read_link(&dest) {
                println!("Mode:      dev (symlink -> {})", target.display());
            }
        }
        match &latest {
            Some(tag) => {
                println!("Latest:    {tag}");
                if tag == current_version {
                    println!("Up to date.");
                } else {
                    println!("Update available. Run: fdl install");
                }
            }
            None => println!("Latest:    (could not check GitHub)"),
        }
        return ExitCode::SUCCESS;
    }

    // Create ~/.local/bin/ if needed
    if let Err(e) = std::fs::create_dir_all(&bin_dir) {
        cli_error!("cannot create {}: {}", bin_dir.display(), e);
        return ExitCode::FAILURE;
    }

    // --dev: symlink to the stable local-build location (~/.cargo/bin/fdl),
    // which is where `cargo install --path flodl-cli` (== `fdl self-build`)
    // drops the freshly-compiled binary. Every subsequent rebuild updates
    // that same path, so the symlink is kept pointing at today's build for
    // free. Falling back to `env::current_exe()` would footgun when the
    // user happens to be running the binary *from* `~/.local/bin/fdl`
    // itself — `canonicalize()` returns that same path and we'd create a
    // symlink to itself (Too many levels of symbolic links).
    if dev {
        #[cfg(unix)]
        {
            let cargo_bin = home.join(".cargo/bin/fdl");
            let self_canonical = self_path.canonicalize().unwrap_or(self_path.clone());

            // Prefer the cargo-install target; fall back to the running
            // binary only when cargo-install isn't present.
            let target = if cargo_bin.is_file() {
                cargo_bin.canonicalize().unwrap_or(cargo_bin)
            } else {
                self_canonical.clone()
            };

            // Self-loop guard: refuse to symlink dest to itself.
            //
            // Compare the fully-resolved `target` against the resolved
            // `dest` when it exists, and fall back to the raw `dest`
            // path otherwise. Raw-only would miss the case where
            // `~/.cargo/bin/fdl` already symlinks (transitively) back
            // to `~/.local/bin/fdl`; canonical-only would fail when
            // `dest` does not yet exist.
            let dest_resolved = dest.canonicalize().unwrap_or_else(|_| dest.clone());
            if target == dest_resolved || target == dest {
                eprintln!(
                    "error: --dev cannot symlink `{}` to itself.",
                    dest.display()
                );
                eprintln!();
                eprintln!(
                    "The currently-running `fdl` is installed at the dest path \
                     and no stable cargo build exists at `{}`.",
                    home.join(".cargo/bin/fdl").display()
                );
                eprintln!();
                eprintln!("Build one first:");
                eprintln!("    cargo install --path flodl-cli");
                eprintln!("    # or (from inside a flodl checkout):");
                eprintln!("    fdl self-build");
                eprintln!();
                eprintln!("Then rerun `fdl install --dev`.");
                return ExitCode::FAILURE;
            }

            // Remove existing (file or symlink) before linking.
            if dest.exists() || dest.is_symlink() {
                let _ = std::fs::remove_file(&dest);
            }

            match std::os::unix::fs::symlink(&target, &dest) {
                Ok(()) => {
                    println!("Linked fdl -> {}", target.display());
                    println!("Global fdl now tracks your local build.");
                    println!("Rebuild with: cargo install --path flodl-cli (or `fdl self-build`).");
                }
                Err(e) => {
                    cli_error!("symlink failed: {e}");
                    return ExitCode::FAILURE;
                }
            }

            return print_path_hint(&bin_dir);
        }

        #[cfg(not(unix))]
        {
            eprintln!("--dev mode requires Unix (symlinks). Use fdl install without --dev.");
            return ExitCode::FAILURE;
        }
    }

    // Normal install: download from GitHub if newer, else copy self
    let latest = fetch_latest_github_tag();

    // Check installed version
    let installed_version = if dest.exists() && !dest.is_symlink() {
        let self_canonical = self_path.canonicalize().unwrap_or(self_path.clone());
        let dest_canonical = dest.canonicalize().unwrap_or(dest.clone());
        if self_canonical == dest_canonical {
            Some(current_version.to_string())
        } else {
            Command::new(&dest)
                .arg("version")
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .output()
                .ok()
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .strip_prefix("flodl-cli ")
                        .map(|v| v.to_string())
                })
        }
    } else {
        None
    };

    // Was it a dev symlink? Switching to copy mode.
    let was_dev = dest.is_symlink();
    if was_dev {
        let _ = std::fs::remove_file(&dest);
    }

    // Decide source: download if newer, else copy self
    let source_path;
    let source_version;
    let mut downloaded_path: Option<PathBuf> = None;

    if let Some(ref tag) = latest {
        if tag != current_version && is_newer(tag, current_version) {
            match download_release_binary(tag, &home) {
                Ok(path) => {
                    source_version = tag.clone();
                    source_path = path.clone();
                    downloaded_path = Some(path);
                }
                Err(e) => {
                    eprintln!("warning: could not download {tag}: {e}");
                    eprintln!("Installing current binary ({current_version}) instead.");
                    source_path = self_path.clone();
                    source_version = current_version.to_string();
                }
            }
        } else {
            source_path = self_path.clone();
            source_version = current_version.to_string();
        }
    } else {
        source_path = self_path.clone();
        source_version = current_version.to_string();
    }

    // Check if update is needed
    if !was_dev {
        if let Some(ref iv) = installed_version {
            if iv == &source_version {
                println!("fdl {} is already installed at {}", iv, dest.display());
                if let Some(ref dl) = downloaded_path {
                    let _ = std::fs::remove_file(dl);
                }
                return ExitCode::SUCCESS;
            }
            println!("Updating fdl {iv} -> {source_version}");
        } else {
            println!("Installing fdl {source_version}");
        }
    } else {
        println!("Switching from dev symlink to installed copy ({source_version})");
    }

    // Copy
    if let Err(e) = std::fs::copy(&source_path, &dest) {
        cli_error!("{e}");
        return ExitCode::FAILURE;
    }

    if let Some(ref dl) = downloaded_path {
        let _ = std::fs::remove_file(dl);
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755));
    }

    println!("Installed fdl {} to {}", source_version, dest.display());
    print_path_hint(&bin_dir)
}

fn print_path_hint(bin_dir: &std::path::Path) -> ExitCode {
    let path_var = env::var("PATH").unwrap_or_default();
    let bin_dir_str = bin_dir.to_string_lossy();
    let in_path = path_var.split(':').any(|p| p == bin_dir_str.as_ref());

    if !in_path {
        println!();
        println!("~/.local/bin is not in your PATH. Add it:");
        println!();
        let shell = env::var("SHELL").unwrap_or_default();
        if shell.contains("zsh") {
            println!("  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.zshrc && source ~/.zshrc");
        } else {
            println!(
                "  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc && source ~/.bashrc"
            );
        }
    }

    ExitCode::SUCCESS
}

/// Fetch the latest release tag from GitHub.
fn fetch_latest_github_tag() -> Option<String> {
    use std::process::{Command, Stdio};
    let output = Command::new("curl")
        .args(["-sI", "https://github.com/flodl-labs/flodl/releases/latest"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.to_lowercase().starts_with("location:") {
            let tag = line.rsplit('/').next()?.trim();
            if !tag.is_empty() {
                return Some(tag.to_string());
            }
        }
    }
    None
}

/// Simple version comparison: "0.3.1" > "0.3.0".
fn is_newer(a: &str, b: &str) -> bool {
    let parse = |v: &str| -> Vec<u32> { v.split('.').filter_map(|s| s.parse().ok()).collect() };
    let va = parse(a);
    let vb = parse(b);
    va > vb
}

/// Download the fdl binary for this platform from a GitHub release.
fn download_release_binary(tag: &str, home: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let os = if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        return Err("unsupported OS".into());
    };

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        if cfg!(target_os = "macos") {
            "arm64"
        } else {
            "aarch64"
        }
    } else {
        return Err("unsupported architecture".into());
    };

    let ext = if cfg!(target_os = "windows") { ".exe" } else { "" };
    let artifact = format!("flodl-cli-{os}-{arch}{ext}");
    let url = format!("https://github.com/flodl-labs/flodl/releases/download/{tag}/{artifact}");

    let tmp = home.join(".flodl").join("tmp");
    std::fs::create_dir_all(&tmp).map_err(|e| format!("cannot create temp dir: {e}"))?;
    let dest = tmp.join(format!("fdl-{tag}{ext}"));

    println!("Downloading fdl {tag} from GitHub...");
    util::http::download_file(&url, &dest)?;

    Ok(dest)
}
