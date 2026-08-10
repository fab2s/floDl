//! Deriving a publish recipe from the training crate's own manifest.
//!
//! Self-contained by design: it reads a `Cargo.toml`, does pure path
//! math, and states its guesses. A path dep on flodl walks the source
//! root UP to the dep so the fetched tree contains it; a workspace above
//! the crate earns an explicit `bin:` caveat rather than a silent wrong
//! answer, because membership is glob-shaped and a wrong guess costs a
//! permanent walk-in failure.

use std::fs;
use std::path::{Path, PathBuf};

// ── The training crate: publish recipe derivation ───────────────────────

/// What one manifest scan decided.
pub(super) struct PublishDerivation {
    pub(super) from_root: PathBuf,
    pub(super) cwd_rel: Option<String>,
    pub(super) bin: String,
    pub(super) build: String,
    pub(super) bin_caveat: Option<String>,
}

/// Derive the publish recipe from the crate's own Cargo.toml. `Ok(None)`
/// when there is no crate here — a farm can be config-only, so absence
/// is a note, not an error.
pub(super) fn derive_publish(crate_dir: &Path) -> Result<Option<PublishDerivation>, String> {
    let manifest_path = crate_dir.join("Cargo.toml");
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("cannot read {}: {e}", manifest_path.display()))?;
    let name = package_name(&manifest).ok_or_else(|| {
        format!(
            "{} has no [package] name — a workspace root? Point --crate at \
             the training crate itself",
            manifest_path.display()
        )
    })?;

    // A path dep on flodl decides the fetched root: the tree must
    // contain the dep or the worker's build dangles on a path outside
    // what it fetched.
    let crate_abs = crate_dir
        .canonicalize()
        .map_err(|e| format!("cannot resolve {}: {e}", crate_dir.display()))?;
    let (from_root, cwd_rel) = match flodl_path_dep(&manifest) {
        Some(rel) => {
            let dep_abs = normalize(&crate_abs.join(&rel));
            let root = common_ancestor(&crate_abs, &dep_abs);
            let cwd_rel = crate_abs
                .strip_prefix(&root)
                .ok()
                .filter(|p| !p.as_os_str().is_empty())
                .map(|p| p.display().to_string());
            (root, cwd_rel)
        }
        None => (crate_abs.clone(), None),
    };

    // The artifact convention: `cargo build` in the crate dir writes to
    // the crate's own target/ unless a WORKSPACE above claims it. The
    // wizard states the guess and the caveat instead of over-guessing —
    // membership is glob-shaped and a wrong silent answer costs a
    // permanent walk-in failure.
    let bin = format!("target/release/{name}");
    let bin_caveat = workspace_above(&crate_abs, &from_root).map(|ws| {
        format!(
            "{} declares [workspace] above the crate — if it claims the \
             crate as a member, the artifact lands in the WORKSPACE \
             target/, so `bin:` must point there (e.g. \
             `{}target/release/{name}`)",
            ws.join("Cargo.toml").display(),
            "../".repeat(
                crate_abs
                    .strip_prefix(&ws)
                    .map(|p| p.components().count())
                    .unwrap_or(1),
            ),
        )
    });

    let features = declares_gpu_features(&manifest);
    let build = if features {
        format!("cargo build --release --features \"$FDL_GPU_FEATURE\" --bin {name}")
    } else {
        format!("cargo build --release --bin {name}")
    };

    Ok(Some(PublishDerivation {
        from_root,
        cwd_rel,
        bin,
        build,
        bin_caveat,
    }))
}

/// `[package] name = "..."` — first `name =` line inside the
/// `[package]` table.
pub(super) fn package_name(manifest: &str) -> Option<String> {
    let mut in_package = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_package = t == "[package]";
            continue;
        }
        if in_package && let Some(rest) = t.strip_prefix("name") {
            let rest = rest.trim_start();
            if let Some(v) = rest.strip_prefix('=') {
                return Some(v.trim().trim_matches('"').to_string());
            }
        }
    }
    None
}

/// The `path = "..."` of a `flodl` dependency, if any — the line-level
/// scan handles both the inline-table form (`flodl = { path = ".." }`)
/// and the `[dependencies.flodl]` table form.
pub(super) fn flodl_path_dep(manifest: &str) -> Option<String> {
    let mut in_flodl_table = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_flodl_table = t == "[dependencies.flodl]"
                || t == "[dev-dependencies.flodl]"
                || t == "[dependencies.flodl-hf]";
            continue;
        }
        let inline = t
            .strip_prefix("flodl")
            .map(|r| r.trim_start())
            .and_then(|r| r.strip_prefix('='))
            .map(|r| r.trim());
        if let Some(spec) = inline {
            if let Some(p) = extract_path_value(spec) {
                return Some(p);
            }
            continue;
        }
        if in_flodl_table
            && let Some(rest) = t.strip_prefix("path")
            && let Some(v) = rest.trim_start().strip_prefix('=')
        {
            return Some(v.trim().trim_matches('"').to_string());
        }
    }
    None
}

/// `path = "..."` inside an inline table value.
pub(super) fn extract_path_value(spec: &str) -> Option<String> {
    let idx = spec.find("path")?;
    let rest = spec[idx + 4..].trim_start().strip_prefix('=')?;
    let rest = rest.trim_start();
    let quoted = rest.strip_prefix('"')?;
    let end = quoted.find('"')?;
    Some(quoted[..end].to_string())
}

/// Whether the crate declares the vendor features the recipe would
/// forward — `cuda` or `rocm` keys under `[features]`.
pub(super) fn declares_gpu_features(manifest: &str) -> bool {
    let mut in_features = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_features = t == "[features]";
            continue;
        }
        if in_features {
            let key = t.split('=').next().unwrap_or("").trim();
            if key == "cuda" || key == "rocm" {
                return true;
            }
        }
    }
    false
}

/// Resolve `..` components without touching the filesystem (the dep
/// path may or may not exist yet on this box).
pub(super) fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

pub(super) fn common_ancestor(a: &Path, b: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for (ca, cb) in a.components().zip(b.components()) {
        if ca == cb {
            out.push(ca);
        } else {
            break;
        }
    }
    out
}

/// Nearest ancestor between the crate and the fetched root (exclusive
/// of the crate itself) whose Cargo.toml declares `[workspace]`.
pub(super) fn workspace_above(crate_abs: &Path, from_root: &Path) -> Option<PathBuf> {
    let mut dir = crate_abs.parent()?;
    loop {
        let m = dir.join("Cargo.toml");
        if m.is_file()
            && let Ok(content) = fs::read_to_string(&m)
            && content.lines().any(|l| l.trim() == "[workspace]")
        {
            return Some(dir.to_path_buf());
        }
        if dir == from_root {
            return None;
        }
        dir = dir.parent()?;
    }
}

pub(super) fn render_publish_block(d: &PublishDerivation) -> String {
    let mut out = String::from("publish:\n");
    out.push_str(&format!("  source: file://{}\n", d.from_root.display()));
    if let Some(cwd) = &d.cwd_rel {
        out.push_str(&format!("  cwd: {cwd}\n"));
    }
    out.push_str(&format!("  build: {}\n", d.build));
    out.push_str(&format!("  bin: {}\n", d.bin));
    out.push_str("  # args: [--model, ..., --epochs, ...]   # the RUN's args\n");
    out
}

/// Does the lockfile still describe the source about to ship? The
/// honest form of "is the compiled version right".
pub(super) fn freshness_report(from_root: &Path) -> String {
    let lock = from_root.join("Cargo.lock");
    let Ok(lock_meta) = fs::metadata(&lock) else {
        return format!(
            "no Cargo.lock at {} yet — the publish gate build will create \
             the verified pin",
            from_root.display(),
        );
    };
    let lock_mtime = lock_meta.modified().ok();
    let newest = newest_source_mtime(from_root);
    match (lock_mtime, newest) {
        (Some(l), Some((n, path))) if n > l => format!(
            "Cargo.lock predates the newest source edit ({}) — the next \
             gate build refreshes the pin; publish before pointing workers \
             at this tree",
            path.display(),
        ),
        (Some(_), Some(_)) => "Cargo.lock is current with the source".to_string(),
        _ => "freshness undetermined (no readable source mtimes)".to_string(),
    }
}

pub(super) fn newest_source_mtime(root: &Path) -> Option<(std::time::SystemTime, PathBuf)> {
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                if matches!(name.as_ref(), "target" | ".git" | ".fdl" | "libtorch") {
                    continue;
                }
                stack.push(path);
            } else if name != "Cargo.lock"
                && let Ok(m) = entry.metadata()
                && let Ok(t) = m.modified()
                && newest.as_ref().is_none_or(|(n, _)| t > *n)
            {
                newest = Some((t, path));
            }
        }
    }
    newest
}
