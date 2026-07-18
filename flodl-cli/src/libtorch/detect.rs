//! libtorch installation detection and .arch metadata parsing.

use std::fs;
use std::path::Path;

use crate::util::system::GpuInfo;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Metadata about an installed libtorch variant (from `.arch` file).
#[derive(Debug)]
pub struct LibtorchInfo {
    /// Relative path from project root (e.g. "precompiled/cu128", "builds/sm61-sm120").
    pub path: String,
    pub torch_version: Option<String>,
    pub cuda_version: Option<String>,
    pub archs: Option<String>,
    pub source: Option<String>,
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Read the active libtorch variant from `<root>/libtorch/.active` and
/// parse its `.arch` metadata.
///
/// On heterogeneous rigs (multiple hosts sharing the same checkout via
/// NFS / virtiofs / S3-FUSE) a single `.active` file can't represent
/// "PT 2.10 on the Blackwell host AND PT 2.7 on the Pascal host" at
/// the same time. The `FDL_LIBTORCH_CASE` env var selects an
/// alternative pointer file `libtorch/.active.<case>`; the file's
/// content is read identically to `.active`.
///
/// Setting `FDL_LIBTORCH_CASE=<case>` with no corresponding pointer
/// file is a hard misconfiguration: this function logs the missing
/// file to stderr and returns `None` (callers surface this as "libtorch
/// not configured") rather than silently falling back to `.active`,
/// which would otherwise mask the user's explicit selection.
pub fn read_active(root: &Path) -> Option<LibtorchInfo> {
    let lt_dir = root.join("libtorch");
    let pointer = match std::env::var("FDL_LIBTORCH_CASE") {
        Ok(case) if !case.trim().is_empty() => {
            let case = case.trim();
            let case_file = lt_dir.join(format!(".active.{case}"));
            if !case_file.exists() {
                eprintln!(
                    "fdl: FDL_LIBTORCH_CASE={case} but `{}` does not exist. \
                     Create it with `fdl libtorch use <variant> --as {case}` \
                     or hand-write the variant path (e.g. \
                     `precompiled/cu128`).",
                    case_file.display(),
                );
                return None;
            }
            case_file
        }
        _ => lt_dir.join(".active"),
    };
    read_active_from(&pointer, &lt_dir)
}

/// Read any `.active*` pointer file and resolve its content (a
/// relative path like `precompiled/cu128` or `builds/sm61-sm120`)
/// against `libtorch_root` to produce a [`LibtorchInfo`].
///
/// Used by [`read_active`] (default `.active`), by callers that have
/// the pointer file path directly (e.g. cluster.yml's per-host
/// `libtorch_path:` set to `…/libtorch/.active.<case>`), and by tests
/// that need to validate a pointer-file shape without setting the
/// `FDL_LIBTORCH_CASE` env var.
pub fn read_active_from(pointer: &Path, libtorch_root: &Path) -> Option<LibtorchInfo> {
    let active = fs::read_to_string(pointer).ok()?;
    let path = active.trim().to_string();
    if path.is_empty() {
        return None;
    }
    let arch_dir = libtorch_root.join(&path);
    Some(libtorch_info_from_dir(path, &arch_dir))
}

/// Build a [`LibtorchInfo`] for a variant directory: `path` is the string
/// recorded in the info (a relative variant path like `precompiled/cu128`
/// or an absolute directory), `arch_dir` is the directory whose `.arch`
/// file supplies the metadata. The four metadata fields stay `None` when
/// the `.arch` file is absent or unreadable. One home for the parse that
/// `read_active_from`, `run::resolve_libtorch_at`, and probe's
/// `check_libtorch*` all used to copy inline.
pub(crate) fn libtorch_info_from_dir(path: String, arch_dir: &Path) -> LibtorchInfo {
    let mut info = LibtorchInfo {
        path,
        torch_version: None,
        cuda_version: None,
        archs: None,
        source: None,
    };
    if let Ok(content) = fs::read_to_string(arch_dir.join(".arch")) {
        parse_arch_into(&content, &mut info);
    }
    info
}

/// Fill a [`LibtorchInfo`]'s metadata fields from `.arch` file content
/// (`torch=` / `cuda=` / `archs=` / `source=` lines; unknown lines ignored).
fn parse_arch_into(content: &str, info: &mut LibtorchInfo) {
    for line in content.lines() {
        if let Some(val) = line.strip_prefix("torch=") {
            info.torch_version = Some(val.to_string());
        } else if let Some(val) = line.strip_prefix("cuda=") {
            info.cuda_version = Some(val.to_string());
        } else if let Some(val) = line.strip_prefix("archs=") {
            info.archs = Some(val.to_string());
        } else if let Some(val) = line.strip_prefix("source=") {
            info.source = Some(val.to_string());
        }
    }
}

/// Record per-GPU arch coverage for a resolved variant and push a loud,
/// actionable issue for every GPU the libtorch build does not cover — or a
/// single issue when the variant carries no `.arch` metadata. Returns
/// `(gpu_index, covered)` pairs in GPU order. One home for the coverage
/// loop probe's three `check_libtorch*` paths used to copy inline (with
/// drifted wording).
pub(crate) fn arch_coverage(
    info: &LibtorchInfo,
    gpus: &[GpuInfo],
    issues: &mut Vec<String>,
) -> Vec<(u8, bool)> {
    let mut archs_match = Vec::new();
    if let Some(archs) = &info.archs {
        for g in gpus {
            let ok = arch_compatible(g, archs);
            archs_match.push((g.index, ok));
            if !ok {
                issues.push(format!(
                    "GPU {} ({}, {}) not covered by libtorch archs `{}`. \
                     Rebuild libtorch with this arch or activate a \
                     compatible variant.",
                    g.index,
                    g.short_name(),
                    g.sm_version(),
                    archs
                ));
            }
        }
    } else {
        issues.push(
            "libtorch is present but `.arch` metadata is missing — cannot \
             verify GPU compatibility. Place an `.arch` file in the variant \
             directory (cuda=, torch=, archs=, source=)."
                .into(),
        );
    }
    archs_match
}

/// List all installed libtorch variants under `<root>/libtorch/`.
///
/// Scans `precompiled/` and `builds/` subdirectories.
pub fn list_variants(root: &Path) -> Vec<String> {
    let mut variants = Vec::new();
    let lt_dir = root.join("libtorch");

    for subdir in ["precompiled", "builds"] {
        let dir = lt_dir.join(subdir);
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if entry.path().join("lib").is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        variants.push(format!("{}/{}", subdir, name));
                    }
                }
            }
        }
    }

    variants.sort();
    variants
}

/// Check whether a GPU's compute capability is covered by the libtorch
/// variant's compiled architectures (from the .arch file).
pub fn arch_compatible(gpu: &GpuInfo, archs: &str) -> bool {
    let exact = format!("{}.{}", gpu.sm_major, gpu.sm_minor);
    archs.contains(&exact) || archs.contains(&format!("{}", gpu.sm_major))
}

/// Check whether a libtorch variant directory looks valid (has lib/).
pub fn is_valid_variant(root: &Path, variant: &str) -> bool {
    root.join(format!("libtorch/{}/lib", variant)).is_dir()
}

/// Set the active libtorch variant by writing `<root>/libtorch/.active`.
pub fn set_active(root: &Path, variant: &str) -> Result<(), String> {
    let lt_dir = root.join("libtorch");
    fs::create_dir_all(&lt_dir)
        .map_err(|e| format!("cannot create libtorch/: {}", e))?;
    fs::write(lt_dir.join(".active"), format!("{}\n", variant))
        .map_err(|e| format!("cannot write libtorch/.active: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test_env::env_lock;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    // Per-process counter so concurrent test binaries don't collide
    // on the unique-suffix algorithm.
    static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

    /// Hand-rolled scratch dir under the system temp dir + RAII
    /// cleanup. flodl-cli is zero-external-crate by policy so we
    /// cannot pull in `tempfile`.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let seq = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir()
                .join(format!("fdl-libtorch-resolver-{}-{}", nanos, seq));
            fs::create_dir_all(&dir).expect("create scratch");
            Self(dir)
        }
        fn path(&self) -> &std::path::Path { &self.0 }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Build a fake project root with two synthetic libtorch variants
    /// (`precompiled/v1` and `builds/v2`), each with a `lib/` dir and
    /// `.arch` metadata. The names are deliberately abstract — the
    /// resolver doesn't care about variant naming, just the
    /// `<kind>/<name>` shape it reads from the pointer file.
    fn make_root() -> Scratch {
        let s = Scratch::new();
        let v1 = s.path().join("libtorch/precompiled/v1");
        fs::create_dir_all(v1.join("lib")).unwrap();
        fs::write(
            v1.join(".arch"),
            "torch=1.0\ncuda=1.0\narchs=0.0\nsource=precompiled\n",
        ).unwrap();
        let v2 = s.path().join("libtorch/builds/v2");
        fs::create_dir_all(v2.join("lib")).unwrap();
        fs::write(
            v2.join(".arch"),
            "torch=2.0\ncuda=2.0\narchs=1.0\nsource=build\n",
        ).unwrap();
        s
    }

    #[test]
    fn read_active_default_pointer() {
        let _guard = env_lock();
        // SAFETY: serialized via env_lock().
        unsafe { std::env::remove_var("FDL_LIBTORCH_CASE"); }
        let root = make_root();
        fs::write(
            root.path().join("libtorch/.active"),
            "precompiled/v1\n",
        ).unwrap();
        let info = read_active(root.path()).expect("read_active");
        assert_eq!(info.path, "precompiled/v1");
        assert_eq!(info.torch_version.as_deref(), Some("1.0"));
    }

    #[test]
    fn fdl_libtorch_case_selects_alternate_pointer() {
        let _guard = env_lock();
        let root = make_root();
        fs::write(
            root.path().join("libtorch/.active"),
            "builds/v2\n",
        ).unwrap();
        fs::write(
            root.path().join("libtorch/.active.alt"),
            "precompiled/v1\n",
        ).unwrap();
        // SAFETY: serialized via env_lock().
        unsafe { std::env::set_var("FDL_LIBTORCH_CASE", "alt"); }
        let info = read_active(root.path()).expect("read_active");
        // SAFETY: serialized via env_lock().
        unsafe { std::env::remove_var("FDL_LIBTORCH_CASE"); }
        assert_eq!(info.path, "precompiled/v1");
        assert_eq!(info.torch_version.as_deref(), Some("1.0"));
    }

    #[test]
    fn fdl_libtorch_case_missing_file_returns_none_loudly() {
        let _guard = env_lock();
        let root = make_root();
        fs::write(
            root.path().join("libtorch/.active"),
            "builds/v2\n",
        ).unwrap();
        // No `.active.nonexistent` file.
        // SAFETY: serialized via env_lock().
        unsafe { std::env::set_var("FDL_LIBTORCH_CASE", "nonexistent"); }
        let info = read_active(root.path());
        // SAFETY: serialized via env_lock().
        unsafe { std::env::remove_var("FDL_LIBTORCH_CASE"); }
        assert!(info.is_none(),
            "explicit case with missing file must not silently fall back to .active");
    }

    #[test]
    fn read_active_from_resolves_pointer_directly() {
        let _guard = env_lock();
        let root = make_root();
        let pointer = root.path().join("libtorch/.active.alt");
        fs::write(&pointer, "builds/v2\n").unwrap();
        let info = read_active_from(
            &pointer, &root.path().join("libtorch"),
        ).expect("read_active_from");
        assert_eq!(info.path, "builds/v2");
        assert_eq!(info.archs.as_deref(), Some("1.0"));
    }
}
