//! libtorch installation detection and .arch metadata parsing.

use std::fs;
use std::path::{Path, PathBuf};

use crate::util::system::GpuInfo;
use flodl_hw::GpuVendor;

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
/// `arch:` naming a case pointer, resolving to
/// `…/libtorch/.active.<case>`), and by tests
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
            let ok = g.covered_by(archs);
            archs_match.push((g.index, ok));
            if !ok {
                issues.push(format!(
                    "GPU {} ({}, {}) not covered by libtorch archs `{}`. \
                     Rebuild libtorch with this arch or activate a \
                     compatible variant.",
                    g.index,
                    g.short_name(),
                    g.arch_label(),
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

/// Which GPU stack a libtorch variant path targets, from its basename.
///
/// `None` means a CPU-only variant. The variant path (`precompiled/cu128`,
/// `builds/sm61-sm120`, `precompiled/cpu`) is the single source of truth
/// here -- no `.arch` metadata file is required -- because the cluster
/// `arch:` field names exactly this path and must resolve without
/// reading the remote host's filesystem.
///
/// | Basename starts with | Target |
/// |---|---|
/// | `cpu` | CPU-only |
/// | `cu<digit>` (`cu128`, `cu126-pt27`) or `sm<digit>` (`sm61-sm120`) | NVIDIA |
/// | `rocm<digit>` or `gfx<digit>` (`gfx1030-gfx1100`) | AMD |
///
/// An unrecognised basename **warns and is treated as NVIDIA**. That
/// preserves the pre-multi-vendor behaviour exactly, which matters
/// because a user may well have a hand-named CUDA variant
/// (`builds/mybuild`) that works today; hard-erroring would break a
/// running setup for the sake of a naming convention. The warning is
/// the point: the old code made the same assumption in silence, and an
/// unrecognised basename on an AMD box would otherwise be cross-built
/// for NVIDIA without a word.
pub fn variant_vendor(variant: &str) -> Option<GpuVendor> {
    // The naming convention has ONE home (flodl-hw, where the join
    // admission gate also reads it); the warn-and-assume-NVIDIA
    // fallback is this router's policy, not the convention's.
    match flodl_hw::classify_variant_label(variant) {
        flodl_hw::VariantClass::Cpu => None,
        flodl_hw::VariantClass::Vendor(v) => Some(v),
        flodl_hw::VariantClass::Unknown => {
            eprintln!(
                "fdl: libtorch variant {variant:?} does not match a known naming \
                 convention (cpu / cu<N> / sm<N> / rocm<N> / gfx<N>); assuming it is \
                 an NVIDIA build. Rename it to match, or pass the feature explicitly."
            );
            Some(GpuVendor::Nvidia)
        }
    }
}

/// The `export` lines a native-build recipe prints for a variant's
/// vendor, in order.
///
/// **On ROCm the system runtime goes FIRST**, ahead of libtorch's own
/// `lib/`. Same D1a ordering `Dockerfile.rocm` and the cluster pre-build
/// carry, and for the same reason: libtorch-rocm bundles the entire
/// userspace ROCm stack (libamdhip64, libhsa-runtime64, libamd_comgr,
/// and the kernel-interface-coupled libdrm / libnuma), so with libtorch
/// first that bundle wins over the host's, and when it disagrees with
/// the host's amdkfd driver the process segfaults at its FIRST GPU op.
/// A recipe printed the other way round IS that configuration, handed
/// to the user to paste.
///
/// `$ROCM_PATH` is honored (these recipes run on the LOCAL host, so its
/// env is the right authority) with `/opt/rocm` as the convention
/// default. A path that does not exist is skipped by the loader, so the
/// prefix costs nothing where there is no system ROCm. The cluster
/// pre-build deliberately does NOT do this: the path it builds names a
/// REMOTE host, where the controller's `$ROCM_PATH` would be the wrong
/// machine's answer.
///
/// `libtorch_lib` is how the recipe spells the libtorch lib directory:
/// an absolute path for the standalone installer, `$LIBTORCH_PATH/lib`
/// where the recipe just exported that variable.
///
/// One home on purpose. Three sites print this recipe (`fdl setup`,
/// `fdl libtorch download`, `fdl libtorch build`) and each grew its own
/// copy; two of them had the order backwards, which is not a cosmetic
/// drift but the segfault configuration.
pub fn ld_library_path_lines(vendor: Option<GpuVendor>, libtorch_lib: &str) -> Vec<String> {
    let tail = "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}";
    match vendor {
        Some(GpuVendor::Amd) => {
            // `lib` vs `lib64` is a distro property (RHEL/SUSE use
            // lib64), and a recipe naming the wrong one is a path the
            // loader silently skips — the segfault configuration again.
            // The recipe runs on THIS host, so probe the actual layout
            // and keep the `$ROCM_PATH` indirection for the root.
            let libdir = flodl_hw::rocm_runtime_lib_dir()
                .and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned()))
                .unwrap_or_else(|| "lib".to_string());
            vec![
                "export ROCM_PATH=\"${ROCM_PATH:-/opt/rocm}\"".to_string(),
                format!("export LD_LIBRARY_PATH=\"$ROCM_PATH/{libdir}:{libtorch_lib}{tail}\""),
            ]
        }
        _ => vec![format!("export LD_LIBRARY_PATH=\"{libtorch_lib}{tail}\"")],
    }
}

/// This box's active libtorch as `(variant directory, variant label)`,
/// anchored on `root` (a project root, or the global one).
///
/// The directory rather than its `lib/`: a build wants `LIBTORCH_PATH`
/// (headers included) and a child process wants `lib/`, so one value
/// serves both and neither caller has to guess which it was handed.
pub fn active_variant(root: &Path) -> Option<(PathBuf, String)> {
    let info = read_active(root)?;
    let dir = root.join("libtorch").join(&info.path);
    dir.join("lib").is_dir().then_some((dir, info.path))
}

/// `LD_LIBRARY_PATH` VALUE for running against `libtorch_lib`, in the
/// order the loader must see it. The sibling of
/// [`ld_library_path_lines`], which prints the same ordering as a shell
/// recipe; this one is for setting on a child process.
///
/// `rocm_lib` is the system runtime's LIBRARY directory, passed rather
/// than resolved here because the two callers describe different
/// filesystems: a locally spawned child gets this box's resolved
/// directory ([`local_rocm_lib_dir`]), while a path composed for a
/// REMOTE host must use the convention (`/opt/rocm/lib`) since our own
/// environment says nothing about theirs.
///
/// **On ROCm the system runtime must come FIRST.** libtorch-rocm bundles
/// the entire userspace ROCm stack, so with libtorch first that bundle
/// wins over the host's, and when it disagrees with the host's amdkfd
/// driver the process segfaults at its FIRST GPU OP — a failure that
/// looks nothing like a library-path problem. A path that does not exist
/// is skipped by the loader, so prefixing costs nothing on a box without
/// ROCm.
pub fn ld_library_path_value(
    vendor: Option<GpuVendor>,
    libtorch_lib: &str,
    rocm_lib: &str,
) -> String {
    match vendor {
        Some(GpuVendor::Amd) => {
            format!("{}:{libtorch_lib}", rocm_lib.trim_end_matches('/'))
        }
        _ => libtorch_lib.to_string(),
    }
}

/// The system ROCm runtime's library directory on THIS box, for
/// [`ld_library_path_value`]'s local callers.
///
/// `flodl-hw` resolves it properly (`$ROCM_PATH` / `$HIP_PATH` /
/// `$HSA_PATH` / `/opt/rocm`, probing `lib` and `lib64` for the actual
/// runtime): detection and the loader path MUST agree, or a box passes
/// the GPU gate on the runtime detection found and then segfaults on
/// the path a weaker resolution composed. Falls back to the
/// `$ROCM_PATH`-or-convention spelling when no runtime is found — the
/// loader skips a missing path, so the prefix stays harmless.
pub fn local_rocm_lib_dir() -> String {
    match flodl_hw::rocm_runtime_lib_dir() {
        Some(dir) => dir.display().to_string(),
        None => format!(
            "{}/lib",
            std::env::var("ROCM_PATH")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| "/opt/rocm".to_string())
                .trim_end_matches('/'),
        ),
    }
}

/// The cargo feature a variant needs, or `""` for a CPU-only variant.
pub fn variant_feature(variant: &str) -> &'static str {
    match variant_vendor(variant) {
        None => "",
        Some(v) => v.cargo_feature(),
    }
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
                if entry.path().join("lib").is_dir()
                    && let Some(name) = entry.file_name().to_str() {
                        variants.push(format!("{}/{}", subdir, name));
                    }
            }
        }
    }

    variants.sort();
    variants
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
    /// cleanup. flodl-cli keeps external deps minimal (the serde
    /// ecosystem only — no utility crates like `tempfile`) so we
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
    fn variant_vendor_reads_the_naming_convention() {
        for (path, want) in [
            ("precompiled/cpu", None),
            ("precompiled/cu128", Some(GpuVendor::Nvidia)),
            ("precompiled/cu126-pt27", Some(GpuVendor::Nvidia)),
            ("builds/sm61-sm120", Some(GpuVendor::Nvidia)),
            ("builds/sm80", Some(GpuVendor::Nvidia)),
            ("precompiled/rocm63", Some(GpuVendor::Amd)),
            ("builds/gfx1030-gfx1100", Some(GpuVendor::Amd)),
            ("builds/gfx942", Some(GpuVendor::Amd)),
        ] {
            assert_eq!(variant_vendor(path), want, "{path}");
        }
    }

    #[test]
    fn variant_vendor_requires_a_digit_after_the_prefix() {
        // `cpu` must not read as a `cu`-something, and a bare `gfx`
        // directory is not an arch.
        assert_eq!(variant_vendor("precompiled/cpu"), None);
        assert_eq!(variant_vendor("x/cpu-static"), None);
        // Unrecognised names warn and fall back to NVIDIA rather than
        // breaking a hand-named CUDA build that works today.
        assert_eq!(variant_vendor("builds/mybuild"), Some(GpuVendor::Nvidia));
        assert_eq!(variant_vendor("builds/gfx"), Some(GpuVendor::Nvidia));
    }

    #[test]
    fn ld_recipe_puts_system_rocm_before_libtorch() {
        // D1a. The ORDER is the whole point: the other way round is the
        // configuration that segfaults at the first GPU op, and these
        // lines are pasted verbatim by whoever ran the command.
        for lib in ["$LIBTORCH_PATH/lib", "/opt/lt/rocm70/lib"] {
            let lines = ld_library_path_lines(Some(GpuVendor::Amd), lib);
            let ld = lines
                .iter()
                .find(|l| l.contains("LD_LIBRARY_PATH="))
                .expect("recipe must set LD_LIBRARY_PATH");
            let rocm = ld.find("$ROCM_PATH/lib").expect("system ROCm must be on the path");
            let libtorch = ld.find(lib).expect("libtorch must be on the path");
            assert!(rocm < libtorch, "system ROCm must come first, got {ld}");
            assert!(
                lines.iter().any(|l| l.contains("ROCM_PATH:-/opt/rocm")),
                "an unset ROCM_PATH must fall back to the convention: {lines:?}"
            );
        }
    }

    #[test]
    fn ld_recipe_is_libtorch_only_for_nvidia_and_cpu() {
        for vendor in [Some(GpuVendor::Nvidia), None] {
            let lines = ld_library_path_lines(vendor, "$LIBTORCH_PATH/lib");
            assert_eq!(lines.len(), 1, "{vendor:?}");
            assert!(!lines[0].contains("rocm"), "{vendor:?}: {}", lines[0]);
            assert!(lines[0].contains("$LIBTORCH_PATH/lib"), "{}", lines[0]);
        }
    }

    #[test]
    fn ld_recipe_preserves_an_existing_ld_library_path() {
        // The `:+` guard keeps a user's existing value and avoids the
        // trailing colon that would otherwise put CWD on the loader path.
        for vendor in [Some(GpuVendor::Amd), Some(GpuVendor::Nvidia), None] {
            let lines = ld_library_path_lines(vendor, "/opt/lt/lib");
            let ld = lines.iter().find(|l| l.contains("LD_LIBRARY_PATH=")).unwrap();
            assert!(
                ld.contains("${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"),
                "{vendor:?}: {ld}"
            );
        }
    }

    #[test]
    fn variant_feature_maps_to_the_cargo_feature() {
        assert_eq!(variant_feature("precompiled/cpu"), "");
        assert_eq!(variant_feature("precompiled/cu128"), "cuda");
        assert_eq!(variant_feature("builds/gfx1030"), "rocm");
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

/// Unmet dynamic-linker requirements of a libtorch variant on THIS host,
/// as the loader itself reports them.
///
/// A libtorch archive is built against some baseline C library, and the
/// baseline is not the same across variants: measured on 2.10.0, the cpu
/// and cu128 trees need `GLIBC_2.29` / `GLIBCXX_3.4.26` while the rocm7.0
/// tree needs `GLIBC_2.35` / `GLIBCXX_3.4.30`. RHEL 9 ships glibc 2.34
/// and cannot be upgraded past it, so that last combination cannot run
/// there at all — and without this check the operator finds out after a
/// download, a compile and a link, from a loader error naming symbol
/// versions rather than the actual problem.
///
/// Asks `ldd`, so it answers by the same rules the real load obeys
/// instead of a table of baselines that would rot at the next release.
/// An empty vector means "nothing unmet", which is also what a missing
/// `ldd` returns: this reports a problem it can prove, never a doubt.
pub fn unmet_loader_requirements(variant_dir: &Path) -> Vec<String> {
    let core = variant_dir.join("lib/libtorch_cpu.so");
    if !core.is_file() {
        return Vec::new();
    }
    let Ok(out) = std::process::Command::new("ldd").arg(&core).output() else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&out.stdout).into_owned()
        + &String::from_utf8_lossy(&out.stderr);
    parse_unmet_versions(&text)
}

/// The symbol versions an `ldd` run reported as missing, de-duplicated
/// in first-seen order. Pure so the parse is testable against real
/// loader output rather than only on a host that happens to fail.
pub(crate) fn parse_unmet_versions(ldd_output: &str) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for line in ldd_output.lines() {
        // `... version `GLIBC_2.35' not found (required by ...)`
        if !line.contains("not found") {
            continue;
        }
        let Some(rest) = line.split("version `").nth(1) else { continue };
        let Some(sym) = rest.split('\'').next() else { continue };
        if !seen.iter().any(|s| s == sym) {
            seen.push(sym.to_string());
        }
    }
    seen
}

#[cfg(test)]
mod loader_tests {
    use super::parse_unmet_versions;

    /// Real `ldd` output, captured 2026-08-07 from the rocm7.0 variant
    /// on rockylinux:9 — the pair CI hit.
    #[test]
    fn it_reads_the_versions_the_loader_could_not_satisfy() {
        let real = "\
/lt/libtorch_cpu.so: /lib64/libm.so.6: version `GLIBC_2.35' not found (required by /lt/libtorch_cpu.so)
/lt/libtorch_cpu.so: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lt/libtorch_cpu.so)
/lt/libtorch_cpu.so: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lt/libc10.so)
\tlinux-vdso.so.1 (0x00007ffd0d7f9000)
\tlibm.so.6 => /lib64/libm.so.6 (0x00007f0e8a000000)
";
        assert_eq!(
            parse_unmet_versions(real),
            vec!["GLIBC_2.35".to_string(), "GLIBCXX_3.4.30".to_string()],
            "de-duplicated, in first-seen order",
        );
    }

    /// A host that CAN load it says nothing, and neither do we: this
    /// reports a problem it can prove, never a doubt.
    #[test]
    fn a_satisfied_load_reports_nothing() {
        let ok = "\
\tlinux-vdso.so.1 (0x00007ffd0d7f9000)
\tlibtorch_cpu.so => /lt/libtorch_cpu.so (0x00007f0e88000000)
\tlibm.so.6 => /lib64/libm.so.6 (0x00007f0e8a000000)
";
        assert!(parse_unmet_versions(ok).is_empty());
        assert!(parse_unmet_versions("").is_empty());
    }
}
