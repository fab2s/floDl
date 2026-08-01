//! What a host needs before floDl will build, and the command that
//! supplies it.
//!
//! # Why this is one module and not checks at each point of use
//!
//! Every requirement here already had a check somewhere, and that was
//! the problem: they fired one failure at a time. `util/http.rs`
//! explains curl when a download starts, `util/archive.rs` explains
//! unzip when an extract starts, `flodl-sys/build.rs` explains the
//! vendor headers minutes into a compile. A person on a fresh box
//! discovers the list by hitting it, one round trip each.
//!
//! `fdl probe` and `fdl setup` are the commands run FIRST, so they name
//! the whole list at once and hand back a single command. The per-site
//! checks stay: they are the backstop for anyone who skipped the tool.
//!
//! # Context decides the list
//!
//! Requirements are not absolute, they depend on how the user will
//! build. The Docker path needs none of the native toolchain -- no C++
//! compiler, no vendor headers -- because all of it lives in the image.
//! Reporting them as missing to a Docker user would be noise, so callers
//! ask for the set that matches the path being taken.

use std::path::Path;

use crate::util::system;

/// Host tools `fdl` itself shells out to: (probe name, Debian package).
///
/// `curl` is special-cased by the caller -- wget satisfies the same
/// need, and `util/http.rs` accepts either.
const HOST_TOOLS: &[(&str, &str)] = &[
    ("curl", "curl"),
    ("unzip", "unzip"),
    ("c++", "g++"),
];

/// Vendor toolkit headers, as (header relative to the toolkit root,
/// package that owns it).
///
/// Each pair was read out of the vendor dev image with `dpkg -S` on the
/// `readlink -f`'d path, never guessed -- `/opt/rocm` is a versioned
/// symlink, so the naive lookup reports "not owned" and reads like the
/// file is unpackaged.
///
/// **These lists are deeper than the shim's own includes on purpose.**
/// Checking only the header flodl names is a false pass, twice proven:
/// `cuda_runtime.h` line 82 pulls `crt/host_config.h` from a different
/// package, and torch's hipified `ATen/hip` tree reaches the whole ROCm
/// math stack (`HIPContextLight.h` -> hipsparse). Both shipped a green
/// guard and a dead compile. Derived by grepping libtorch's own
/// `ATen/hip` + `c10/hip` trees for external includes.
///
/// No metapackage shortcut exists for ROCm: `rocm-dev` covers only
/// hip-dev and rocm-smi-lib of these nine.
pub const ROCM_HEADERS: &[(&str, &str)] = &[
    ("hip/hip_runtime.h", "hip-dev"),
    ("rccl/rccl.h", "rccl-dev"),
    ("hipblas/hipblas.h", "hipblas-dev"),
    ("hipblaslt/hipblaslt.h", "hipblaslt-dev"),
    ("hipcub/hipcub.hpp", "hipcub-dev"),
    ("hipsolver/hipsolver.h", "hipsolver-dev"),
    ("hipsparse/hipsparse.h", "hipsparse-dev"),
    ("rocblas/rocblas.h", "rocblas-dev"),
    ("rocm_smi/rocm_smi.h", "rocm-smi-lib"),
];

/// CUDA equivalent. The version placeholders are deliberate: the exact
/// package name carries the toolkit version and we do not know which
/// one the user wants.
pub const CUDA_HEADERS: &[(&str, &str)] = &[
    ("cuda_runtime.h", "cuda-cudart-dev-<M>-<m>"),
    ("crt/host_config.h", "cuda-crt-<M>-<m>"),
    ("nccl.h", "libnccl-dev"),
];

/// Host tools that are absent, as Debian package names.
pub fn missing_host_tools() -> Vec<&'static str> {
    HOST_TOOLS
        .iter()
        .filter(|(probe, _)| {
            if *probe == "curl" {
                // Either satisfies the download requirement.
                return !system::has_command("curl") && !system::has_command("wget");
            }
            !system::has_command(probe)
        })
        .map(|(_, pkg)| *pkg)
        .collect()
}

/// Standard include directories the compiler searches by default.
///
/// Load-bearing, not belt-and-braces: `nccl.h` ships in `libnccl-dev`
/// at **`/usr/include/nccl.h`**, NOT under `$CUDA_HOME`. Checking only
/// the toolkit root reported it missing on an image where the build
/// works, which is a false negative -- strictly worse than the gap the
/// check exists to close, because it breaks a working setup.
const SYSTEM_INCLUDE_DIRS: &[&str] = &["/usr/include", "/usr/local/include"];

/// Vendor headers that are absent, as (header, package) pairs.
///
/// A header counts as present if it is under `<root>/include` OR any
/// default system include dir, because that is what the compiler will
/// do. Header paths in the tables are relative to an include dir, with
/// no `include/` prefix, precisely so both can be searched.
///
/// Pure: the toolkit root is a parameter rather than an env read, so
/// every arm is testable without mutating process-global state. This
/// crate's test binary runs in parallel and an env-mutating test only
/// works if every reader takes the same lock, which they do not.
pub fn missing_headers<'a>(
    root: &Path,
    headers: &'a [(&'a str, &'a str)],
) -> Vec<&'a (&'a str, &'a str)> {
    headers
        .iter()
        .filter(|(h, _)| {
            if root.join("include").join(h).exists() {
                return false;
            }
            !SYSTEM_INCLUDE_DIRS
                .iter()
                .any(|d| Path::new(d).join(h).exists())
        })
        .collect()
}

/// De-duplicated package list for a set of missing headers.
pub fn packages_for(missing: &[&(&str, &str)]) -> Vec<String> {
    let mut pkgs: Vec<String> = missing.iter().map(|(_, p)| (*p).to_string()).collect();
    pkgs.dedup();
    pkgs
}

/// The install line for a package list, contextual to the platform.
///
/// Debian is spelled out because it is the platform the cloud hosts
/// use; the others get a direction rather than a fabricated command,
/// which is the honest thing when the package names are not verified.
pub fn install_hint(packages: &[String]) -> String {
    if packages.is_empty() {
        return String::new();
    }
    let list = packages.join(" ");
    if cfg!(target_os = "macos") {
        format!("brew install {list}   (names may differ on macOS)")
    } else if cfg!(target_os = "windows") {
        "no native Windows build is supported; use WSL2 \
         (https://flodl.dev/guide/windows-wsl)"
            .to_string()
    } else {
        format!("sudo apt install {list}   (or your distribution's equivalent)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A throwaway `<tmp>/include/` tree. Built here rather than
    /// pointed at a real toolkit: `libtorch/` is a gitignored download,
    /// so a test that depends on one passes locally and fails in CI.
    fn scratch_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("flodl-req-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(root.join("include/sub")).unwrap();
        std::fs::write(root.join("include/present.h"), "").unwrap();
        std::fs::write(root.join("include/sub/nested.h"), "").unwrap();
        root
    }

    #[test]
    fn missing_headers_lists_only_what_is_absent() {
        // One entry really exists, so the negative half is proven rather
        // than vacuously true: a check against a path that can never
        // exist is green for the wrong reason.
        let root = scratch_root("absent");
        let table: &[(&str, &str)] = &[
            ("present.h", "present-pkg"),
            ("sub/nested.h", "nested-pkg"),
            ("nope.h", "absent-pkg"),
        ];
        let missing = missing_headers(&root, table);
        assert_eq!(missing.len(), 1, "{missing:?}");
        assert_eq!(missing[0].1, "absent-pkg");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_system_header_counts_as_present() {
        // The regression that broke a working CUDA image: `nccl.h` ships
        // at /usr/include/nccl.h, not under $CUDA_HOME, so a toolkit-root
        // -only check called it missing and failed a build that compiles.
        let root = scratch_root("sys");
        let sys_header = SYSTEM_INCLUDE_DIRS
            .iter()
            .map(|d| Path::new(d).join("stdio.h"))
            .find(|p| p.exists());
        if let Some(h) = sys_header {
            let name = h.file_name().unwrap().to_str().unwrap();
            let table: &[(&str, &str)] = &[("stdio.h", "libc6-dev")];
            assert!(
                missing_headers(&root, table).is_empty(),
                "{name} is in a default include dir and must not be reported missing"
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn packages_are_deduplicated() {
        let a = ("h1", "pkg"); let b = ("h2", "pkg"); let c = ("h3", "other");
        let missing = vec![&a, &b, &c];
        assert_eq!(packages_for(&missing), vec!["pkg", "other"]);
    }

    #[test]
    fn every_rocm_header_names_a_package() {
        // Nine headers, nine owners -- if a pair ever loses its package
        // the message would tell a user to install "".
        assert_eq!(ROCM_HEADERS.len(), 9);
        for (h, p) in ROCM_HEADERS {
            assert!(!h.is_empty() && !p.is_empty(), "{h} -> {p}");
        }
        for (h, p) in CUDA_HEADERS {
            assert!(!h.is_empty() && !p.is_empty(), "{h} -> {p}");
        }
    }

    #[test]
    fn install_hint_is_empty_when_nothing_is_missing() {
        assert!(install_hint(&[]).is_empty());
    }

    #[test]
    fn install_hint_names_the_packages() {
        let h = install_hint(&["curl".into(), "g++".into()]);
        if cfg!(target_os = "windows") {
            assert!(h.contains("WSL2"), "{h}");
        } else {
            assert!(h.contains("curl") && h.contains("g++"), "{h}");
        }
    }
}
