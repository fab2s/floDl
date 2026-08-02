//! What a host needs before floDl will build, and the command that
//! supplies it.
//!
//! Each requirement is also checked at its point of use (`util/http.rs`
//! for curl, `util/archive.rs` for unzip, `flodl-sys/build.rs` for the
//! vendor headers). Those checks report one missing item at a time, at
//! the moment it is needed. This module exists so `fdl probe` and
//! `fdl setup` can report the whole set up front instead; the per-site
//! checks remain as the backstop when neither was run.
//!
//! The set is not absolute. A Docker build needs no C++ compiler and no
//! vendor headers, since both live in the image, so callers request the
//! set matching the build path in use.

use std::path::Path;

use crate::util::system;

/// Host tools `fdl` itself shells out to: (probe name, Debian package).
///
/// `curl` is special-cased by the caller: `util/http.rs` accepts wget
/// as well, so either satisfies the requirement.
const HOST_TOOLS: &[(&str, &str)] = &[
    ("curl", "curl"),
    ("unzip", "unzip"),
    ("c++", "g++"),
];

/// Vendor toolkit headers, as (header relative to an include dir,
/// package that owns it).
///
/// The set covers the shim's whole include chain, not the headers it
/// names directly: torch's vendor trees pull in more (`cuda_runtime.h`
/// includes `crt/host_config.h`, which a different package owns;
/// `ATen/hip` reaches hipsparse and hipblas). Checking only the direct
/// includes therefore passes while the compile still fails.
///
/// Regenerate after a libtorch bump by asking the compiler for the real
/// dependency set:
///
/// ```text
/// c++ -std=c++17 -M -I . -I <libtorch>/include \
///     -I <libtorch>/include/torch/csrc/api/include -I <toolkit>/include \
///     <the -D flags build.rs sets> shim.cpp
/// ```
///
/// Use `-M`, not `-MM`: `-MM` omits system headers, which drops
/// `nccl.h` (it lives in `/usr/include`, not under `$CUDA_HOME`).
/// Grepping the vendor tree instead over-reports, listing headers the
/// chain never reaches.
///
/// Map a header to its package inside the vendor dev image. `dpkg -S`
/// needs a `readlink -f`'d path under ROCm, since `/opt/rocm` is a
/// versioned symlink; the CUDA image needs the forward lookup
/// (`dpkg -L` per candidate package) instead.
///
/// ROCm has no metapackage covering these: `rocm-dev` supplies only
/// `hip-dev`.
pub const ROCM_HEADERS: &[(&str, &str)] = &[
    ("hip/hip_runtime.h", "hip-dev"),
    ("rccl/rccl.h", "rccl-dev"),
    ("hipblas/hipblas.h", "hipblas-dev"),
    ("hipblas-common/hipblas-common.h", "hipblas-common-dev"),
    ("hipblaslt/hipblaslt.h", "hipblaslt-dev"),
    ("hipsolver/hipsolver.h", "hipsolver-dev"),
    ("hipsparse/hipsparse.h", "hipsparse-dev"),
];

/// CUDA equivalent. The version placeholders are deliberate: the exact
/// package name carries the toolkit version and we do not know which
/// one the user wants.
pub const CUDA_HEADERS: &[(&str, &str)] = &[
    ("cuda_runtime.h", "cuda-cudart-dev-<M>-<m>"),
    ("crt/host_config.h", "cuda-crt-<M>-<m>"),
    ("cublas_v2.h", "libcublas-dev-<M>-<m>"),
    ("cusolverDn.h", "libcusolver-dev-<M>-<m>"),
    ("cusparse.h", "libcusparse-dev-<M>-<m>"),
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
/// Required, not defensive: some vendor headers install outside the
/// toolkit root. `nccl.h` ships in `libnccl-dev` at `/usr/include`, so
/// a toolkit-root-only check reports it missing on hosts where the
/// build succeeds.
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
        assert_eq!(ROCM_HEADERS.len(), 7);
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
