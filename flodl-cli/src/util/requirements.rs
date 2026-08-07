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
    let root_include = root.join("include");
    headers
        .iter()
        .filter(|(h, _)| {
            if root_include.join(h).exists() {
                return false;
            }
            if SYSTEM_INCLUDE_DIRS
                .iter()
                .any(|d| Path::new(d).join(h).exists())
            {
                return false;
            }
            // The path scan came up empty, which is exactly when its
            // three-directory view is worth doubting. Ask the compiler
            // that will do the build, with the same include dir it will
            // get, before reporting a gap. Unreachable OR unanswerable
            // (no compiler) both leave it reported.
            !matches!(header_reachable(h, &[root_include.as_path()]), Some(true))
        })
        .collect()
}

/// Whether the C++ compiler can actually resolve `#include <header>`.
///
/// The path scan above knows three directories. The compiler knows every
/// rule the real build obeys — its own defaults, `CPATH`, multiarch
/// directories, spec files, whatever a distro did — so it is the second
/// opinion worth having before telling someone to install something they
/// already have. Pascal is the case in point: its CUDA headers live in
/// `/usr/include` with no `/usr/local/cuda` at all, and it compiles
/// `flodl-sys --features cuda` in 12s.
///
/// `None` when there is no compiler to ask, which is not the same answer
/// as "missing" and must not be collapsed into one: a box without a C++
/// compiler has a different problem, and [`missing_host_tools`] reports
/// it.
///
/// Cost is one preprocessor invocation, ~30ms, and it is paid only for a
/// header the path scan already failed to find — the happy path spawns
/// nothing.
pub fn header_reachable(header: &str, include_dirs: &[&Path]) -> Option<bool> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    if !system::has_command(&cxx) {
        return None;
    }
    let mut cmd = Command::new(&cxx);
    for dir in include_dirs {
        cmd.arg("-I").arg(dir);
    }
    // Preprocess only: resolving the include is the whole question, and
    // -fsyntax-only would drag in a parse we do not need. The output goes
    // nowhere via `Stdio::null()` rather than `-o /dev/null`, which is not
    // a path on Windows: gcc there tries to create a `\dev\` directory,
    // fails, and the non-zero exit reads as "header missing" for every
    // header on the box.
    cmd.args(["-E", "-x", "c++", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = cmd.spawn().ok()?;
    child
        .stdin
        .as_mut()?
        .write_all(format!("#include <{header}>\n").as_bytes())
        .ok()?;
    Some(child.wait().ok()?.success())
}

/// De-duplicated package list for a set of missing headers, in table
/// order.
///
/// `Vec::dedup` is not enough: it only collapses *adjacent* duplicates,
/// so two headers owned by one package would list it twice unless they
/// happened to sit next to each other in the table.
pub fn packages_for(missing: &[&(&str, &str)]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    missing
        .iter()
        .filter(|(_, p)| seen.insert(*p))
        .map(|(_, p)| (*p).to_string())
        .collect()
}

/// A vendor toolkit gap on THIS box: the headers a `--features
/// <vendor>` compile will not find, and the line that installs them.
#[derive(Debug)]
pub struct ToolkitGap {
    /// Toolkit root the check ran against.
    pub root: std::path::PathBuf,
    /// Missing headers, as printed to the operator.
    pub headers: Vec<String>,
    /// The full install line ([`install_hint`] over the owning
    /// packages; the NVIDIA arm uses the `cuda-toolkit` metapackage
    /// since the per-header names carry version placeholders).
    pub install: String,
}

/// Resolve the toolkit gap for a vendor, `None` when the headers are
/// all present — or when the vendor has no known toolkit layout, since
/// guessing one produces a confidently wrong apt command.
///
/// The ROCm root comes from `flodl-hw`'s resolution (env chain +
/// convention, runtime-verified) so this check and the loader path
/// cannot disagree about where ROCm lives.
pub fn toolkit_gap(vendor: flodl_hw::GpuVendor) -> Option<ToolkitGap> {
    use std::path::PathBuf;
    let (root, headers, metapackages): (PathBuf, _, Option<&[&str]>) = match vendor {
        flodl_hw::GpuVendor::Amd => (
            flodl_hw::rocm_runtime_root()
                .or_else(|| std::env::var("ROCM_PATH").ok().map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("/opt/rocm")),
            ROCM_HEADERS,
            None,
        ),
        flodl_hw::GpuVendor::Nvidia => (
            PathBuf::from(
                std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string()),
            ),
            CUDA_HEADERS,
            Some(&["cuda-toolkit", "libnccl-dev"]),
        ),
        _ => return None,
    };
    let missing = missing_headers(&root, headers);
    if missing.is_empty() {
        return None;
    }
    let packages: Vec<String> = match metapackages {
        Some(m) => m.iter().map(|p| p.to_string()).collect(),
        None => packages_for(&missing),
    };
    Some(ToolkitGap {
        root,
        headers: missing.iter().map(|(h, _)| h.to_string()).collect(),
        install: install_hint(&packages),
    })
}

/// Debian package name in the RHEL-family spelling.
///
/// Both vendors ship the same packages to their Debian and RHEL repos
/// with identical stems and two dev-suffix conventions, so this is a
/// transform rather than a second table -- every result was verified by
/// repoquery against the cuda-rhel9 and rocm rhel9 repositories.
/// `g++` is the one host tool whose rpm goes by a different name.
/// Kept in sync by hand with `flodl-sys/build.rs`'s `rpm` closure.
pub fn rpm_name(deb: &str) -> String {
    if deb == "g++" {
        return "gcc-c++".to_string();
    }
    match deb.strip_suffix("-dev") {
        Some(stem) => format!("{stem}-devel"),
        None => deb.replace("-dev-", "-devel-"),
    }
}

/// The install line for a package list, contextual to the platform.
///
/// Debian and RHEL-family are spelled out because those are the
/// platforms cloud hosts use (package names verified on both); the
/// others get a direction rather than a fabricated command, which is
/// the honest thing when the names are not verified.
pub fn install_hint(packages: &[String]) -> String {
    if packages.is_empty() {
        return String::new();
    }
    if cfg!(target_os = "macos") {
        format!(
            "brew install {}   (names may differ on macOS)",
            packages.join(" ")
        )
    } else if cfg!(target_os = "windows") {
        "no native Windows build is supported; use WSL2 \
         (https://flodl.dev/guide/windows-wsl)"
            .to_string()
    } else if crate::util::platform::Platform::detect()
        == crate::util::platform::Platform::Rhel
    {
        let list = packages.iter().map(|p| rpm_name(p)).collect::<Vec<_>>();
        format!(
            "sudo dnf install {}   (or your distribution's equivalent)",
            list.join(" ")
        )
    } else {
        format!(
            "sudo apt install {}   (or your distribution's equivalent)",
            packages.join(" ")
        )
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
        // A toolkit-root-only check reports headers that install outside
        // it as missing, failing a build that in fact compiles: `nccl.h`
        // ships at /usr/include/nccl.h, not under $CUDA_HOME.
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
        // The duplicates are deliberately non-adjacent: adjacent ones
        // pass under a plain `Vec::dedup`, which does not dedup a table
        // where one package owns two headers listed apart.
        let a = ("h1", "pkg");
        let b = ("h2", "other");
        let c = ("h3", "pkg");
        let missing = vec![&a, &b, &c];
        assert_eq!(packages_for(&missing), vec!["pkg", "other"]);
    }

    #[test]
    fn every_header_names_a_package() {
        // A pair that loses its package would tell the user to install "".
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
            // The compiler package is spelled per family (g++ on
            // Debian, gcc-c++ on RHEL), so assert either.
            assert!(h.contains("curl"), "{h}");
            assert!(h.contains("g++") || h.contains("gcc-c++"), "{h}");
        }
    }

    #[test]
    fn the_compiler_answers_for_headers_the_path_scan_cannot_see() {
        // A header every C++ toolchain resolves, in no directory this
        // module lists: only the compiler's own view finds it.
        match header_reachable("cstdio", &[]) {
            Some(true) => {}
            Some(false) => panic!("the compiler could not resolve <cstdio>"),
            // No compiler here: unanswerable is a distinct third state
            // and must not be read as present.
            None => {}
        }
        // And it says no to something that does not exist, rather than
        // waving everything through.
        if header_reachable("cstdio", &[]) == Some(true) {
            assert_eq!(
                header_reachable("flodl_no_such_header_42.h", &[]),
                Some(false),
            );
        }
    }

    #[test]
    fn a_header_outside_the_toolkit_root_is_not_reported_missing() {
        // The pascal shape: nothing under the toolkit root, but the
        // compiler resolves the header anyway (there, CUDA lives in
        // /usr/include). Reporting that as a gap tells the operator to
        // install what they already have.
        let root = scratch_root("reach");
        let table: &[(&str, &str)] = &[("cstdio", "libstdc++-dev")];
        let missing = missing_headers(&root, table);
        if header_reachable("cstdio", &[]) == Some(true) {
            assert!(missing.is_empty(), "compiler-visible header reported missing");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rpm_names_match_the_verified_rhel_spellings() {
        // Every pair was checked by repoquery against the vendors'
        // rhel9 repositories; unversioned names pass through untouched.
        for (deb, rpm) in [
            ("hip-dev", "hip-devel"),
            ("rccl-dev", "rccl-devel"),
            ("hipblas-common-dev", "hipblas-common-devel"),
            ("hipblaslt-dev", "hipblaslt-devel"),
            ("cuda-cudart-dev-<M>-<m>", "cuda-cudart-devel-<M>-<m>"),
            ("libcublas-dev-<M>-<m>", "libcublas-devel-<M>-<m>"),
            ("libnccl-dev", "libnccl-devel"),
            ("cuda-crt-<M>-<m>", "cuda-crt-<M>-<m>"),
            ("cuda-toolkit", "cuda-toolkit"),
            ("g++", "gcc-c++"),
            ("curl", "curl"),
            ("unzip", "unzip"),
        ] {
            assert_eq!(rpm_name(deb), rpm, "{deb}");
        }
    }

}
