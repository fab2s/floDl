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

use crate::util::platform::Platform;
use crate::util::system;

/// Host tools `fdl` itself shells out to: (probe name, Debian package).
///
/// `curl` is special-cased by the caller: `util/http.rs` accepts wget
/// as well, so either satisfies the requirement.
const HOST_TOOLS: &[(&str, &str)] = &[("curl", "curl"), ("unzip", "unzip"), ("c++", "g++")];

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
    match deb {
        "g++" => return "gcc-c++".to_string(),
        // EPEL carries it under the fuse- prefix (verified by repoquery
        // on EL9); a bare `sshfs` matches nothing there.
        "sshfs" => return "fuse-sshfs".to_string(),
        _ => {}
    }
    match deb.strip_suffix("-dev") {
        Some(stem) => format!("{stem}-devel"),
        None => deb.replace("-dev-", "-devel-"),
    }
}

/// The install line for a package list, contextual to the platform.
///
/// The one place a package-manager spelling is decided: every "X is not
/// installed" message routes through here, so a message written on an
/// apt box does not tell a dnf box the wrong command. Debian and
/// RHEL-family are spelled out because those are the platforms cloud
/// hosts use (package names verified on both, via [`rpm_name`]); the
/// others get a direction rather than a fabricated command, which is
/// the honest thing when the names are not verified.
pub fn install_hint(packages: &[String]) -> String {
    if packages.is_empty() {
        return String::new();
    }
    if cfg!(target_os = "windows") {
        return "no native Windows build is supported; use WSL2 \
                (https://flodl.dev/guide/windows-wsl)"
            .to_string();
    }
    let plat = Platform::detect();
    let names: Vec<String> = match plat {
        Platform::Rhel => packages.iter().map(|p| rpm_name(p)).collect(),
        _ => packages.to_vec(),
    };
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    match plat.install(&refs) {
        Some(cmd) if plat == Platform::MacOs => format!("{cmd}   (names may differ on macOS)"),
        Some(cmd) => format!("{cmd}   (or your distribution's equivalent)"),
        None => format!("install {} with your package manager", names.join(" ")),
    }
}

/// A tool a walk-in reaches for and does not have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolGap {
    /// Command probed on PATH.
    pub tool: &'static str,
    /// The `join:` field that reaches for it, when one does. `None` is
    /// the advisory case: nothing in this box's config names the tool,
    /// so `fdl join` will not miss it unless the config grows a use.
    pub needed_by: Option<&'static str>,
    /// The install line for this platform.
    pub install: String,
}

/// The tools a `join:` block's transports reach for, checked against
/// PATH: `data_source: sshfs://` needs sshfs, `source.from: rsync://`
/// needs rsync, `source.from: git+...` needs git.
///
/// A tool the block names is required: `fdl join` refuses the box
/// without it, permanently. The two door tools (sshfs, rsync) are also
/// reported when the block does NOT name them, as advisory gaps: a box
/// that mounts its data root at provisioning time and runs a declared
/// `bin:` needs neither, and the operator is trusted to know which
/// modes the box will be asked for. The toolchain is out of scope here:
/// `fdl join` names rustup itself, since a distribution's cargo is not
/// the answer.
pub fn walkin_tool_gaps(join: &crate::config::WorkerJoin) -> Vec<ToolGap> {
    walkin_tool_gaps_with(join, system::has_command)
}

/// [`walkin_tool_gaps`] with the PATH probe injected, so the table is
/// testable from a host that has everything installed.
pub fn walkin_tool_gaps_with(
    join: &crate::config::WorkerJoin,
    has: impl Fn(&str) -> bool,
) -> Vec<ToolGap> {
    use crate::spec::split_scheme;
    let data_scheme = join.data_source.as_deref().map(|s| split_scheme(s).0);
    let source_scheme = join.source.as_ref().map(|s| split_scheme(&s.from).0);
    let named = |tool: &str| -> Option<&'static str> {
        match tool {
            "sshfs" if data_scheme == Some(Some("sshfs")) => Some("data_source: sshfs://"),
            "rsync" if source_scheme == Some(Some("rsync")) => Some("source.from: rsync://"),
            "git" if matches!(source_scheme, Some(Some(s)) if s.starts_with("git+")) => {
                Some("source.from: git+")
            }
            _ => None,
        }
    };
    ["sshfs", "rsync", "git"]
        .into_iter()
        .filter(|tool| !has(tool))
        .filter_map(|tool| {
            let needed_by = named(tool);
            // git only matters when a spec names it: it is not a door
            // tool, and most boxes have it for unrelated reasons.
            if tool == "git" && needed_by.is_none() {
                return None;
            }
            Some(ToolGap {
                tool,
                needed_by,
                install: install_hint(&[tool.to_string()]),
            })
        })
        .collect()
}

/// One line per gap, in the severity the caller asked for: a named
/// tool reads as a refusal `fdl join` will issue, an advisory one as
/// what the box cannot do until it is installed.
pub fn tool_gap_message(gap: &ToolGap) -> String {
    match gap.needed_by {
        Some(field) => format!(
            "`{field}` in this box's join: block needs {tool}, which is not installed; \
             `fdl join` will refuse to dial. {install}",
            tool = gap.tool,
            install = gap.install,
        ),
        None => {
            let does = match gap.tool {
                "sshfs" => "mount a `data_source: sshfs://` root",
                _ => "pull a `source.from: rsync://` tree",
            };
            format!(
                "{tool} is not installed, so this box cannot {does}. Fine if it is \
                 never asked to; otherwise {install}",
                tool = gap.tool,
                install = gap.install,
            )
        }
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

    fn join(data_source: Option<&str>, source: Option<&str>) -> crate::config::WorkerJoin {
        crate::config::WorkerJoin {
            data_source: data_source.map(str::to_string),
            source: source.map(|from| crate::config::WorkerSource {
                from: from.to_string(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn walkin_gaps_are_required_when_the_block_names_the_transport() {
        let nothing = |_: &str| false;
        let gaps = walkin_tool_gaps_with(
            &join(
                Some("sshfs://op@ctrl:/flodl/data"),
                Some("rsync://op@ctrl:/tree"),
            ),
            nothing,
        );
        let by_tool = |t: &str| gaps.iter().find(|g| g.tool == t).cloned();
        assert_eq!(
            by_tool("sshfs").unwrap().needed_by,
            Some("data_source: sshfs://")
        );
        assert_eq!(
            by_tool("rsync").unwrap().needed_by,
            Some("source.from: rsync://")
        );
        // git is not named by an rsync source, and it is not a door tool.
        assert!(by_tool("git").is_none(), "{gaps:?}");

        let gaps = walkin_tool_gaps_with(&join(None, Some("git+https://h/o/r#v1")), nothing);
        let git = gaps.iter().find(|g| g.tool == "git").unwrap();
        assert_eq!(git.needed_by, Some("source.from: git+"));
        let msg = tool_gap_message(git);
        assert!(msg.contains("refuse to dial"), "{msg}");
        assert!(msg.contains(&install_hint(&["git".to_string()])), "{msg}");
    }

    #[test]
    fn walkin_door_tools_are_advisory_when_unnamed_and_silent_when_present() {
        // A bare data_path and a declared bin name no transport: the two
        // door tools are still reported, softly, and git is not.
        let gaps = walkin_tool_gaps_with(&join(None, None), |_| false);
        let tools: Vec<_> = gaps.iter().map(|g| g.tool).collect();
        assert_eq!(tools, ["sshfs", "rsync"]);
        assert!(gaps.iter().all(|g| g.needed_by.is_none()), "{gaps:?}");
        let msg = tool_gap_message(&gaps[0]);
        assert!(msg.contains("Fine if it is never asked to"), "{msg}");
        assert!(!msg.contains("refuse"), "{msg}");

        // Everything installed: nothing to say, named or not.
        let gaps =
            walkin_tool_gaps_with(&join(Some("sshfs://h:/d"), Some("rsync://h:/t")), |_| true);
        assert!(gaps.is_empty(), "{gaps:?}");
    }

    #[test]
    fn install_hint_speaks_this_platforms_manager() {
        // Whatever the family, the hint is the Platform decider's line
        // (plus a caveat), never a spelling of its own, which is the
        // whole point of having one.
        let plat = Platform::detect();
        let hint = install_hint(&["rsync".to_string()]);
        match plat.install(&["rsync"]) {
            Some(cmd) if !cfg!(target_os = "windows") => assert!(hint.starts_with(&cmd), "{hint}"),
            _ => assert!(!hint.contains("sudo"), "{hint}"),
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
            assert!(
                missing.is_empty(),
                "compiler-visible header reported missing"
            );
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
            ("sshfs", "fuse-sshfs"),
            ("rsync", "rsync"),
        ] {
            assert_eq!(rpm_name(deb), rpm, "{deb}");
        }
    }
}
