//! Execution context: project-local vs global (~/.flodl).
//!
//! When running inside a floDl project (detected by walking up from cwd),
//! libtorch operations target `./libtorch/`. When standalone, they target
//! `~/.flodl/libtorch/`.

use std::env;
use std::fs;
use std::path::PathBuf;

pub struct Context {
    /// Root directory for libtorch storage. Either a project root or ~/.flodl.
    pub root: PathBuf,
    /// Whether we are operating inside a detected project.
    pub is_project: bool,
}

impl Context {
    /// Auto-detect: walk up from cwd looking for a flodl project, fall back
    /// to `~/.flodl/`.
    pub fn resolve() -> Self {
        if let Some(project_root) = find_project_root() {
            Context {
                root: project_root,
                is_project: true,
            }
        } else {
            Context {
                root: global_root(),
                is_project: false,
            }
        }
    }

    /// The global root, whether or not cwd sits inside a project:
    /// `$FLODL_HOME`, else `~/.flodl`.
    ///
    /// [`Context::resolve`] prefers a project root, which is right for a
    /// developer managing their checkout's libtorch and wrong for a box
    /// that is only *consuming* artifacts: a dial-in worker's project
    /// root is frequently a read-only shared mount, so its acquisitions
    /// (libtorch, the fetched source tree) belong under the root fdl
    /// manages on that one box. The two are named apart on purpose.
    pub fn global() -> Self {
        Context {
            root: global_root(),
            is_project: false,
        }
    }

    /// Force a specific root (for --path overrides).
    pub fn with_root(root: PathBuf) -> Self {
        let is_project = root.join("Cargo.toml").exists();
        Context { root, is_project }
    }

    /// The libtorch directory under this root.
    pub fn libtorch_dir(&self) -> PathBuf {
        self.root.join("libtorch")
    }

    /// Print a short context line (for diagnostics).
    pub fn label(&self) -> String {
        if self.is_project {
            format!("project ({})", self.root.display())
        } else {
            format!("global ({})", self.root.display())
        }
    }
}

/// Walk up from cwd looking for a flodl project.
fn find_project_root() -> Option<PathBuf> {
    let mut dir = env::current_dir().ok()?;
    loop {
        // Strongest signal: libtorch/.active exists here
        if dir.join("libtorch/.active").exists() {
            return Some(dir);
        }
        // Secondary signal: Cargo.toml mentioning flodl
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists()
            && let Ok(contents) = fs::read_to_string(&cargo_toml)
            && contents.contains("flodl")
        {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Global root: $FLODL_HOME or ~/.flodl/
fn global_root() -> PathBuf {
    env::var("FLODL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir().join(".flodl"))
}

pub(crate) fn home_dir() -> PathBuf {
    env::var("HOME")
        .or_else(|_| env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}
