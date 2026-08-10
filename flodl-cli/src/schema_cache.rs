//! `--fdl-schema` binary contract: probe, validate, and cache.
//!
//! A sub-command binary that opts into the contract exposes a single
//! `--fdl-schema` flag printing a JSON schema describing its CLI surface.
//! `flodl-cli` caches the output under `<cmd_dir>/.fdl/schema-cache/<cmd>.json`
//! and prefers it over any inline YAML schema declared in `fdl.yaml`.
//!
//! **Cargo entries** (`entry: cargo run ...`) are probed only when the command
//! opts in with `compile: true` *and* the caller asked for an authoritative
//! surface (see [`crate::config::ProbeCost`]) — invoking one forces a full
//! compile, which is unacceptable latency for `fdl --help` or a tab-press.
//! Without the opt-in, users run `fdl <cmd> --refresh-schema` explicitly after
//! a build.
//!
//! Cache invalidation is mtime-based: the cache file's mtime is compared
//! against every path that could change the schema — the command's config
//! file AND, for a binary that declares its own surface, the sources that
//! surface is compiled from (see
//! [`schema_source_refs`](crate::schema_cache::schema_source_refs)). A cache
//! older than any of them is stale. Users can also force-refresh.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::SystemTime;

use crate::config::{self, Schema};

/// Directory where all schema caches live, relative to the command dir.
const CACHE_DIR: &str = ".fdl/schema-cache";

/// Records which `fdl` wrote the caches in a directory.
///
/// The mtime references answer "did the command change"; they cannot answer
/// "did the SCHEMA FORMAT change", and that is a real event: the day
/// `--fdl-schema` began emitting block-structured descriptions (paragraphs and
/// list items preserved rather than flattened), every existing cache held the
/// old flat text and no reference file had been touched. Help would have kept
/// rendering the pre-upgrade shape until someone happened to edit their crate
/// -- the same silently-stale failure the mtime references were added to fix,
/// one level up.
pub(crate) const CACHE_STAMP: &str = ".fdl-version";

/// Directories never worth walking when collecting schema sources: build
/// output, caches, and data. Skipping `target` is the one that matters —
/// it dwarfs the source tree.
const SOURCE_SKIP_DIRS: &[&str] = &[
    "target",
    ".fdl",
    ".git",
    "node_modules",
    "runs",
    "data",
    "baselines",
    "libtorch",
    ".cargo",
];

/// Hard ceiling on files examined. `fdl <cmd> -h` must never be slow, so a
/// pathological tree costs a bounded scan and then gives up — degrading to
/// today's config-only invalidation rather than stalling help.
const MAX_SOURCE_REFS: usize = 4096;

/// Every file whose edit could change a command's `--fdl-schema` output.
///
/// The schema of a cargo entry is *compiled from* the crate's Rust sources, so
/// watching only `fdl.yml` (as this did originally) meant editing a CLI struct
/// left a stale cache with no signal at all: `-h` kept rendering the previous
/// surface until someone touched the yml or deleted the cache by hand.
///
/// Deliberately coarse — every `.rs` plus `Cargo.toml` under the command dir,
/// not an attempt to find the files that *define* the schema. Over-watching
/// costs one extra probe, exactly what editing the yml already costs.
/// Under-watching is the bug being fixed, and a precise scan would reintroduce
/// it the moment a `#[derive(FdlArgs)]` struct referenced a constant from
/// another module. Measured at 23 files / 0.12 ms for `ddp-bench`, against a
/// probe that spins a container and runs cargo — the precision is not worth
/// buying.
///
/// Scoped to the command's OWN directory on purpose. Following its dependency
/// crates would invalidate the cache on every edit anywhere in the workspace,
/// which in a repo whose library changes constantly means compiling on nearly
/// every `-h` — the cost this cache exists to avoid.
pub fn schema_source_refs(cmd_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![cmd_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if out.len() >= MAX_SOURCE_REFS {
            break;
        }
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            let Ok(ft) = entry.file_type() else { continue };
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if ft.is_dir() {
                if !name.starts_with('.') && !SOURCE_SKIP_DIRS.contains(&name.as_ref()) {
                    stack.push(path);
                }
            } else if name.ends_with(".rs") || name == "Cargo.toml" {
                out.push(path);
            }
        }
    }
    out
}

/// Resolve the cache file path for a given command dir and name.
pub fn cache_path(cmd_dir: &Path, cmd_name: &str) -> PathBuf {
    cmd_dir.join(CACHE_DIR).join(format!("{cmd_name}.json"))
}

/// Read a schema cache file, returning `Some` only if it parses cleanly
/// and survives validation. Parse or validation errors are treated as
/// "no cache" (the caller falls through to the inline/YAML schema).
pub fn read_cache(path: &Path) -> Option<Schema> {
    let content = fs::read_to_string(path).ok()?;
    let schema: Schema = serde_json::from_str(&content).ok()?;
    config::validate_schema(&schema).ok()?;
    Some(schema)
}

/// Consider a cache "stale" if it is older than the command's fdl.yml
/// (config changes), or older than a sentinel binary path when supplied.
///
/// Missing cache ⇒ stale (return true). Missing reference mtime ⇒ treat
/// the cache as fresh (conservative: don't refresh what we can't justify).
pub fn is_stale(cache: &Path, reference_mtimes: &[PathBuf]) -> bool {
    let Some(cache_mtime) = mtime(cache) else {
        return true;
    };
    if !stamp_matches(cache) {
        return true;
    }
    reference_mtimes
        .iter()
        .filter_map(|p| mtime(p))
        .any(|ref_m| ref_m > cache_mtime)
}

/// Whether the cache dir was written by this `fdl`. See [`CACHE_STAMP`].
///
/// A missing stamp counts as a mismatch: caches predating the stamp were
/// written by an older format by definition.
fn stamp_matches(cache: &Path) -> bool {
    let Some(dir) = cache.parent() else {
        return false;
    };
    fs::read_to_string(dir.join(CACHE_STAMP))
        .map(|s| s.trim() == env!("CARGO_PKG_VERSION"))
        .unwrap_or(false)
}

fn mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok()?.modified().ok()
}

/// Serialize a schema to the cache file, creating parent dirs as needed.
pub fn write_cache(path: &Path, schema: &Schema) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {}", parent.display(), e))?;
        // Best-effort: a missing stamp only costs one extra probe.
        let _ = fs::write(parent.join(CACHE_STAMP), env!("CARGO_PKG_VERSION"));
    }
    let json =
        serde_json::to_string_pretty(schema).map_err(|e| format!("schema serialize: {e}"))?;
    fs::write(path, json).map_err(|e| format!("cannot write {}: {}", path.display(), e))
}

/// Probe a binary for its schema by running `<entry> --fdl-schema` via the
/// shell and parsing stdout as JSON.
///
/// `cmd_dir` is the directory containing the `fdl.yml` that declared the
/// entry — it serves as the cwd for the shell unless the entry is wrapped
/// through docker (then the wrap walks up to the nearest
/// `docker-compose.yml` for compose's cwd).
///
/// `docker_service` carries the `docker:` field from the resolved
/// command config. When set AND we're not already inside a container,
/// the invocation is wrapped as
/// `docker compose run --rm <service> bash -c '<entry> --fdl-schema'`
/// so cargo entries that need libtorch get probed inside the dev
/// container instead of failing silently on the host. When unset, the
/// entry runs directly on the host.
///
/// The wrap must reproduce what the *exec* path does, not merely resemble
/// it, and two pieces of that are easy to leave out:
///
/// - `docker:` is resolved through [`crate::run::resolve_docker_service`],
///   because `gpu` is a LOGICAL service picked from the active libtorch
///   variant. Handing compose the literal string yields `no such service:
///   gpu` — and since probe errors are a documented fall-through, the only
///   symptom is a `--help` that quietly stops listing options.
/// - The libtorch env vars go on the child, because they are what
///   `--features "$FDL_GPU_FEATURE"` and compose's own
///   `${LIBTORCH_HOST_PATH}` / `${CUDA_TAG}` interpolations read. Without
///   them the probe either builds the wrong feature set or mounts the
///   wrong variant, both of which look like a broken binary rather than a
///   missing export.
///
/// On failure returns a string error rather than panicking — callers
/// almost always want to fall back to the inline schema (or none).
pub fn probe(entry: &str, cmd_dir: &Path, docker_service: Option<&str>) -> Result<Schema, String> {
    if entry.trim().is_empty() {
        return Err("entry is empty".into());
    }

    let inner = format!("{entry} --fdl-schema");
    let (invocation, run_cwd) = match docker_service {
        Some(svc) if !inside_docker() => {
            let compose_root = find_docker_compose_root(cmd_dir).ok_or_else(|| {
                format!(
                    "cannot probe schema: docker:{svc} declared but no \
                     docker-compose.yml found above {}",
                    cmd_dir.display()
                )
            })?;
            let wrapped = docker_invocation(&inner, cmd_dir, svc, &compose_root);
            (wrapped, compose_root)
        }
        _ => (inner, cmd_dir.to_path_buf()),
    };

    let (shell, flag) = if cfg!(target_os = "windows") {
        ("cmd", "/C")
    } else {
        ("sh", "-c")
    };
    let mut cmd = Command::new(shell);
    cmd.args([flag, &invocation])
        .current_dir(&run_cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // `run_cwd` is the compose root when wrapped and the command dir
    // otherwise; the env is anchored at the project root either way, which
    // for the wrapped case is the same directory. Errors are ignored on
    // purpose: a project with no resolvable libtorch still probes fine (a
    // CPU-only or script entry needs none of these), and the probe's own
    // failure is the better diagnostic if it does need them.
    let env_root = match crate::config::find_project_config(&run_cwd) {
        Some(cfg) => cfg.parent().unwrap_or(&run_cwd).to_path_buf(),
        None => run_cwd.clone(),
    };
    if let Ok(env_vars) = crate::run::libtorch_env(&env_root) {
        for (key, val) in &env_vars {
            cmd.env(key, val);
        }
    }
    let output = cmd
        .output()
        .map_err(|e| format!("spawn `{invocation}`: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "`{invocation}` exited with {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    // Tolerate leading lines of cargo chatter by locating the first `{`.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let start = stdout
        .find('{')
        .ok_or_else(|| "no JSON object in --fdl-schema output".to_string())?;
    let schema: Schema = serde_json::from_str(&stdout[start..])
        .map_err(|e| format!("--fdl-schema did not emit valid JSON: {e}"))?;
    config::validate_schema(&schema)
        .map_err(|e| format!("--fdl-schema output failed validation: {e}"))?;
    Ok(schema)
}

/// Wrap a probe command for `docker compose run`.
///
/// Split out of [`probe`] so the service-resolution step can be asserted
/// without spawning a container — the regression it guards (a literal `gpu`
/// reaching compose) is invisible from the outside, since a failed probe is a
/// documented fall-through and the only symptom was a `--help` that stopped
/// listing options.
fn docker_invocation(inner: &str, cmd_dir: &Path, svc: &str, compose_root: &Path) -> String {
    // `gpu` is logical, standing for whichever GPU container matches the
    // active libtorch variant; compose knows only the concrete names.
    let svc = crate::run::resolve_docker_service(svc, compose_root);
    // The container starts in its configured workdir (the compose root's
    // mount), NOT the command dir — without a `cd`, a cargo entry builds and
    // probes the WORKSPACE default binary instead of the command's (observed:
    // `fdl ddp-bench --refresh-schema` probing `fdl` itself, which rejects
    // `--fdl-schema`). The command dir's path relative to the compose root is
    // the same on both sides of the mount, so prefix the entry with a
    // relative cd.
    let inner_in_container = match cmd_dir
        .strip_prefix(compose_root)
        .ok()
        .filter(|rel| !rel.as_os_str().is_empty())
    {
        Some(rel) => format!("cd {} && {inner}", posix_quote(&rel.to_string_lossy())),
        None => inner.to_string(),
    };
    format!(
        "docker compose run --rm {svc} bash -c {}",
        posix_quote(&inner_in_container)
    )
}

/// Heuristic: cargo entries compile-on-run, which is what makes them the
/// expensive class. They are probed only under `compile: true` plus a
/// [`ProbeCost::Compile`](crate::config::ProbeCost::Compile) load; otherwise
/// probing is explicit (`fdl <cmd> --refresh-schema`).
pub fn is_cargo_entry(entry: &str) -> bool {
    entry.trim_start().starts_with("cargo ")
}

/// True when this process is running inside a Docker container. Mirrors
/// the `/.dockerenv` heuristic used elsewhere in the crate.
fn inside_docker() -> bool {
    Path::new("/.dockerenv").exists()
}

/// Climb from `start` looking for a directory containing
/// `docker-compose.yml` (the compose root used as cwd for `docker
/// compose` invocations). Returns `None` if none is found before
/// hitting the filesystem root.
fn find_docker_compose_root(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        if dir.join("docker-compose.yml").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

use crate::util::shell::posix_quote;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::Write;

    /// Scoped test directory under `std::env::temp_dir()` that cleans up on drop.
    /// Zero-external-dep replacement for `tempfile::tempdir()`.
    struct TestDir {
        path: PathBuf,
    }

    /// Uniqueness within the process comes from a counter, not from the
    /// clock. `SystemTime::now` is quantised to whatever tick the platform
    /// keeps, which `as_nanos` does not tell you, and nine of these tests
    /// share the `sc` tag and start within the same millisecond: two that
    /// land on one tick get one directory, and the first to finish deletes
    /// the other's script out from under a running `sh`. Seen once on the
    /// macOS runner as an exit-127 `bad.sh: No such file or directory`,
    /// where the timestamp was the only thing keeping the paths apart.
    static TEST_DIR_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    impl TestDir {
        fn new(tag: &str) -> Self {
            let n = TEST_DIR_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let pid = std::process::id();
            let path = std::env::temp_dir().join(format!("fdl-test-{tag}-{pid}-{n}"));
            fs::create_dir_all(&path).expect("create test dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn minimal_schema() -> Schema {
        let mut options = BTreeMap::new();
        options.insert(
            "model".into(),
            config::OptionSpec {
                ty: "string".into(),
                description: Some("pick a model".into()),
                default: Some(serde_json::json!("mlp")),
                choices: Some(vec![serde_json::json!("mlp"), serde_json::json!("resnet")]),
                short: Some("m".into()),
                env: None,
                completer: None,
            },
        );
        Schema {
            args: Vec::new(),
            options,
            strict: false,
            ..Schema::default()
        }
    }

    #[test]
    fn cache_roundtrip_preserves_schema() {
        let tmp = TestDir::new("sc");
        let path = cache_path(tmp.path(), "ddp-bench");
        let schema = minimal_schema();
        write_cache(&path, &schema).expect("write cache");

        let read = read_cache(&path).expect("round-trip parses");
        let orig_model = schema.options.get("model").unwrap();
        let round_model = read.options.get("model").unwrap();
        assert_eq!(orig_model.ty, round_model.ty);
        assert_eq!(orig_model.short, round_model.short);
        assert_eq!(orig_model.choices, round_model.choices);
    }

    #[test]
    fn read_cache_rejects_invalid_json() {
        let tmp = TestDir::new("sc");
        let path = tmp.path().join("bad.json");
        fs::write(&path, "not json at all").unwrap();
        assert!(read_cache(&path).is_none());
    }

    #[test]
    fn read_cache_unknown_field_falls_back_to_none() {
        // Version-skew guard: a schema emitted by a NEWER
        // flodl-cli-macros than this fdl knows (extra field) must not
        // parse partially (deny_unknown_fields) — and the probe layer
        // degrades to "no cache" so help still renders from the inline
        // yml schema or none.
        let tmp = TestDir::new("sc");
        let path = tmp.path().join("newer.json");
        let body = r#"{
            "options": {
                "model": { "type": "string" }
            },
            "field_from_the_future": true
        }"#;
        fs::write(&path, body).unwrap();
        assert!(read_cache(&path).is_none());
    }

    #[test]
    fn read_cache_rejects_validation_failure() {
        // A schema that clears validation at struct level but fails
        // semantic validation: shadowed fdl-level flag `--help`.
        let tmp = TestDir::new("sc");
        let path = tmp.path().join("bad_sem.json");
        let body = r#"{
            "options": {
                "help": { "type": "bool" }
            }
        }"#;
        fs::write(&path, body).unwrap();
        assert!(
            read_cache(&path).is_none(),
            "cache must not return a schema that fails validate_schema"
        );
    }

    #[test]
    fn is_stale_missing_cache_is_stale() {
        let tmp = TestDir::new("sc");
        let path = tmp.path().join("missing.json");
        assert!(is_stale(&path, &[]));
    }

    /// A cache written by a different `fdl` is stale however fresh its mtime,
    /// because the SCHEMA FORMAT may have changed under it. Without this, the
    /// upgrade that made descriptions carry paragraphs and list items would
    /// have left every existing cache serving the old flattened text with no
    /// signal at all.
    #[test]
    fn is_stale_when_the_cache_was_written_by_another_fdl() {
        let tmp = TestDir::new("sc");
        let cache = cache_path(tmp.path(), "cmd");
        write_cache(&cache, &minimal_schema()).expect("write");
        assert!(
            !is_stale(&cache, &[]),
            "a cache this fdl just wrote must be fresh"
        );

        let stamp = cache.parent().unwrap().join(CACHE_STAMP);
        fs::write(&stamp, "0.0.1-from-the-past").unwrap();
        assert!(is_stale(&cache, &[]), "version mismatch ⇒ stale");

        fs::remove_file(&stamp).unwrap();
        assert!(
            is_stale(&cache, &[]),
            "a cache predating the stamp is stale by definition"
        );
    }

    #[test]
    fn is_stale_compares_mtimes() {
        let tmp = TestDir::new("sc");
        let cache = tmp.path().join("cache.json");
        let source = tmp.path().join("fdl.yml");
        fs::write(&cache, "{}").unwrap();
        fs::write(tmp.path().join(CACHE_STAMP), env!("CARGO_PKG_VERSION")).unwrap();
        // Sleep a moment then touch source so its mtime is newer.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut f = fs::File::create(&source).unwrap();
        writeln!(f, "newer").unwrap();
        assert!(
            is_stale(&cache, std::slice::from_ref(&source)),
            "source newer than cache ⇒ stale"
        );
    }

    #[test]
    fn is_cargo_entry_detects_common_shapes() {
        assert!(is_cargo_entry("cargo run --release --features cuda --"));
        assert!(is_cargo_entry("  cargo run -- "));
        assert!(!is_cargo_entry("./target/release/ddp-bench"));
        assert!(!is_cargo_entry("python ./train.py"));
        assert!(!is_cargo_entry(""));
    }

    #[test]
    fn probe_round_trips_with_mock_binary() {
        // Build a tiny shell script that emits the schema JSON and use it
        // as the "entry". This tests the full probe path end-to-end
        // without pulling in cargo.
        //
        // The entry is `sh <name>`, not the script path on its own, and
        // that is load-bearing on Windows: `probe` shells out through
        // `cmd /C`, which cannot execute a `.sh` and -- worse -- returns
        // success with empty stdout when handed one. Naming `sh`
        // explicitly runs it under Git Bash's sh on every host, and the
        // relative name keeps backslash paths out of sh's hands (probe
        // runs in `cmd_dir`). Same shape as the fdl.yml entry in
        // config::tests::command_tests, which is why that one was
        // portable already.
        let tmp = TestDir::new("sc");
        let script = tmp.path().join("mock-bin.sh");
        let body = r#"#!/bin/sh
cat <<'JSON'
{
  "options": {
    "model": {
      "type": "string",
      "short": "m",
      "description": "pick a model",
      "default": "mlp",
      "choices": ["mlp", "resnet"]
    }
  }
}
JSON
"#;
        fs::write(&script, body).unwrap();

        let schema = probe("sh mock-bin.sh", tmp.path(), None).expect("probe should succeed");
        let model = schema.options.get("model").expect("model opt");
        assert_eq!(model.ty, "string");
        assert_eq!(model.short.as_deref(), Some("m"));
    }

    #[test]
    fn probe_rejects_non_json_output() {
        let tmp = TestDir::new("sc");
        let script = tmp.path().join("junk.sh");
        fs::write(&script, "#!/bin/sh\necho not json\n").unwrap();
        // `sh <name>`: see probe_round_trips_with_mock_binary. This test
        // in particular used to pass on Windows for the wrong reason --
        // cmd /C on a .sh yields empty stdout, which trips the same "no
        // JSON" error the test asserts, so it was green without ever
        // running the script.
        let err = probe("sh junk.sh", tmp.path(), None).expect_err("non-json must fail");
        assert!(
            err.contains("no JSON") || err.contains("valid JSON"),
            "err was: {err}"
        );
    }

    /// The logical `gpu` service must be resolved before compose sees it.
    ///
    /// Regression: `probe` wrapped the raw `docker:` value while the exec path
    /// resolved it, so the day `ddp-bench` switched to `docker: gpu` every
    /// probe died with `no such service: gpu`. Nothing said so — a probe
    /// error is a documented fall-through — and `fdl ddp-bench --help` simply
    /// stopped listing the binary's 59 options.
    #[test]
    fn docker_invocation_resolves_the_logical_gpu_service() {
        let tmp = TestDir::new("sc");
        let cmd_dir = tmp.path().join("bench");
        fs::create_dir_all(&cmd_dir).unwrap();

        let out = docker_invocation("run --fdl-schema", &cmd_dir, "gpu", tmp.path());
        assert!(
            !out.contains(" gpu "),
            "the logical name must never reach compose; got: {out}"
        );
        // No libtorch variant resolves under a temp root, which is
        // `resolve_docker_service`'s documented `cuda` fallback.
        assert!(
            out.contains("--rm cuda "),
            "expected the resolved service; got: {out}"
        );
        assert!(
            out.contains("cd bench &&"),
            "the command dir's relative cd must survive the split; got: {out}"
        );
    }

    /// An explicit service name is a pin and passes through untouched.
    #[test]
    fn docker_invocation_passes_explicit_services_through() {
        let tmp = TestDir::new("sc");
        for svc in ["dev", "rocm", "cuda"] {
            let out = docker_invocation("run --fdl-schema", tmp.path(), svc, tmp.path());
            assert!(out.contains(&format!("--rm {svc} ")), "{svc}: {out}");
        }
    }

    #[test]
    fn probe_rejects_semantically_invalid_schema() {
        let tmp = TestDir::new("sc");
        let script = tmp.path().join("bad.sh");
        // Emits JSON that parses but declares a reserved flag.
        let body = r#"#!/bin/sh
cat <<'JSON'
{ "options": { "help": { "type": "bool" } } }
JSON
"#;
        fs::write(&script, body).unwrap();
        let err = probe("sh bad.sh", tmp.path(), None).expect_err("semantic fail must propagate");
        assert!(
            err.contains("validation") || err.contains("reserved"),
            "err was: {err}"
        );
    }
}
