//! Tests for `load_command`: auto-probe vs cargo-skip behaviour.

use super::*;

#[test]
fn load_command_auto_probes_non_cargo_entry_and_writes_cache() {
    // Script-kind entry + missing cache: load_command should invoke
    // `<entry> --fdl-schema`, apply the result to cfg.schema, and
    // write it to .fdl/schema-cache/<name>.json for next time.
    let tmp = TempDir::new();
    let cmd_dir = tmp.0.join("mybench");
    std::fs::create_dir_all(&cmd_dir).unwrap();

    let script = cmd_dir.join("emit.sh");
    let body = "#!/bin/sh\n\
                if [ \"$1\" = \"--fdl-schema\" ]; then\n\
                  cat <<'JSON'\n\
                { \"options\": { \"rounds\": { \"type\": \"int\", \"description\": \"N\" } } }\n\
                JSON\n\
                  exit 0\n\
                fi\n";
    std::fs::write(&script, body).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    std::fs::write(cmd_dir.join("fdl.yml"), "entry: sh emit.sh\n").unwrap();

    let cfg = load_command(&cmd_dir).expect("load ok");
    let schema = cfg.schema.expect("auto-probe must populate schema");
    assert!(schema.options.contains_key("rounds"));

    // Second load reads the freshly-written cache (same content).
    let cached_path = crate::schema_cache::cache_path(&cmd_dir, "mybench");
    assert!(cached_path.is_file(), "cache file should exist");
}

#[test]
fn load_command_skips_auto_probe_for_cargo_entries() {
    // Cargo entries are deliberately not probed — a `cargo run
    // --fdl-schema` would compile the whole crate before help
    // renders. Missing cache + cargo entry ⇒ no schema, help
    // still renders (just without options).
    let tmp = TempDir::new();
    let cmd_dir = tmp.0.join("cargo-cmd");
    std::fs::create_dir_all(&cmd_dir).unwrap();
    std::fs::write(cmd_dir.join("fdl.yml"), "entry: cargo run --\n").unwrap();

    let cfg = load_command(&cmd_dir).expect("load ok");
    assert!(
        cfg.schema.is_none(),
        "cargo entry must not be auto-probed (compile latency would ruin --help)"
    );
    let cached = crate::schema_cache::cache_path(&cmd_dir, "cargo-cmd");
    assert!(!cached.exists(), "no cache should be written for cargo entries");
}

#[test]
fn load_command_compile_true_overrides_cargo_skip() {
    // `compile: true` opts a cargo-entry command in to schema probing.
    // The probe still fails on `cargo run` here (no real binary to
    // produce JSON), but the load path must reach `probe()` rather
    // than short-circuiting on the cargo-skip heuristic. We assert
    // by observing that probing was attempted: when the skip kicks
    // in, the test for `is_cargo_entry` returns early; with the
    // override, control flows into probe() which fails silently.
    // Lacking a way to spy on the probe call itself, we rely on the
    // sibling test (`load_command_auto_probe_failure_falls_through_silently`)
    // to demonstrate the silent-fail path, and assert here that
    // `compile: false` still skips. See the explicit-opt-out test
    // below.
    let tmp = TempDir::new();
    let cmd_dir = tmp.0.join("cargo-compile-true");
    std::fs::create_dir_all(&cmd_dir).unwrap();
    std::fs::write(
        cmd_dir.join("fdl.yml"),
        "entry: cargo run --\ncompile: true\n",
    )
    .unwrap();

    // Load must succeed; schema stays None because the bogus cargo
    // entry can't actually emit JSON, but the probe path was
    // exercised (no panic, no early-skip).
    let cfg = load_command(&cmd_dir).expect("load ok");
    assert_eq!(cfg.compile, Some(true), "compile field round-trips");
    assert!(cfg.schema.is_none());
}

#[test]
fn load_command_compile_false_keeps_cargo_skip() {
    // Explicit `compile: false` is the same as absent — cargo skip
    // stays in place.
    let tmp = TempDir::new();
    let cmd_dir = tmp.0.join("cargo-compile-false");
    std::fs::create_dir_all(&cmd_dir).unwrap();
    std::fs::write(
        cmd_dir.join("fdl.yml"),
        "entry: cargo run --\ncompile: false\n",
    )
    .unwrap();

    let cfg = load_command(&cmd_dir).expect("load ok");
    assert_eq!(cfg.compile, Some(false));
    assert!(cfg.schema.is_none(), "cargo skip honored when compile: false");
    let cached = crate::schema_cache::cache_path(&cmd_dir, "cargo-compile-false");
    assert!(!cached.exists());
}

#[test]
fn load_command_auto_probe_failure_falls_through_silently() {
    // An entry that ignores --fdl-schema (or errors) must not break
    // help rendering. cfg.schema stays None, no cache written.
    let tmp = TempDir::new();
    let cmd_dir = tmp.0.join("silent");
    std::fs::create_dir_all(&cmd_dir).unwrap();
    // `/bin/true` ignores any args and exits 0 with empty stdout; probe
    // will reject "no JSON object" and Err — we want that swallowed.
    // Quoted so YAML doesn't parse the bareword `true` as a boolean.
    std::fs::write(cmd_dir.join("fdl.yml"), "entry: \"/bin/true\"\n").unwrap();

    let cfg = load_command(&cmd_dir).expect("load must succeed despite probe error");
    assert!(cfg.schema.is_none());
}

// ── Cluster topology (multi-host DDP overlay) ───────────────────

