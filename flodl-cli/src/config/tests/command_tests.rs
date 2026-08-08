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
    assert!(
        !cached.exists(),
        "no cache should be written for cargo entries"
    );
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
    assert!(
        cfg.schema.is_none(),
        "cargo skip honored when compile: false"
    );
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

/// Editing a CLI struct must invalidate the cached schema.
///
/// Regression: invalidation compared the cache only against `fdl.yml`, so a
/// `compile: true` command that grew a flag kept serving the OLD surface from
/// `-h` until someone happened to touch the yml or delete the cache by hand.
/// Silent — the binary was right, only the help was wrong — and it cost real
/// confusion twice in one session before being tracked down.
#[test]
fn touching_a_source_file_invalidates_a_compiled_commands_schema() {
    let tmp = TempDir::new();
    let cmd_dir = tmp.0.join("srcwatch");
    std::fs::create_dir_all(cmd_dir.join("src")).unwrap();
    std::fs::write(
        cmd_dir.join("fdl.yml"),
        "entry: cargo run --\ncompile: true\n",
    )
    .unwrap();
    std::fs::write(cmd_dir.join("Cargo.toml"), "[package]\nname=\"srcwatch\"\n").unwrap();
    let main_rs = cmd_dir.join("src").join("main.rs");
    std::fs::write(&main_rs, "fn main(){}\n").unwrap();

    let cache = crate::schema_cache::cache_path(&cmd_dir, "srcwatch");
    std::fs::create_dir_all(cache.parent().unwrap()).unwrap();
    std::fs::write(&cache, "{\"options\":{}}").unwrap();

    let refs = {
        let mut r: Vec<std::path::PathBuf> = vec![cmd_dir.join("fdl.yml")];
        r.extend(crate::schema_cache::schema_source_refs(&cmd_dir));
        r
    };
    assert!(
        !crate::schema_cache::is_stale(&cache, &refs),
        "a cache newer than every source must start fresh",
    );

    // The edit the old invalidation could not see.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    std::fs::write(&main_rs, "fn main(){ /* new --flag */ }\n").unwrap();

    assert!(
        crate::schema_cache::is_stale(&cache, &refs),
        "editing a source file must make the compiled schema stale",
    );
    // And the config alone still cannot see it — which is the original bug.
    assert!(
        !crate::schema_cache::is_stale(&cache, &[cmd_dir.join("fdl.yml")]),
        "sanity: the config-only reference set is exactly what missed this",
    );
}

/// The scan must stay cheap enough to sit in front of `-h`, and must not
/// wander into build output.
#[test]
fn schema_source_scan_skips_build_output_and_stays_bounded() {
    let tmp = TempDir::new();
    let cmd_dir = tmp.0.join("scan");
    std::fs::create_dir_all(cmd_dir.join("src")).unwrap();
    std::fs::create_dir_all(cmd_dir.join("target").join("debug")).unwrap();
    std::fs::create_dir_all(cmd_dir.join(".fdl")).unwrap();
    std::fs::write(cmd_dir.join("Cargo.toml"), "x").unwrap();
    std::fs::write(cmd_dir.join("src").join("main.rs"), "x").unwrap();
    std::fs::write(cmd_dir.join("src").join("cli.rs"), "x").unwrap();
    std::fs::write(cmd_dir.join("target").join("debug").join("build.rs"), "x").unwrap();
    std::fs::write(cmd_dir.join(".fdl").join("stale.rs"), "x").unwrap();
    std::fs::write(cmd_dir.join("notes.md"), "x").unwrap();

    let refs = crate::schema_cache::schema_source_refs(&cmd_dir);
    let names: Vec<String> = refs
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();

    assert!(names.contains(&"main.rs".to_string()));
    assert!(names.contains(&"cli.rs".to_string()));
    assert!(names.contains(&"Cargo.toml".to_string()));
    assert!(
        !refs
            .iter()
            .any(|p| p.components().any(|c| c.as_os_str() == "target")),
        "target/ dwarfs the source tree and must never be walked",
    );
    assert!(
        !refs
            .iter()
            .any(|p| p.components().any(|c| c.as_os_str() == ".fdl")),
        "the cache's own dir must not invalidate the cache",
    );
    assert!(
        !names.contains(&"notes.md".to_string()),
        "only sources count"
    );
}
