use super::*;
use std::path::{Path, PathBuf};

fn args(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

/// Zero-dep tempdir helper — matches the pattern used in overlay.rs /
/// dispatch.rs (no `tempfile` crate dependency in flodl-cli).
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("fdl-env-test-{pid}-{n}"));
        std::fs::create_dir_all(&dir).expect("tempdir creation");
        TempDir(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn touch(path: &Path, contents: &str) {
    std::fs::write(path, contents).expect("write fixture");
}

#[test]
fn extract_env_flag_absent_returns_none() {
    let (out, env) = extract_env_flag(&args(&["fdl", "test"])).unwrap();
    assert_eq!(out, args(&["fdl", "test"]));
    assert!(env.is_none());
}

#[test]
fn extract_env_flag_long_separated_form() {
    let (out, env) = extract_env_flag(&args(&["fdl", "--env", "ci", "test"])).unwrap();
    assert_eq!(out, args(&["fdl", "test"]));
    assert_eq!(env.as_deref(), Some("ci"));
}

#[test]
fn extract_env_flag_equals_form() {
    let (out, env) = extract_env_flag(&args(&["fdl", "--env=ci", "test"])).unwrap();
    assert_eq!(out, args(&["fdl", "test"]));
    assert_eq!(env.as_deref(), Some("ci"));
}

#[test]
fn extract_env_flag_scans_anywhere() {
    // Matches `-v`/`-q` global-flag convention: strippable from any position.
    let (out, env) = extract_env_flag(&args(&["fdl", "test", "--env", "prod"])).unwrap();
    assert_eq!(out, args(&["fdl", "test"]));
    assert_eq!(env.as_deref(), Some("prod"));
}

#[test]
fn extract_env_flag_missing_value_errors() {
    let err = extract_env_flag(&args(&["fdl", "--env"])).unwrap_err();
    assert!(err.contains("--env requires a value"), "got: {err}");
}

#[test]
fn extract_env_flag_empty_equals_errors() {
    let err = extract_env_flag(&args(&["fdl", "--env="])).unwrap_err();
    assert!(err.contains("requires a value"), "got: {err}");
}

#[test]
fn extract_env_flag_value_looks_like_flag_errors() {
    // `fdl --env --help` almost certainly means the user forgot the value;
    // loud error beats silently treating `--help` as the env name.
    let err = extract_env_flag(&args(&["fdl", "--env", "--help"])).unwrap_err();
    assert!(err.contains("--env requires a value"), "got: {err}");
}

#[test]
fn extract_env_flag_duplicate_errors() {
    let err = extract_env_flag(&args(&["fdl", "--env", "ci", "--env", "prod"])).unwrap_err();
    assert!(err.contains("more than once"), "got: {err}");
}

#[test]
fn extract_env_flag_duplicate_mixed_forms_errors() {
    let err = extract_env_flag(&args(&["fdl", "--env=ci", "--env", "prod"])).unwrap_err();
    assert!(err.contains("more than once"), "got: {err}");
}

// --- extract_gpus_flag ----------------------------------------------------
//
// Same shape as extract_env_flag: both forms, scan-anywhere, duplicate
// detection, missing-value error, value-looks-like-flag error.

#[test]
fn extract_gpus_flag_absent_returns_none() {
    let (out, spec) = extract_gpus_flag(&args(&["fdl", "test"])).unwrap();
    assert_eq!(out, args(&["fdl", "test"]));
    assert!(spec.is_none());
}

#[test]
fn extract_gpus_flag_long_separated_form() {
    let (out, spec) = extract_gpus_flag(&args(&["fdl", "--gpus", "0,1", "train"])).unwrap();
    assert_eq!(out, args(&["fdl", "train"]));
    assert_eq!(spec, Some(gpus::GpusSpec::List(vec![0, 1])));
}

#[test]
fn extract_gpus_flag_equals_form() {
    let (out, spec) = extract_gpus_flag(&args(&["fdl", "--gpus=0,2", "train"])).unwrap();
    assert_eq!(out, args(&["fdl", "train"]));
    assert_eq!(spec, Some(gpus::GpusSpec::List(vec![0, 2])));
}

#[test]
fn extract_gpus_flag_all_keyword() {
    let (out, spec) = extract_gpus_flag(&args(&["fdl", "--gpus", "all", "train"])).unwrap();
    assert_eq!(out, args(&["fdl", "train"]));
    assert_eq!(spec, Some(gpus::GpusSpec::All));
}

#[test]
fn extract_gpus_flag_scans_anywhere() {
    // Global flag, accepted after the command name too.
    let (out, spec) =
        extract_gpus_flag(&args(&["fdl", "train", "--gpus", "0,1", "--epochs=5"])).unwrap();
    assert_eq!(out, args(&["fdl", "train", "--epochs=5"]));
    assert_eq!(spec, Some(gpus::GpusSpec::List(vec![0, 1])));
}

#[test]
fn extract_gpus_flag_missing_value_errors() {
    let err = extract_gpus_flag(&args(&["fdl", "--gpus"])).unwrap_err();
    assert!(err.contains("requires a value"), "got: {err}");
}

#[test]
fn extract_gpus_flag_empty_equals_errors() {
    let err = extract_gpus_flag(&args(&["fdl", "--gpus="])).unwrap_err();
    assert!(err.contains("requires a value"), "got: {err}");
}

#[test]
fn extract_gpus_flag_value_looks_like_flag_errors() {
    // Catches `fdl --gpus --help` shape -- user forgot a value.
    let err = extract_gpus_flag(&args(&["fdl", "--gpus", "--help"])).unwrap_err();
    assert!(err.contains("requires a value"), "got: {err}");
}

#[test]
fn extract_gpus_flag_stops_at_dashdash() {
    // A `--gpus` after `--` is bound for the inner script, not fdl's own
    // GPU selector; forward it (and its value) verbatim.
    let (out, spec) = extract_gpus_flag(&args(&["fdl", "run", "--", "--gpus", "0,1"])).unwrap();
    assert_eq!(out, args(&["fdl", "run", "--", "--gpus", "0,1"]));
    assert!(spec.is_none());
}

#[test]
fn extract_gpus_flag_duplicate_errors() {
    let err = extract_gpus_flag(&args(&["fdl", "--gpus", "0,1", "--gpus", "2,3"])).unwrap_err();
    assert!(err.contains("more than once"), "got: {err}");
}

#[test]
fn extract_gpus_flag_duplicate_mixed_forms_errors() {
    let err = extract_gpus_flag(&args(&["fdl", "--gpus=0,1", "--gpus", "2,3"])).unwrap_err();
    assert!(err.contains("more than once"), "got: {err}");
}

#[test]
fn extract_gpus_flag_propagates_parse_error() {
    // Invalid index value -> parse error surfaces from GpusSpec::parse.
    let err = extract_gpus_flag(&args(&["fdl", "--gpus", "0,abc"])).unwrap_err();
    assert!(err.contains("cannot parse"), "got: {err}");
    assert!(err.contains("abc"), "got: {err}");
}

// --- extract_at_env: `@<env>` sigil token -----------------------------

#[test]
fn extract_at_env_absent_returns_none() {
    let (out, env) = extract_at_env(&args(&["fdl", "test"])).unwrap();
    assert_eq!(out, args(&["fdl", "test"]));
    assert!(env.is_none());
}

#[test]
fn extract_at_env_consumes_leading_token() {
    let (out, env) = extract_at_env(&args(&["fdl", "@cluster", "probe"])).unwrap();
    assert_eq!(out, args(&["fdl", "probe"]));
    assert_eq!(env.as_deref(), Some("cluster"));
}

#[test]
fn extract_at_env_after_command_is_not_consumed() {
    // `@<env>` is a PRE-COMMAND selector: an `@`-token after the command
    // ("probe") belongs to the command and is forwarded verbatim.
    let (out, env) = extract_at_env(&args(&["fdl", "probe", "@cluster"])).unwrap();
    assert_eq!(out, args(&["fdl", "probe", "@cluster"]));
    assert!(env.is_none());
}

#[test]
fn extract_at_env_option_value_not_stolen() {
    // M20: an `@`-prefixed OPTION VALUE after the command must not be
    // mistaken for an env selector.
    let (out, env) = extract_at_env(&args(&["fdl", "train", "--tag", "@best"])).unwrap();
    assert_eq!(out, args(&["fdl", "train", "--tag", "@best"]));
    assert!(env.is_none());
}

#[test]
fn extract_at_env_stops_at_dashdash() {
    // A `@`-prefixed argument after `--` is bound for the inner command.
    let (out, env) = extract_at_env(&args(&["fdl", "run", "--", "@literal"])).unwrap();
    assert_eq!(out, args(&["fdl", "run", "--", "@literal"]));
    assert!(env.is_none());
}

#[test]
fn extract_at_env_bare_sigil_errors() {
    let err = extract_at_env(&args(&["fdl", "@", "test"])).unwrap_err();
    assert!(err.contains("requires an env name"), "got: {err}");
}

#[test]
fn extract_at_env_duplicate_errors() {
    let err = extract_at_env(&args(&["fdl", "@ci", "@prod", "test"])).unwrap_err();
    assert!(err.contains("more than once"), "got: {err}");
}

#[test]
fn extract_env_flag_stops_at_dashdash() {
    // `--env` past the separator is forwarded to the inner command.
    let (out, env) = extract_env_flag(&args(&["fdl", "run", "--", "--env", "x"])).unwrap();
    assert_eq!(out, args(&["fdl", "run", "--", "--env", "x"]));
    assert!(env.is_none());
}

// --- resolve_env: precedence + resolution (NO overlay validation) ------
//
// resolve_env only RESOLVES + strips the selector; it does NOT check that
// the overlay exists (that is the config-load path's job — see
// config::load_project_with_env / resolve_config_layers). So these need no
// fixture files on disk. `fdl_env` is injected to keep them hermetic.

#[test]
fn resolve_env_at_sigil_resolves_and_strips() {
    let (env, rest) = resolve_env(&args(&["fdl", "@cluster", "probe"]), None).unwrap();
    assert_eq!(env.as_deref(), Some("cluster"));
    assert_eq!(rest, args(&["fdl", "probe"]));
}

#[test]
fn resolve_env_at_sigil_after_command_not_consumed() {
    // Pre-command sigil only: `@cluster` after the command "probe" is the
    // command's own arg, so no env resolves and the token is preserved.
    let (env, rest) = resolve_env(&args(&["fdl", "probe", "@cluster"]), None).unwrap();
    assert!(env.is_none());
    assert_eq!(rest, args(&["fdl", "probe", "@cluster"]));
}

#[test]
fn resolve_env_does_not_validate_overlay_existence() {
    // M21: resolve_env resolves a selector to its name WITHOUT checking the
    // overlay exists — no upfront validation, so `--help` and env-agnostic
    // builtins are never blocked. Existence is enforced later at the
    // config-load path. Covers all three selector forms.
    let (env, rest) = resolve_env(&args(&["fdl", "@nope", "test"]), None).unwrap();
    assert_eq!(env.as_deref(), Some("nope"));
    assert_eq!(rest, args(&["fdl", "test"]));

    let (env, _) = resolve_env(&args(&["fdl", "--env", "nope", "test"]), None).unwrap();
    assert_eq!(env.as_deref(), Some("nope"));

    let (env, _) = resolve_env(&args(&["fdl", "test"]), Some("nope")).unwrap();
    assert_eq!(env.as_deref(), Some("nope"));
}

#[test]
fn resolve_env_cli_selector_wins_over_env_var() {
    // `@prod` (command-line) beats FDL_ENV=stage.
    let (env, rest) = resolve_env(&args(&["fdl", "@prod", "test"]), Some("stage")).unwrap();
    assert_eq!(env.as_deref(), Some("prod"));
    assert_eq!(rest, args(&["fdl", "test"]));
}

#[test]
fn resolve_env_at_and_flag_same_value_ok() {
    let (env, rest) = resolve_env(&args(&["fdl", "@ci", "--env", "ci", "test"]), None).unwrap();
    assert_eq!(env.as_deref(), Some("ci"));
    assert_eq!(rest, args(&["fdl", "test"]));
}

#[test]
fn resolve_env_at_and_flag_conflict_errors() {
    let err = resolve_env(&args(&["fdl", "@ci", "--env", "prod", "test"]), None).unwrap_err();
    assert!(err.contains("conflicting"), "got: {err}");
}

#[test]
fn resolve_env_env_var_used_when_no_cli_selector() {
    let (env, rest) = resolve_env(&args(&["fdl", "test"]), Some("stage")).unwrap();
    assert_eq!(env.as_deref(), Some("stage"));
    assert_eq!(rest, args(&["fdl", "test"]));
}

#[test]
fn resolve_env_bare_overlay_name_is_not_consumed() {
    // The retired positional convention: an overlay named like the first
    // token is NOT auto-selected. `ci` stays a command, env stays None —
    // the `@` sigil is the only positional form.
    let (env, rest) = resolve_env(&args(&["fdl", "ci", "test"]), None).unwrap();
    assert!(env.is_none());
    assert_eq!(rest, args(&["fdl", "ci", "test"]));
}

#[test]
fn resolve_env_at_sigil_after_dashdash_not_consumed() {
    // `@ci` past `--` is a literal arg, not an env selector.
    let (env, rest) = resolve_env(&args(&["fdl", "run", "--", "@ci"]), None).unwrap();
    assert!(env.is_none());
    assert_eq!(rest, args(&["fdl", "run", "--", "@ci"]));
}

#[test]
fn resolve_env_equals_form_consumes_single_token() {
    let (env, rest) = resolve_env(&args(&["fdl", "test", "--env=ci"]), None).unwrap();
    assert_eq!(env.as_deref(), Some("ci"));
    assert_eq!(rest, args(&["fdl", "test"]));
}

#[test]
fn resolve_env_no_selector_returns_none() {
    // `deploy` isn't an env overlay and there's no selector — leave it as
    // the first positional command, env None.
    let (env, rest) = resolve_env(&args(&["fdl", "deploy", "--now"]), None).unwrap();
    assert!(env.is_none());
    assert_eq!(rest, args(&["fdl", "deploy", "--now"]));
}

#[test]
fn load_path_still_errors_on_missing_overlay() {
    // M21 moved overlay validation OUT of resolve_env, not away: the
    // config-load path still errors loudly on a missing overlay, so a
    // command that actually consumes the config (not --help / a builtin)
    // fails with a clear message.
    let tmp = TempDir::new();
    touch(&tmp.path().join("fdl.yml"), "");
    let base = tmp.path().join("fdl.yml");
    let err = config::load_project_with_env(&base, Some("nope")).unwrap_err();
    assert!(err.contains("nope"), "got: {err}");
    assert!(err.contains("not found"), "got: {err}");
}

#[test]
fn help_env_or_note_degrades_on_missing_overlay() {
    // M21 Option B: --help must always render. A present overlay passes
    // through; a missing one degrades to base help (None); no env → None.
    let tmp = TempDir::new();
    touch(&tmp.path().join("fdl.yml"), "");
    touch(&tmp.path().join("fdl.ci.yml"), "");
    assert_eq!(help_env_or_note(tmp.path(), Some("ci")), Some("ci"));
    assert_eq!(help_env_or_note(tmp.path(), Some("nope")), None);
    assert_eq!(help_env_or_note(tmp.path(), None), None);
}

// ── --ansi / --no-ansi extraction ────────────────────────────────────

#[test]
fn extract_ansi_flags_absent_returns_none() {
    let (rest, choice) = extract_ansi_flags(&args(&["fdl", "setup"])).unwrap();
    assert_eq!(rest, args(&["fdl", "setup"]));
    assert!(choice.is_none());
}

#[test]
fn extract_ansi_flags_ansi_forces_always() {
    let (rest, choice) = extract_ansi_flags(&args(&["fdl", "--ansi", "setup"])).unwrap();
    assert_eq!(rest, args(&["fdl", "setup"]));
    assert_eq!(choice, Some(style::ColorChoice::Always));
}

#[test]
fn extract_ansi_flags_no_ansi_forces_never() {
    let (rest, choice) = extract_ansi_flags(&args(&["fdl", "--no-ansi", "setup"])).unwrap();
    assert_eq!(rest, args(&["fdl", "setup"]));
    assert_eq!(choice, Some(style::ColorChoice::Never));
}

#[test]
fn extract_ansi_flags_scans_anywhere() {
    // Position-independent, consistent with -v / --env.
    let (rest, choice) = extract_ansi_flags(&args(&["fdl", "setup", "--no-ansi"])).unwrap();
    assert_eq!(rest, args(&["fdl", "setup"]));
    assert_eq!(choice, Some(style::ColorChoice::Never));
}

#[test]
fn extract_ansi_flags_both_set_errors() {
    let err = extract_ansi_flags(&args(&["fdl", "--ansi", "--no-ansi"])).unwrap_err();
    assert!(err.contains("mutually exclusive"), "got: {err}");
}

#[test]
fn extract_ansi_flags_stops_at_dashdash() {
    // `--no-ansi` after `--` belongs to the inner script; don't consume it.
    let (rest, choice) = extract_ansi_flags(&args(&["fdl", "run", "--", "--no-ansi"])).unwrap();
    assert_eq!(rest, args(&["fdl", "run", "--", "--no-ansi"]));
    assert!(choice.is_none());
}

// --- extract_verbosity: `-v` family, scan-anywhere, `--` boundary --------

#[test]
fn extract_verbosity_strips_flag_before_dashdash() {
    let (out, level) = extract_verbosity(&args(&["fdl", "-vv", "train"]));
    assert_eq!(out, args(&["fdl", "train"]));
    assert_eq!(level, Some(3));
}

#[test]
fn extract_verbosity_stops_at_dashdash() {
    // `-v` after `--` is the inner script's flag, not fdl's verbosity.
    let (out, level) = extract_verbosity(&args(&["fdl", "run", "--", "-v"]));
    assert_eq!(out, args(&["fdl", "run", "--", "-v"]));
    assert!(level.is_none());
}

#[test]
fn extract_no_append_absent_returns_false() {
    let (out, found) = extract_no_append(&args(&["fdl", "test"]));
    assert_eq!(out, args(&["fdl", "test"]));
    assert!(!found);
}

#[test]
fn extract_no_append_strips_flag_from_anywhere_before_dashdash() {
    let (out, found) = extract_no_append(&args(&["fdl", "test", "--no-append", "--", "-p", "x"]));
    assert_eq!(out, args(&["fdl", "test", "--", "-p", "x"]));
    assert!(found);
}

#[test]
fn extract_no_append_preserves_flag_after_dashdash() {
    // Past the first `--` the token is bound for the inner script,
    // so don't swallow it — escape hatch for run-scripts whose own
    // proxy flags collide.
    let (out, found) = extract_no_append(&args(&["fdl", "test", "--", "--no-append"]));
    assert_eq!(out, args(&["fdl", "test", "--", "--no-append"]));
    assert!(!found);
}

#[test]
fn extract_no_prebuild_strips_flag_from_anywhere_before_dashdash() {
    let (out, found) = extract_no_prebuild(&args(&[
        "fdl",
        "@cluster",
        "ddp-bench",
        "--no-prebuild",
        "--mode",
        "nccl-sync",
    ]));
    assert_eq!(
        out,
        args(&["fdl", "@cluster", "ddp-bench", "--mode", "nccl-sync"]),
    );
    assert!(found);
}

#[test]
fn extract_no_prebuild_preserves_flag_after_dashdash() {
    let (out, found) = extract_no_prebuild(&args(&["fdl", "ddp-bench", "--", "--no-prebuild"]));
    assert_eq!(out, args(&["fdl", "ddp-bench", "--", "--no-prebuild"]));
    assert!(!found);
}

#[test]
fn extract_no_prebuild_returns_false_when_absent() {
    let (out, found) = extract_no_prebuild(&args(&["fdl", "@cluster", "train"]));
    assert_eq!(out, args(&["fdl", "@cluster", "train"]));
    assert!(!found);
}
