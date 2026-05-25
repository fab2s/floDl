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
    fn extract_gpus_flag_duplicate_errors() {
        let err =
            extract_gpus_flag(&args(&["fdl", "--gpus", "0,1", "--gpus", "2,3"])).unwrap_err();
        assert!(err.contains("more than once"), "got: {err}");
    }

    #[test]
    fn extract_gpus_flag_duplicate_mixed_forms_errors() {
        let err =
            extract_gpus_flag(&args(&["fdl", "--gpus=0,1", "--gpus", "2,3"])).unwrap_err();
        assert!(err.contains("more than once"), "got: {err}");
    }

    #[test]
    fn extract_gpus_flag_propagates_parse_error() {
        // Invalid index value -> parse error surfaces from GpusSpec::parse.
        let err = extract_gpus_flag(&args(&["fdl", "--gpus", "0,abc"])).unwrap_err();
        assert!(err.contains("cannot parse"), "got: {err}");
        assert!(err.contains("abc"), "got: {err}");
    }

    // --- resolve_env: precedence + error paths ----------------------------
    //
    // These exercise the full flag / env-var / first-arg composition with
    // real fixture files on disk. Injecting `cwd` + `fdl_env` keeps them
    // hermetic (no mutation of the process cwd or environment).

    #[test]
    fn resolve_env_flag_wins_over_env_var_and_first_arg() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");
        touch(&tmp.path().join("fdl.ci.yml"), "");
        touch(&tmp.path().join("fdl.prod.yml"), "");
        touch(&tmp.path().join("fdl.stage.yml"), "");

        // --env prod is explicit → beats FDL_ENV=stage and the `ci` first-arg.
        let (env, rest) = resolve_env(
            &args(&["fdl", "ci", "--env", "prod", "test"]),
            tmp.path(),
            Some("stage"),
        )
        .unwrap();
        assert_eq!(env.as_deref(), Some("prod"));
        // --env/--env=value tokens stripped; `ci` stays as the first-arg candidate,
        // which resolve_env_first_arg is skipped for entirely.
        assert_eq!(rest, args(&["fdl", "ci", "test"]));
    }

    #[test]
    fn resolve_env_env_var_wins_over_first_arg() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");
        touch(&tmp.path().join("fdl.ci.yml"), "");
        touch(&tmp.path().join("fdl.stage.yml"), "");

        let (env, rest) =
            resolve_env(&args(&["fdl", "ci", "test"]), tmp.path(), Some("stage")).unwrap();
        assert_eq!(env.as_deref(), Some("stage"));
        // First-arg `ci` left untouched when FDL_ENV takes over.
        assert_eq!(rest, args(&["fdl", "ci", "test"]));
    }

    #[test]
    fn resolve_env_empty_env_var_falls_through_to_first_arg() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");
        touch(&tmp.path().join("fdl.ci.yml"), "");

        let (env, rest) = resolve_env(&args(&["fdl", "ci", "test"]), tmp.path(), Some("")).unwrap();
        assert_eq!(env.as_deref(), Some("ci"));
        assert_eq!(rest, args(&["fdl", "test"]));
    }

    #[test]
    fn resolve_env_first_arg_still_works_when_no_explicit_selector() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");
        touch(&tmp.path().join("fdl.ci.yml"), "");

        let (env, rest) = resolve_env(&args(&["fdl", "ci", "test"]), tmp.path(), None).unwrap();
        assert_eq!(env.as_deref(), Some("ci"));
        assert_eq!(rest, args(&["fdl", "test"]));
    }

    #[test]
    fn resolve_env_flag_errors_on_missing_overlay() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");

        let err = resolve_env(&args(&["fdl", "--env", "nope", "test"]), tmp.path(), None)
            .unwrap_err();
        assert!(err.contains("--env"), "got: {err}");
        assert!(err.contains("nope"), "got: {err}");
        assert!(err.contains("not found"), "got: {err}");
    }

    #[test]
    fn resolve_env_env_var_errors_on_missing_overlay() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");

        let err = resolve_env(&args(&["fdl", "test"]), tmp.path(), Some("nope")).unwrap_err();
        assert!(err.contains("FDL_ENV"), "got: {err}");
        assert!(err.contains("nope"), "got: {err}");
        assert!(err.contains("not found"), "got: {err}");
    }

    #[test]
    fn resolve_env_equals_form_consumes_single_token() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");
        touch(&tmp.path().join("fdl.ci.yml"), "");

        let (env, rest) =
            resolve_env(&args(&["fdl", "test", "--env=ci"]), tmp.path(), None).unwrap();
        assert_eq!(env.as_deref(), Some("ci"));
        assert_eq!(rest, args(&["fdl", "test"]));
    }

    #[test]
    fn resolve_env_first_arg_unknown_falls_through() {
        // `deploy` isn't an env overlay — leave it as the first positional.
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");

        let (env, rest) =
            resolve_env(&args(&["fdl", "deploy", "--now"]), tmp.path(), None).unwrap();
        assert!(env.is_none());
        assert_eq!(rest, args(&["fdl", "deploy", "--now"]));
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
        let (rest, choice) =
            extract_ansi_flags(&args(&["fdl", "setup", "--no-ansi"])).unwrap();
        assert_eq!(rest, args(&["fdl", "setup"]));
        assert_eq!(choice, Some(style::ColorChoice::Never));
    }

    #[test]
    fn extract_ansi_flags_both_set_errors() {
        let err = extract_ansi_flags(&args(&["fdl", "--ansi", "--no-ansi"])).unwrap_err();
        assert!(err.contains("mutually exclusive"), "got: {err}");
    }

    #[test]
    fn extract_no_append_absent_returns_false() {
        let (out, found) = extract_no_append(&args(&["fdl", "test"]));
        assert_eq!(out, args(&["fdl", "test"]));
        assert!(!found);
    }

    #[test]
    fn extract_no_append_strips_flag_from_anywhere_before_dashdash() {
        let (out, found) =
            extract_no_append(&args(&["fdl", "test", "--no-append", "--", "-p", "x"]));
        assert_eq!(out, args(&["fdl", "test", "--", "-p", "x"]));
        assert!(found);
    }

    #[test]
    fn extract_no_append_preserves_flag_after_dashdash() {
        // Past the first `--` the token is bound for the inner script,
        // so don't swallow it — escape hatch for run-scripts whose own
        // proxy flags collide.
        let (out, found) =
            extract_no_append(&args(&["fdl", "test", "--", "--no-append"]));
        assert_eq!(out, args(&["fdl", "test", "--", "--no-append"]));
        assert!(!found);
    }

    #[test]
    fn extract_no_prebuild_strips_flag_from_anywhere_before_dashdash() {
        let (out, found) = extract_no_prebuild(&args(&[
            "fdl", "@cluster", "ddp-bench", "--no-prebuild", "--mode", "nccl-sync",
        ]));
        assert_eq!(
            out,
            args(&["fdl", "@cluster", "ddp-bench", "--mode", "nccl-sync"]),
        );
        assert!(found);
    }

    #[test]
    fn extract_no_prebuild_preserves_flag_after_dashdash() {
        let (out, found) = extract_no_prebuild(&args(&[
            "fdl", "ddp-bench", "--", "--no-prebuild",
        ]));
        assert_eq!(out, args(&["fdl", "ddp-bench", "--", "--no-prebuild"]));
        assert!(!found);
    }

    #[test]
    fn extract_no_prebuild_returns_false_when_absent() {
        let (out, found) = extract_no_prebuild(&args(&["fdl", "@cluster", "train"]));
        assert_eq!(out, args(&["fdl", "@cluster", "train"]));
        assert!(!found);
    }
