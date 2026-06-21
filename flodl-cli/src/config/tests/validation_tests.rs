//! Tail-validation and preset-validation tests:
//! `validate_tail`, `validate_presets_strict`, `validate_preset_values`.

use super::*;
use std::collections::BTreeMap;

#[test]
fn validate_tail_accepts_known_long_flag() {
    let schema = strict_schema_with_model_option();
    let tail = vec!["--epochs".into(), "3".into()];
    validate_tail(&tail, &schema).expect("known flag must pass");
}

#[test]
fn validate_tail_accepts_known_short_flag() {
    let schema = strict_schema_with_model_option();
    let tail = vec!["-m".into(), "mlp".into()];
    validate_tail(&tail, &schema).expect("known short must pass");
}

#[test]
fn validate_tail_accepts_bool_flag() {
    let schema = strict_schema_with_model_option();
    let tail = vec!["--validate".into()];
    validate_tail(&tail, &schema).expect("bool flag must pass");
}

#[test]
fn validate_tail_strict_rejects_unknown_long_flag() {
    let schema = strict_schema_with_model_option();
    let tail = vec!["--nope".into()];
    let err = validate_tail(&tail, &schema)
        .expect_err("unknown long flag must error in strict mode");
    assert!(err.contains("--nope"), "err was: {err}");
}

#[test]
fn validate_tail_strict_suggests_did_you_mean() {
    // "--epoch" is one char off "--epochs" — edit distance ≤ 2.
    let schema = strict_schema_with_model_option();
    let tail = vec!["--epoch".into(), "3".into()];
    let err = validate_tail(&tail, &schema).expect_err("typo must error");
    assert!(err.contains("did you mean"), "err was: {err}");
    assert!(err.contains("--epochs"), "suggestion missing: {err}");
}

#[test]
fn validate_tail_strict_rejects_unknown_short_flag() {
    let schema = strict_schema_with_model_option();
    let tail = vec!["-z".into()];
    let err = validate_tail(&tail, &schema)
        .expect_err("unknown short must error in strict mode");
    assert!(err.contains("-z"), "err was: {err}");
}

#[test]
fn validate_tail_rejects_bad_choice_always_strict() {
    let schema = strict_schema_with_model_option();
    let tail = vec!["--model".into(), "lenet".into()];
    let err = validate_tail(&tail, &schema)
        .expect_err("out-of-set choice must error");
    assert!(err.contains("lenet"), "err was: {err}");
    assert!(err.contains("allowed"), "err should list allowed values: {err}");
}

#[test]
fn validate_tail_rejects_bad_choice_even_when_not_strict() {
    // The main change in this rollout: `choices:` is a positive
    // assertion by the author, so it must be enforced regardless
    // of `schema.strict`. Only *unknown* flags relax without
    // strict.
    let schema = schema_with_model_option(false);
    let tail = vec!["--model".into(), "lenet".into()];
    let err = validate_tail(&tail, &schema)
        .expect_err("out-of-set choice must error without strict");
    assert!(err.contains("lenet"), "err was: {err}");
    assert!(err.contains("allowed"), "err should list allowed values: {err}");
}

#[test]
fn validate_tail_non_strict_tolerates_unknown_flag() {
    // Without strict, unknown flags are legitimate pass-through
    // candidates (the binary handles them itself).
    let schema = schema_with_model_option(false);
    let tail = vec!["--fancy-passthrough".into(), "value".into()];
    validate_tail(&tail, &schema)
        .expect("unknown flag must be tolerated when strict is off");
}

#[test]
fn validate_tail_non_strict_still_checks_known_short_choices() {
    // The declared short `-m` has choices; a bad value fails even
    // when strict is off. Unknown options would be tolerated, but
    // once the user reaches a declared option, its contract holds.
    let schema = schema_with_model_option(false);
    let tail = vec!["-m".into(), "lenet".into()];
    let err = validate_tail(&tail, &schema)
        .expect_err("out-of-set choice via short must error");
    assert!(err.contains("lenet"), "err was: {err}");
}

#[test]
fn validate_tail_allows_reserved_help() {
    // Reserved universal flags must pass even though they are not
    // declared in the schema. Defense-in-depth against edge cases
    // where `--help` somehow reaches dispatch.
    let schema = strict_schema_with_model_option();
    let tail = vec!["--help".into()];
    validate_tail(&tail, &schema).expect("--help must be allowed");
}

#[test]
fn validate_tail_allows_reserved_fdl_schema() {
    // `fdl ddp-bench --fdl-schema` is forwarded to the binary.
    let schema = strict_schema_with_model_option();
    let tail = vec!["--fdl-schema".into()];
    validate_tail(&tail, &schema).expect("--fdl-schema must be allowed");
}

#[test]
fn validate_tail_passthrough_after_double_dash() {
    // `--` terminates flag parsing. Tokens after it are positionals
    // and must never trigger "unknown flag" errors.
    let schema = strict_schema_with_model_option();
    let tail = vec!["--".into(), "--arbitrary".into(), "anything".into()];
    validate_tail(&tail, &schema).expect("passthrough must work");
}


// ── Tree (variant-shaped) tail validation ───────────────────────────

/// Build a strict tree schema: `train` (with --model choices) + `eval`.
fn strict_tree_schema() -> Schema {
    let mut train = strict_schema_with_model_option();
    train.strict = true;
    let mut eval = Schema {
        strict: true,
        ..Schema::default()
    };
    eval.args.push(arg("checkpoint", "path"));
    let mut root = Schema::default();
    root.commands.insert("train".into(), train);
    root.commands.insert("eval".into(), eval);
    root
}

#[test]
fn validate_tail_descends_into_subcommand_leaf() {
    let schema = strict_tree_schema();
    // A flag valid for `train` passes once routed to the train leaf.
    validate_tail(&["train".into(), "--model".into(), "mlp".into()], &schema)
        .expect("declared train flag must pass");
    // Strict mode on the train leaf rejects an undeclared flag.
    let err = validate_tail(&["train".into(), "--bogus".into()], &schema)
        .expect_err("undeclared flag under train must fail in strict mode");
    assert!(err.contains("bogus"), "got: {err}");
}

#[test]
fn validate_tail_rejects_unknown_subcommand_with_suggestion() {
    let schema = strict_tree_schema();
    let err = validate_tail(&["trian".into()], &schema)
        .expect_err("misspelled subcommand must fail");
    assert!(err.contains("did you mean `train`"), "got: {err}");
}

#[test]
fn validate_tail_tree_choice_violation_caught_at_leaf() {
    let schema = strict_tree_schema();
    let err = validate_tail(&["train".into(), "--model".into(), "nope".into()], &schema)
        .expect_err("invalid choice under train must fail");
    assert!(err.contains("allowed"), "got: {err}");
}

#[test]
fn validate_tail_tree_no_subcommand_is_ok() {
    // Nothing to validate at the leaf yet — the binary handles arity.
    let schema = strict_tree_schema();
    validate_tail(&[], &schema).expect("bare tree invocation must not error");
}

#[test]
fn validate_presets_strict_rejects_unknown_option() {
    let schema = strict_schema_with_model_option();
    let mut commands = BTreeMap::new();
    let mut bad_options = BTreeMap::new();
    bad_options.insert("batchsize".into(), serde_json::json!(32));
    commands.insert(
        "quick".into(),
        CommandSpec {
            options: bad_options,
            ..Default::default()
        },
    );
    let err = validate_presets_strict(&commands, &schema)
        .expect_err("preset pinning undeclared option must error");
    assert!(err.contains("quick"), "err should name the preset: {err}");
    assert!(err.contains("batchsize"), "err should name the key: {err}");
}

#[test]
fn validate_presets_strict_accepts_known_options() {
    let schema = strict_schema_with_model_option();
    let mut commands = BTreeMap::new();
    let mut good_options = BTreeMap::new();
    good_options.insert("model".into(), serde_json::json!("mlp"));
    good_options.insert("epochs".into(), serde_json::json!(5));
    commands.insert(
        "quick".into(),
        CommandSpec {
            options: good_options,
            ..Default::default()
        },
    );
    validate_presets_strict(&commands, &schema)
        .expect("presets with declared options must pass");
}

#[test]
fn validate_presets_strict_ignores_run_and_path_kinds() {
    // Only Preset-kind entries share the parent schema. Run/Path
    // siblings are independent, so strict must not touch them.
    let schema = strict_schema_with_model_option();
    let mut commands = BTreeMap::new();
    commands.insert(
        "helper".into(),
        CommandSpec {
            run: Some("echo hi".into()),
            ..Default::default()
        },
    );
    commands.insert(
        "nested".into(),
        CommandSpec {
            path: Some("./nested/".into()),
            ..Default::default()
        },
    );
    validate_presets_strict(&commands, &schema)
        .expect("run/path siblings must be ignored by preset strict check");
}

// ── Preset value validation (always-on `choices:`) ──────────────

#[test]
fn validate_preset_values_rejects_bad_choice_even_without_strict() {
    // Schema has `choices:` on model; a preset pinning model to
    // something outside the list must fail at load, strict or not.
    let schema = schema_with_model_option(false);
    let mut commands = BTreeMap::new();
    let mut opts = BTreeMap::new();
    opts.insert("model".into(), serde_json::json!("lenet"));
    commands.insert(
        "quick".into(),
        CommandSpec {
            options: opts,
            ..Default::default()
        },
    );
    let err = validate_preset_values(&commands, &schema)
        .expect_err("out-of-choices preset must error");
    assert!(err.contains("quick"), "preset name missing: {err}");
    assert!(err.contains("model"), "option name missing: {err}");
    assert!(err.contains("lenet"), "bad value missing: {err}");
    assert!(err.contains("allowed"), "allowed list missing: {err}");
}

#[test]
fn validate_preset_values_accepts_in_choices_preset() {
    let schema = schema_with_model_option(false);
    let mut commands = BTreeMap::new();
    let mut opts = BTreeMap::new();
    opts.insert("model".into(), serde_json::json!("mlp"));
    commands.insert(
        "quick".into(),
        CommandSpec {
            options: opts,
            ..Default::default()
        },
    );
    validate_preset_values(&commands, &schema)
        .expect("in-choices preset must pass");
}

#[test]
fn validate_preset_values_ignores_undeclared_keys() {
    // Unknown keys aren't our concern here — that's for
    // `validate_presets_strict`, which only runs under strict.
    let schema = schema_with_model_option(false);
    let mut commands = BTreeMap::new();
    let mut opts = BTreeMap::new();
    opts.insert("extra".into(), serde_json::json!("whatever"));
    commands.insert(
        "quick".into(),
        CommandSpec {
            options: opts,
            ..Default::default()
        },
    );
    validate_preset_values(&commands, &schema)
        .expect("undeclared key must be ignored by value validator");
}

#[test]
fn validate_preset_values_ignores_options_without_choices() {
    // `epochs` is declared as int with no `choices:`, so any value
    // passes the choice check (type validation is a separate pass).
    let schema = schema_with_model_option(false);
    let mut commands = BTreeMap::new();
    let mut opts = BTreeMap::new();
    opts.insert("epochs".into(), serde_json::json!(999));
    commands.insert(
        "quick".into(),
        CommandSpec {
            options: opts,
            ..Default::default()
        },
    );
    validate_preset_values(&commands, &schema)
        .expect("no-choices option must accept any value");
}
