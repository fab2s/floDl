//! Schema-validation tests: `validate_schema` invariants,
//! `CommandSpec::kind` semantics, arg/option deserialization.

use super::*;
use std::collections::BTreeMap;

#[test]
fn validate_schema_accepts_minimal_valid() {
    let mut s = Schema::default();
    s.options.insert("model".into(), opt("string"));
    s.options.insert("epochs".into(), opt("int"));
    s.args.push(arg("run-id", "string"));
    validate_schema(&s).expect("minimal valid schema must pass");
}

#[test]
fn validate_schema_rejects_unknown_option_type() {
    let mut s = Schema::default();
    s.options.insert("bad".into(), opt("integer"));
    let err = validate_schema(&s).expect_err("unknown type should fail");
    assert!(err.contains("unknown type"), "err was: {err}");
}

#[test]
fn validate_schema_rejects_reserved_long() {
    let mut s = Schema::default();
    s.options.insert("help".into(), opt("bool"));
    let err = validate_schema(&s).expect_err("reserved --help must fail");
    assert!(err.contains("reserved"), "err was: {err}");
}

#[test]
fn validate_schema_rejects_reserved_short() {
    let mut s = Schema::default();
    let mut o = opt("string");
    o.short = Some("h".into());
    s.options.insert("host".into(), o);
    let err = validate_schema(&s).expect_err("short -h must fail");
    assert!(err.contains("reserved"), "err was: {err}");
}

#[test]
fn validate_schema_rejects_duplicate_short() {
    let mut s = Schema::default();
    let mut a = opt("string");
    a.short = Some("m".into());
    let mut b = opt("string");
    b.short = Some("m".into());
    s.options.insert("model".into(), a);
    s.options.insert("mode".into(), b);
    let err = validate_schema(&s).expect_err("duplicate -m must fail");
    assert!(err.contains("both declare short"), "err was: {err}");
}

#[test]
fn validate_schema_rejects_non_last_variadic() {
    let mut s = Schema::default();
    let mut first = arg("files", "string");
    first.variadic = true;
    s.args.push(first);
    s.args.push(arg("trailer", "string"));
    let err = validate_schema(&s).expect_err("variadic-not-last must fail");
    assert!(err.contains("variadic"), "err was: {err}");
}

#[test]
fn validate_schema_rejects_required_after_optional() {
    let mut s = Schema::default();
    let mut first = arg("maybe", "string");
    first.required = false;
    s.args.push(first);
    s.args.push(arg("need", "string"));
    let err = validate_schema(&s).expect_err("required-after-optional must fail");
    assert!(err.contains("cannot follow"), "err was: {err}");
}

// ── Tree (variant-shaped) schemas: leaf-XOR-branch + recursion ───────

/// A branch node with valid leaf children passes, JSON round-trips, and
/// the leaf children keep serializing in the flat shape (wire BC).
#[test]
fn validate_schema_accepts_variant_tree() {
    let mut train = Schema {
        description: Some("Train a model".into()),
        ..Schema::default()
    };
    train.options.insert("epochs".into(), opt("int"));
    let mut eval = Schema {
        description: Some("Evaluate a model".into()),
        ..Schema::default()
    };
    eval.args.push(arg("checkpoint", "path"));

    let mut root = Schema::default();
    root.commands.insert("train".into(), train);
    root.commands.insert("eval".into(), eval);

    validate_schema(&root).expect("a valid subcommand tree must pass");

    // Round-trip: the tree serializes and parses back identically.
    let json = serde_json::to_string(&root).expect("tree serializes");
    let back: Schema = serde_json::from_str(&json).expect("tree parses back");
    assert_eq!(back.commands.len(), 2);
    assert_eq!(
        back.commands["train"].description.as_deref(),
        Some("Train a model")
    );

    // A leaf with no tree must NOT emit a `commands` key (flat wire BC).
    let leaf_json = serde_json::to_string(&root.commands["eval"]).unwrap();
    assert!(
        !leaf_json.contains("commands"),
        "leaf schema must not serialize an empty commands map; got: {leaf_json}"
    );
}

/// A node cannot be both a leaf (args/options) and a branch (commands).
#[test]
fn validate_schema_rejects_leaf_and_branch_mix() {
    let mut root = Schema::default();
    root.options.insert("global".into(), opt("string"));
    root.commands.insert("train".into(), Schema::default());
    let err = validate_schema(&root).expect_err("leaf+branch mix must fail");
    assert!(err.contains("leaf or a branch"), "err was: {err}");
}

/// Validation recurses into children; a bad leaf surfaces with its path.
#[test]
fn validate_schema_recurses_into_subcommands() {
    let mut bad = Schema::default();
    bad.options.insert("help".into(), opt("bool")); // reserved
    let mut root = Schema::default();
    root.commands.insert("train".into(), bad);
    let err = validate_schema(&root).expect_err("reserved flag in child must fail");
    assert!(
        err.contains("subcommand `train`") && err.contains("reserved"),
        "err must name the offending subcommand; got: {err}"
    );
}

// ── Tail validation (always-on) + strict unknown-rejection ─────

#[test]
fn validate_schema_rejects_required_with_default() {
    let mut s = Schema::default();
    let mut a = arg("x", "string");
    a.default = Some(serde_json::json!("foo"));
    s.args.push(a);
    let err = validate_schema(&s).expect_err("required+default must fail");
    assert!(err.contains("contradiction"), "err was: {err}");
}

/// Regression guard: fdl.yml.example must keep a working `doc` command.
/// The fdl.doc pipeline (api-ref for the port skill, rustdoc warning
/// enforcement in CI) depends on this entry existing and producing output.
#[test]
fn fdl_yml_example_has_doc_script() {
    let cfg = load_example();
    let doc = cfg.commands.get("doc").unwrap_or_else(|| {
        panic!(
            "fdl.yml.example is missing a `doc` command; the rustdoc pipeline \
             depends on `fdl doc` being defined"
        )
    });
    let cmd = doc
        .run
        .as_deref()
        .expect("fdl.yml.example `doc` command must be a `run:` entry");
    assert!(
        !cmd.trim().is_empty(),
        "fdl.yml.example `doc` command has an empty `run:` command"
    );
    assert!(
        cmd.contains("cargo doc"),
        "fdl.yml.example `doc` command must invoke `cargo doc`, got: {cmd}"
    );
    // Must assert some output was produced -- otherwise rustdoc can
    // silently succeed without writing anything useful (e.g. when the
    // target crate fails to resolve). Keeping the exact check liberal:
    // any mention of target/doc as a produced artifact counts.
    assert!(
        cmd.contains("target/doc"),
        "fdl.yml.example `doc` command must verify output was produced \
         (expected a `test -f target/doc/...` check), got: {cmd}"
    );
}

#[test]
fn command_spec_kind_mutex_run_and_path() {
    let spec = CommandSpec {
        run: Some("echo".into()),
        path: Some("x/".into()),
        ..Default::default()
    };
    let err = spec.kind().expect_err("run + path must fail");
    assert!(err.contains("both"), "err was: {err}");
}

#[test]
fn command_spec_kind_path_convention() {
    let spec = CommandSpec::default();
    assert_eq!(spec.kind().unwrap(), CommandKind::Path);
}

#[test]
fn command_spec_kind_preset_when_preset_fields_set() {
    let spec = CommandSpec {
        training: Some(TrainingConfig {
            epochs: Some(1),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert_eq!(spec.kind().unwrap(), CommandKind::Preset);
}

#[test]
fn command_spec_kind_preset_when_only_options_set() {
    // `options:` alone is enough to make a preset — not every preset
    // overrides the structured ddp/training/output blocks.
    let mut options = BTreeMap::new();
    options.insert("model".into(), serde_json::json!("linear"));
    let spec = CommandSpec {
        options,
        ..Default::default()
    };
    assert_eq!(spec.kind().unwrap(), CommandKind::Preset);
}

#[test]
fn command_spec_kind_path_explicit() {
    // Explicit `path:` is a Path even if preset fields are also set;
    // the presence of `path:` is the kind-selecting field.
    let spec = CommandSpec {
        path: Some("./sub/".into()),
        ..Default::default()
    };
    assert_eq!(spec.kind().unwrap(), CommandKind::Path);
}

#[test]
fn command_spec_kind_rejects_docker_without_run() {
    // `docker:` is meaningful only as a wrapper around an inline
    // `run:` script. Pairing it with path/preset is a silent noop
    // at dispatch time, so we reject at load.
    let spec = CommandSpec {
        docker: Some("cuda".into()),
        ..Default::default()
    };
    let err = spec.kind().expect_err("docker without run must fail");
    assert!(err.contains("docker"), "err was: {err}");
}

#[test]
fn command_spec_kind_allows_docker_with_run() {
    let spec = CommandSpec {
        run: Some("cargo test".into()),
        docker: Some("dev".into()),
        ..Default::default()
    };
    assert_eq!(spec.kind().unwrap(), CommandKind::Run);
}

#[test]
fn command_spec_deserialize_from_null() {
    let yaml = "cmd: ~";
    let map: BTreeMap<String, CommandSpec> =
        serde_yaml_ng::from_str(yaml).expect("null must deserialize to default");
    let spec = map.get("cmd").expect("cmd missing");
    assert!(spec.run.is_none() && spec.path.is_none());
    assert_eq!(spec.kind().unwrap(), CommandKind::Path);
}

#[test]
fn command_config_arg_name_deserializes_kebab_case() {
    // YAML uses `arg-name:`, Rust field is `arg_name`.
    let yaml = "arg-name: recipe\nentry: echo\n";
    let cfg: CommandConfig = serde_yaml_ng::from_str(yaml).expect("arg-name must parse");
    assert_eq!(cfg.arg_name.as_deref(), Some("recipe"));
}

#[test]
fn command_config_arg_name_defaults_to_none() {
    let cfg: CommandConfig =
        serde_yaml_ng::from_str("entry: echo\n").expect("minimal cfg must parse");
    assert!(cfg.arg_name.is_none());
}

// ── resolve_config_layers: inherit-from + env composition ────────────
//
// Integration coverage for how `inherit-from:` chains compose with env
// overlays at the config-module boundary. The overlay module already
// tests `resolve_chain` in isolation; here we verify the concat+dedup
