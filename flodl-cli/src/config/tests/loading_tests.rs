//! Tests for `resolve_config_layers`: env-overlay chain composition,
//! deduplication, error surfacing.

use super::*;

#[test]
fn resolve_config_layers_base_only() {
    let tmp = TempDir::new();
    let base = tmp.0.join("fdl.yml");
    std::fs::write(&base, "a: 1\n").unwrap();
    let layers = resolve_config_layers(&base, None).unwrap();
    assert_eq!(filenames(&layers), vec!["fdl.yml"]);
}

#[test]
fn resolve_config_layers_base_with_env_overlay() {
    let tmp = TempDir::new();
    let base = tmp.0.join("fdl.yml");
    let env = tmp.0.join("fdl.ci.yml");
    std::fs::write(&base, "a: 1\n").unwrap();
    std::fs::write(&env, "b: 2\n").unwrap();
    let layers = resolve_config_layers(&base, Some("ci")).unwrap();
    assert_eq!(filenames(&layers), vec!["fdl.yml", "fdl.ci.yml"]);
}

#[test]
fn resolve_config_layers_env_inherits_from_mixin() {
    // fdl.ci.yml inherits from fdl.cloud.yml (standalone mix-in, not
    // derived from base). Combined chain: [base, cloud, ci].
    let tmp = TempDir::new();
    let base = tmp.0.join("fdl.yml");
    let cloud = tmp.0.join("fdl.cloud.yml");
    let ci = tmp.0.join("fdl.ci.yml");
    std::fs::write(&base, "a: 1\n").unwrap();
    std::fs::write(&cloud, "b: 2\n").unwrap();
    std::fs::write(&ci, "inherit-from: fdl.cloud.yml\nc: 3\n").unwrap();
    let layers = resolve_config_layers(&base, Some("ci")).unwrap();
    assert_eq!(
        filenames(&layers),
        vec!["fdl.yml", "fdl.cloud.yml", "fdl.ci.yml"]
    );
}

#[test]
fn resolve_config_layers_dedups_when_env_inherits_from_base() {
    // fdl.ci.yml inherits from fdl.yml directly. Base is already in
    // the layer list, so env's chain collapses into it — the final
    // list must not have fdl.yml twice.
    let tmp = TempDir::new();
    let base = tmp.0.join("fdl.yml");
    let ci = tmp.0.join("fdl.ci.yml");
    std::fs::write(&base, "a: 1\n").unwrap();
    std::fs::write(&ci, "inherit-from: fdl.yml\nb: 2\n").unwrap();
    let layers = resolve_config_layers(&base, Some("ci")).unwrap();
    assert_eq!(filenames(&layers), vec!["fdl.yml", "fdl.ci.yml"]);
}

#[test]
fn resolve_config_layers_merged_value_matches_chain() {
    // End-to-end: the merge result should reflect the chain order
    // (base < cloud < ci), with each subsequent layer overriding.
    let tmp = TempDir::new();
    let base = tmp.0.join("fdl.yml");
    let cloud = tmp.0.join("fdl.cloud.yml");
    let ci = tmp.0.join("fdl.ci.yml");
    std::fs::write(&base, "value: base\nkeep_base: yes\n").unwrap();
    std::fs::write(&cloud, "value: cloud\nkeep_cloud: yes\n").unwrap();
    std::fs::write(
        &ci,
        "inherit-from: fdl.cloud.yml\nvalue: ci\nkeep_ci: yes\n",
    )
    .unwrap();
    let merged = load_merged_value(&base, Some("ci")).unwrap();
    let m = merged.as_mapping().unwrap();
    // Last writer wins on `value`.
    assert_eq!(
        m.get(serde_yaml::Value::String("value".into())).unwrap(),
        &serde_yaml::Value::String("ci".into())
    );
    // Each layer's unique key survives.
    assert!(m.contains_key(serde_yaml::Value::String("keep_base".into())));
    assert!(m.contains_key(serde_yaml::Value::String("keep_cloud".into())));
    assert!(m.contains_key(serde_yaml::Value::String("keep_ci".into())));
}

#[test]
fn resolve_config_layers_missing_env_errors() {
    let tmp = TempDir::new();
    let base = tmp.0.join("fdl.yml");
    std::fs::write(&base, "a: 1\n").unwrap();
    let err = resolve_config_layers(&base, Some("nope")).unwrap_err();
    assert!(err.contains("nope"));
    assert!(err.contains("not found"));
}

#[test]
fn resolve_config_layers_base_inherit_from_chain() {
    // Base itself uses inherit-from: shared-defaults.yml. The
    // defaults live in a sibling file and are merged UNDER the base.
    let tmp = TempDir::new();
    let defaults = tmp.0.join("shared.yml");
    let base = tmp.0.join("fdl.yml");
    std::fs::write(&defaults, "policy: default\n").unwrap();
    std::fs::write(&base, "inherit-from: shared.yml\npolicy: override\n").unwrap();
    let layers = resolve_config_layers(&base, None).unwrap();
    assert_eq!(filenames(&layers), vec!["shared.yml", "fdl.yml"]);
}

#[test]
fn load_project_with_env_empty_config_is_default() {
    // L10 #4: an empty (or comments-only) fdl.yml merges to a null value.
    // That is a valid "nothing configured" state — it must load as an
    // all-defaults ProjectConfig, not fail with a cryptic serde
    // "invalid type: unit value" error.
    let tmp = TempDir::new();
    let base = tmp.0.join("fdl.yml");
    std::fs::write(&base, "# just a comment\n").unwrap();
    let cfg = load_project_with_env(&base, None)
        .expect("empty config must load as defaults, not error");
    assert!(cfg.commands.is_empty());
    assert!(cfg.cluster.is_none());
    assert!(cfg.description.is_none());
}
