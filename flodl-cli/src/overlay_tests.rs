use super::*;
use std::collections::BTreeMap;

fn yaml(s: &str) -> Value {
    serde_yaml_ng::from_str(s).expect("test fixture must parse")
}

/// Build `Vec<String>` from string literals — shorter than repeating
/// `.to_string()` in every path assertion.
fn p(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

#[test]
fn scalar_over_scalar_replaces() {
    let base = yaml("42");
    let over = yaml("99");
    assert_eq!(deep_merge(base, over), yaml("99"));
}

#[test]
fn map_keys_deep_merge() {
    let base = yaml(
        r"
            a: 1
            nested:
              x: one
              y: two
            ",
    );
    let over = yaml(
        r"
            nested:
              y: TWO
              z: three
            b: 2
            ",
    );
    let expected = yaml(
        r"
            a: 1
            b: 2
            nested:
              x: one
              y: TWO
              z: three
            ",
    );
    assert_eq!(deep_merge(base, over), expected);
}

#[test]
fn lists_replace_not_append() {
    let base = yaml(
        r"
            items: [a, b, c]
            ",
    );
    let over = yaml(
        r"
            items: [x, y]
            ",
    );
    let expected = yaml(
        r"
            items: [x, y]
            ",
    );
    assert_eq!(deep_merge(base, over), expected);
}

#[test]
fn null_in_overlay_deletes_key() {
    let base = yaml(
        r"
            ddp:
              policy: cadence
              anchor: 3
            training:
              epochs: 10
            ",
    );
    let over = yaml(
        r"
            ddp: ~
            training:
              epochs: 20
            ",
    );
    // `ddp: null` removes the whole block; training.epochs updates.
    let expected = yaml(
        r"
            training:
              epochs: 20
            ",
    );
    assert_eq!(deep_merge(base, over), expected);
}

#[test]
fn null_leaf_removes_single_key() {
    let base = yaml(
        r"
            ddp:
              policy: cadence
              anchor: 3
            ",
    );
    let over = yaml(
        r"
            ddp:
              anchor: ~
            ",
    );
    let expected = yaml(
        r"
            ddp:
              policy: cadence
            ",
    );
    assert_eq!(deep_merge(base, over), expected);
}

#[test]
fn overlay_adds_new_top_level_key() {
    let base = yaml("a: 1");
    let over = yaml("b: 2");
    let expected = yaml(
        r"
            a: 1
            b: 2
            ",
    );
    assert_eq!(deep_merge(base, over), expected);
}

#[test]
fn merge_chain_three_layers() {
    let l1 = yaml("a: 1\nb: 1");
    let l2 = yaml("b: 2\nc: 2");
    let l3 = yaml("c: 3");
    let got = merge_layers(vec![l1, l2, l3]);
    let expected = yaml(
        r"
            a: 1
            b: 2
            c: 3
            ",
    );
    assert_eq!(got, expected);
}

#[test]
fn type_change_overlay_replaces_wholesale() {
    let base = yaml(
        r"
            ddp:
              policy: cadence
            ",
    );
    let over = yaml(
        r"
            ddp: solo-0
            ",
    );
    let expected = yaml(
        r"
            ddp: solo-0
            ",
    );
    assert_eq!(deep_merge(base, over), expected);
}

#[test]
fn type_change_scalar_base_mapping_overlay_replaces() {
    // Symmetry with `type_change_overlay_replaces_wholesale`: when
    // the base is a scalar and the overlay is a mapping, the mapping
    // wins wholesale. No attempt at cross-type merging.
    let base = yaml(
        r"
            ddp: solo-0
            ",
    );
    let over = yaml(
        r"
            ddp:
              policy: cadence
              anchor: 3
            ",
    );
    let expected = yaml(
        r"
            ddp:
              policy: cadence
              anchor: 3
            ",
    );
    assert_eq!(deep_merge(base, over), expected);
}

#[test]
fn list_envs_discovers_sibling_overlays() {
    let tmp = tempdir();
    std::fs::write(tmp.path().join("fdl.yml"), "description: base").unwrap();
    std::fs::write(tmp.path().join("fdl.ci.yml"), "description: ci").unwrap();
    std::fs::write(tmp.path().join("fdl.cloud.yaml"), "description: cloud").unwrap();
    std::fs::write(tmp.path().join("fdl.prod.json"), "{}").unwrap();
    // Decoys — must NOT be listed.
    std::fs::write(tmp.path().join("fdl.yml.example"), "").unwrap();
    std::fs::write(tmp.path().join("other.ci.yml"), "").unwrap();
    std::fs::write(tmp.path().join("fdl.yml.bak"), "").unwrap();

    let envs = list_envs(&tmp.path().join("fdl.yml"));
    assert_eq!(envs, vec!["ci".to_string(), "cloud".into(), "prod".into()]);
}

#[test]
fn find_env_file_prefers_yaml_matching_base_precedence() {
    // Extension precedence matches config::CONFIG_NAMES (`.yaml` before
    // `.yml`), so overlay resolution picks the same extension the base file
    // would when both siblings exist. (Previously overlay preferred `.yml`,
    // diverging from the base — L10 #3.)
    let tmp = tempdir();
    std::fs::write(tmp.path().join("fdl.yml"), "").unwrap();
    std::fs::write(tmp.path().join("fdl.ci.yml"), "# yml loses").unwrap();
    std::fs::write(tmp.path().join("fdl.ci.yaml"), "# yaml wins").unwrap();

    let got = find_env_file(&tmp.path().join("fdl.yml"), "ci").unwrap();
    assert_eq!(got.file_name().unwrap().to_str(), Some("fdl.ci.yaml"));
}

#[test]
fn find_env_file_missing_returns_none() {
    let tmp = tempdir();
    std::fs::write(tmp.path().join("fdl.yml"), "").unwrap();
    assert!(find_env_file(&tmp.path().join("fdl.yml"), "nope").is_none());
}

// ── Annotated merge ──────────────────────────────────────────────────

/// Collect every leaf's (key-path, source-index) from an AnnotatedNode.
/// Key path elements are YAML `Value`s (almost always strings in our
/// configs) for parity with [`AnnotatedNode::Map`]'s key type.
fn leaves(node: &AnnotatedNode) -> Vec<(Vec<String>, usize)> {
    fn walk(node: &AnnotatedNode, path: &mut Vec<String>, out: &mut Vec<(Vec<String>, usize)>) {
        match node {
            AnnotatedNode::Leaf { source, .. } => out.push((path.clone(), *source)),
            AnnotatedNode::Map { entries } => {
                for (k, v) in entries {
                    let key = match k {
                        Value::String(s) => s.clone(),
                        other => format!("{other:?}"),
                    };
                    path.push(key);
                    walk(v, path, out);
                    path.pop();
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(node, &mut Vec::new(), &mut out);
    out
}

#[test]
fn annotated_single_layer_tags_every_leaf_with_zero() {
    let layers = vec![yaml(
        "ddp:\n  policy: cadence\n  anchor: 3\ntraining:\n  epochs: 10\n",
    )];
    let node = merge_layers_annotated(&layers);
    for (path, src) in leaves(&node) {
        assert_eq!(src, 0, "{path:?} should be tagged with layer 0");
    }
}

#[test]
fn annotated_overlay_replaces_key_source() {
    let layers = vec![
        yaml("ddp:\n  policy: cadence\n  anchor: 3\n"),
        yaml("ddp:\n  anchor: 5\n"),
    ];
    let node = merge_layers_annotated(&layers);
    let by_path: BTreeMap<Vec<String>, usize> = leaves(&node).into_iter().collect();
    assert_eq!(by_path[&p(&["ddp", "policy"])], 0);
    assert_eq!(by_path[&p(&["ddp", "anchor"])], 1);
}

#[test]
fn annotated_added_key_tagged_with_overlay() {
    let layers = vec![
        yaml("ddp:\n  policy: cadence\n"),
        yaml("training:\n  epochs: 20\n"),
    ];
    let node = merge_layers_annotated(&layers);
    let by_path: BTreeMap<Vec<String>, usize> = leaves(&node).into_iter().collect();
    assert_eq!(by_path[&p(&["training", "epochs"])], 1);
}

#[test]
fn annotated_null_deletes_key_and_removes_leaf() {
    let layers = vec![
        yaml("ddp:\n  policy: cadence\n  anchor: 3\n"),
        yaml("ddp:\n  anchor: ~\n"),
    ];
    let node = merge_layers_annotated(&layers);
    let paths: Vec<Vec<String>> = leaves(&node).into_iter().map(|(path, _)| path).collect();
    assert!(paths.contains(&p(&["ddp", "policy"])));
    assert!(!paths.iter().any(|path| path == &p(&["ddp", "anchor"])));
}

#[test]
fn annotated_type_change_resets_source_to_overlay() {
    // Mapping in base → scalar in overlay: the whole subtree collapses
    // to a Leaf tagged with the overlay's index.
    let layers = vec![yaml("ddp:\n  policy: cadence\n"), yaml("ddp: solo-0\n")];
    let node = merge_layers_annotated(&layers);
    let by_path: BTreeMap<Vec<String>, usize> = leaves(&node).into_iter().collect();
    assert_eq!(by_path[&p(&["ddp"])], 1);
    assert!(!by_path.contains_key(&p(&["ddp", "policy"])));
}

#[test]
fn annotated_list_replaced_wholesale_tagged_with_setter() {
    // Lists are replace-not-append, so the whole sequence is attributed
    // to the layer that last wrote it.
    let layers = vec![
        yaml("regions: [eu-west]\n"),
        yaml("regions: [us-east, ap-south]\n"),
    ];
    let node = merge_layers_annotated(&layers);
    let by_path: BTreeMap<Vec<String>, usize> = leaves(&node).into_iter().collect();
    assert_eq!(by_path[&p(&["regions"])], 1);
}

#[test]
fn annotated_three_layer_chain() {
    let layers = vec![
        yaml("a: 1\nb: 1\nc: 1\n"),
        yaml("b: 2\nc: 2\n"),
        yaml("c: 3\n"),
    ];
    let node = merge_layers_annotated(&layers);
    let by_path: BTreeMap<Vec<String>, usize> = leaves(&node).into_iter().collect();
    assert_eq!(by_path[&p(&["a"])], 0);
    assert_eq!(by_path[&p(&["b"])], 1);
    assert_eq!(by_path[&p(&["c"])], 2);
}

#[test]
fn annotated_to_value_matches_deep_merge() {
    let l1 = yaml("ddp:\n  policy: cadence\n  anchor: 3\ntraining:\n  epochs: 10\n");
    let l2 = yaml("ddp:\n  anchor: 5\ntraining:\n  seed: 42\n");
    let annotated = merge_layers_annotated(&[l1.clone(), l2.clone()]);
    let plain = deep_merge(l1, l2);
    assert_eq!(annotated.to_value(), plain);
}

// ── Rendering ────────────────────────────────────────────────────────

fn labels(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

/// Render with color forced off. `render_annotated_yaml` colorizes
/// keys when `style::color_enabled()` is true; inside Docker that
/// fires whenever stderr is a TTY (the docker compose run side),
/// which makes text-content assertions on `out` fail with stray
/// ANSI bytes. Lock against `style::tests`'s env-mutating tests via
/// the shared `TEST_ENV_LOCK` and restore the prior choice on exit.
fn render_no_color(node: &AnnotatedNode, source_labels: &[String]) -> String {
    let _g = crate::style::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let saved = crate::style::color_choice();
    crate::style::set_color_choice(crate::style::ColorChoice::Never);
    let out = render_annotated_yaml(node, source_labels);
    crate::style::set_color_choice(saved);
    out
}

#[test]
fn render_tags_every_leaf_with_filename() {
    let layers = vec![yaml("ddp:\n  policy: cadence\n  anchor: 3\n")];
    let node = merge_layers_annotated(&layers);
    let out = render_no_color(&node, &labels(&["fdl.yml"]));
    for line in out.lines() {
        if line.contains(':') && !line.trim_end().ends_with(':') {
            assert!(line.contains("# fdl.yml"), "missing tag on: `{line}`");
        }
    }
}

#[test]
fn render_tags_overlay_keys_with_overlay_filename() {
    let layers = vec![
        yaml("ddp:\n  policy: cadence\n  anchor: 3\n"),
        yaml("ddp:\n  anchor: 5\n"),
    ];
    let node = merge_layers_annotated(&layers);
    let out = render_no_color(&node, &labels(&["fdl.yml", "fdl.ci.yml"]));
    // policy unchanged → tagged with base.
    let policy_line = out.lines().find(|l| l.contains("policy:")).unwrap();
    assert!(policy_line.contains("# fdl.yml") && !policy_line.contains("# fdl.ci.yml"));
    // anchor overridden → tagged with overlay.
    let anchor_line = out.lines().find(|l| l.contains("anchor:")).unwrap();
    assert!(anchor_line.contains("# fdl.ci.yml"));
}

#[test]
fn render_aligns_comment_column() {
    let layers = vec![yaml("a: 1\nbb: 22\nccc: 333\n")];
    let node = merge_layers_annotated(&layers);
    let out = render_no_color(&node, &labels(&["fdl.yml"]));
    // All `#` symbols must land in the same column.
    let cols: Vec<usize> = out.lines().filter_map(|l| l.find('#')).collect();
    assert!(cols.len() >= 3);
    let first = cols[0];
    assert!(
        cols.iter().all(|c| *c == first),
        "mismatched columns: {cols:?}"
    );
}

#[test]
fn render_inline_short_scalar_list() {
    // `serde_yaml_ng::Number::to_string` preserves `1.0` as `1.0`.
    let layers = vec![yaml("ratios: [1.5, 1.0]\n")];
    let node = merge_layers_annotated(&layers);
    let out = render_no_color(&node, &labels(&["fdl.yml"]));
    assert!(out.contains("ratios: [1.5, 1.0]"), "got:\n{out}");
    assert!(out.lines().next().unwrap().contains("# fdl.yml"));
}

#[test]
fn render_deleted_key_absent_from_output() {
    let layers = vec![
        yaml("ddp:\n  policy: cadence\n  anchor: 3\n"),
        yaml("ddp:\n  anchor: ~\n"),
    ];
    let node = merge_layers_annotated(&layers);
    let out = render_no_color(&node, &labels(&["fdl.yml", "fdl.ci.yml"]));
    assert!(!out.contains("anchor"), "deleted key leaked: {out}");
    assert!(out.contains("policy"));
}

#[test]
fn render_header_lines_have_no_comment() {
    // The `ddp:` header line is a nested-map opener — it has no single
    // source, so it gets no trailing `# <label>`.
    let layers = vec![yaml("ddp:\n  policy: cadence\n")];
    let node = merge_layers_annotated(&layers);
    let out = render_no_color(&node, &labels(&["fdl.yml"]));
    let header = out.lines().find(|l| l.trim() == "ddp:").unwrap();
    assert!(!header.contains('#'));
}

#[test]
fn render_quotes_ambiguous_strings() {
    // `true` as a literal string must be quoted so it doesn't
    // round-trip as a boolean.
    let layers = vec![yaml("flag: \"true\"\n")];
    let node = merge_layers_annotated(&layers);
    let out = render_no_color(&node, &labels(&["fdl.yml"]));
    assert!(out.contains("flag: \"true\""), "got:\n{out}");
}

#[test]
fn render_long_scalar_list_drops_to_block_form() {
    let long: Vec<String> = (0..30).map(|i| format!("item-number-{i}")).collect();
    let yaml_src = format!("items: [{}]\n", long.join(", "));
    let layers = vec![yaml(&yaml_src)];
    let node = merge_layers_annotated(&layers);
    let out = render_no_color(&node, &labels(&["fdl.yml"]));
    assert!(out.contains("items:  "), "expected header line with tag");
    assert!(out.contains("- item-number-0"));
}

#[test]
fn render_sequence_in_mapping_list_item_inlines_scalars() {
    // Regression: previously rendered `ranks: - 0` because
    // format_scalar's defensive Sequence fallback emitted block YAML.
    let layers = vec![yaml(
        "cluster:\n  hosts:\n    - name: exa\n      ranks: [0, 1]\n      local_devices: [2, 3]\n",
    )];
    let node = merge_layers_annotated(&layers);
    let out = render_no_color(&node, &labels(&["fdl.yml"]));
    assert!(
        out.contains("ranks: [0, 1]"),
        "ranks should inline as `[0, 1]`, got:\n{out}"
    );
    assert!(
        out.contains("local_devices: [2, 3]"),
        "local_devices should inline as `[2, 3]`, got:\n{out}"
    );
    assert!(
        !out.contains("ranks: - "),
        "must not produce `ranks: - 0`, got:\n{out}"
    );
}

#[test]
fn render_long_line_does_not_push_comments_past_cap() {
    // Regression: previously, one ~90-char clippy command pushed every
    // other comment to ~100+ cols, wrapping in the terminal.
    let layers = vec![yaml(
        "short: 1\nlong: \"some pretty long shell command with many flags and pipes\"\nshort2: 2\n",
    )];
    let node = merge_layers_annotated(&layers);
    let out = render_no_color(&node, &labels(&["fdl.yml"]));
    let short_col = out
        .lines()
        .find(|l| l.starts_with("short:"))
        .unwrap()
        .find('#')
        .unwrap();
    // ALIGN_CAP + small padding; assert short lines align to a sane
    // column (well under terminal width).
    assert!(
        short_col < 60,
        "short-line comment ended up at col {short_col}, expected < 60"
    );
}

// ── inherit-from chain resolution ────────────────────────────────────

/// Canonicalise a path so tests can compare against `resolve_chain`'s
/// returned paths (which are always canonical).
fn canon(p: &Path) -> PathBuf {
    p.canonicalize().expect("canonicalize fixture path")
}

#[test]
fn resolve_chain_single_file_no_inherit() {
    let tmp = tempdir();
    let f = tmp.path().join("fdl.yml");
    std::fs::write(&f, "description: test\nddp:\n  policy: cadence\n").unwrap();
    let chain = resolve_chain(&f).unwrap();
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0].0, canon(&f));
}

#[test]
fn resolve_chain_strips_inherit_from_key() {
    let tmp = tempdir();
    let parent = tmp.path().join("fdl.yml");
    let child = tmp.path().join("fdl.ci.yml");
    std::fs::write(&parent, "a: 1\n").unwrap();
    std::fs::write(&child, "inherit-from: fdl.yml\nb: 2\n").unwrap();
    let chain = resolve_chain(&child).unwrap();
    assert_eq!(chain.len(), 2);
    // First layer is the parent (deepest), second is the child.
    assert_eq!(chain[0].0, canon(&parent));
    assert_eq!(chain[1].0, canon(&child));
    // inherit-from must not appear in the returned values.
    for (_, v) in &chain {
        if let Value::Mapping(m) = v {
            assert!(!m.contains_key(Value::String("inherit-from".to_string())));
        }
    }
}

#[test]
fn resolve_chain_three_level_ordering() {
    // c inherits from b, b inherits from a. Merge order must be [a, b, c].
    let tmp = tempdir();
    let a = tmp.path().join("a.yml");
    let b = tmp.path().join("b.yml");
    let c = tmp.path().join("c.yml");
    std::fs::write(&a, "x: from-a\n").unwrap();
    std::fs::write(&b, "inherit-from: a.yml\ny: from-b\n").unwrap();
    std::fs::write(&c, "inherit-from: b.yml\nz: from-c\n").unwrap();
    let chain = resolve_chain(&c).unwrap();
    let paths: Vec<PathBuf> = chain.iter().map(|(p, _)| p.clone()).collect();
    assert_eq!(paths, vec![canon(&a), canon(&b), canon(&c)]);
}

#[test]
fn resolve_chain_relative_paths_resolve_from_declaring_file() {
    // Declaring file sits one dir down; inherit-from uses `../base.yml`.
    let tmp = tempdir();
    let base = tmp.path().join("base.yml");
    let nested_dir = tmp.path().join("nested");
    std::fs::create_dir_all(&nested_dir).unwrap();
    let child = nested_dir.join("child.yml");
    std::fs::write(&base, "shared: true\n").unwrap();
    std::fs::write(&child, "inherit-from: ../base.yml\nlocal: true\n").unwrap();
    let chain = resolve_chain(&child).unwrap();
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].0, canon(&base));
    assert_eq!(chain[1].0, canon(&child));
}

#[test]
fn resolve_chain_absolute_path_works() {
    let tmp = tempdir();
    let parent = tmp.path().join("parent.yml");
    let child = tmp.path().join("child.yml");
    std::fs::write(&parent, "a: 1\n").unwrap();
    // Use absolute path in inherit-from.
    let abs = canon(&parent);
    std::fs::write(&child, format!("inherit-from: {}\nb: 2\n", abs.display())).unwrap();
    let chain = resolve_chain(&child).unwrap();
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].0, canon(&parent));
}

/// `~/` names the invoking user's home, not a path under the
/// declaring file — the shape a global farm base under `~/.flodl`
/// needs. Proven by routing rather than by redirecting HOME (which
/// is process-global): a `~/` parent must NOT resolve relative to
/// the declaring file's directory.
#[test]
fn resolve_chain_expands_tilde_against_home() {
    let tmp = tempdir();
    let child = tmp.path().join("child.yml");
    std::fs::write(
        &child,
        "inherit-from: ~/definitely-not-here-fdl.yml\nb: 2\n",
    )
    .unwrap();
    let err = resolve_chain(&child).unwrap_err();
    // The failure must point at a HOME-anchored path, not the temp dir.
    assert!(
        !err.contains(&tmp.path().display().to_string()),
        "a ~/ parent resolved against the declaring dir: {err}"
    );
    assert!(err.contains("definitely-not-here-fdl.yml"), "got: {err}");
}

/// A scheme-shaped parent is reserved grammar: refused by name, not
/// left to fail as a nonexistent local path.
#[test]
fn resolve_chain_refuses_remote_parents_loudly() {
    let tmp = tempdir();
    let child = tmp.path().join("child.yml");
    std::fs::write(
        &child,
        "inherit-from: rsync://host:/srv/farms/b300.yml\nb: 2\n",
    )
    .unwrap();
    let err = resolve_chain(&child).unwrap_err();
    assert!(
        err.contains("remote parents are not supported"),
        "got: {err}"
    );
    assert!(
        err.contains("rsync://host:/srv/farms/b300.yml"),
        "got: {err}"
    );
}

#[test]
fn resolve_chain_self_inheritance_errors() {
    let tmp = tempdir();
    let f = tmp.path().join("fdl.yml");
    std::fs::write(&f, "inherit-from: fdl.yml\nx: 1\n").unwrap();
    let err = resolve_chain(&f).unwrap_err();
    assert!(err.contains("cycle"), "got: {err}");
    // Self-loop appears as the same path on both sides of the arrow.
    assert!(err.matches("fdl.yml").count() >= 2, "got: {err}");
}

#[test]
fn resolve_chain_two_file_cycle_errors() {
    // a inherits from b, b inherits from a — classic cycle.
    let tmp = tempdir();
    let a = tmp.path().join("a.yml");
    let b = tmp.path().join("b.yml");
    std::fs::write(&a, "inherit-from: b.yml\nx: 1\n").unwrap();
    std::fs::write(&b, "inherit-from: a.yml\ny: 2\n").unwrap();
    let err = resolve_chain(&a).unwrap_err();
    assert!(err.contains("cycle"), "got: {err}");
    assert!(err.contains("a.yml"));
    assert!(err.contains("b.yml"));
}

#[test]
fn resolve_chain_missing_parent_errors() {
    let tmp = tempdir();
    let f = tmp.path().join("fdl.yml");
    std::fs::write(&f, "inherit-from: missing.yml\nx: 1\n").unwrap();
    let err = resolve_chain(&f).unwrap_err();
    assert!(
        err.contains("cannot resolve inherit-from target"),
        "got: {err}"
    );
    assert!(err.contains("missing.yml"), "got: {err}");
}

#[test]
fn resolve_chain_non_string_inherit_errors() {
    let tmp = tempdir();
    let f = tmp.path().join("fdl.yml");
    std::fs::write(&f, "inherit-from: 42\nx: 1\n").unwrap();
    let err = resolve_chain(&f).unwrap_err();
    assert!(err.contains("must be a string path"), "got: {err}");
    assert!(err.contains("got number"), "got: {err}");
}

#[test]
fn resolve_chain_empty_string_inherit_errors() {
    let tmp = tempdir();
    let f = tmp.path().join("fdl.yml");
    std::fs::write(&f, "inherit-from: \"\"\nx: 1\n").unwrap();
    let err = resolve_chain(&f).unwrap_err();
    assert!(err.contains("non-empty"), "got: {err}");
}

#[test]
fn resolve_chain_null_inherit_ignored() {
    // Explicit `inherit-from: null` == key absent. No error, no parent.
    let tmp = tempdir();
    let f = tmp.path().join("fdl.yml");
    std::fs::write(&f, "inherit-from: ~\nx: 1\n").unwrap();
    let chain = resolve_chain(&f).unwrap();
    assert_eq!(chain.len(), 1);
}

// Tiny tempdir helper — standalone so we don't pull in the tempfile crate.
fn tempdir() -> TempDir {
    TempDir::new()
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        // Process-wide counter, NOT a timestamp: concurrent test
        // threads can construct TempDirs within one SystemTime tick,
        // and create_dir_all on the colliding path succeeds silently —
        // two tests then share (and Drop-delete) one directory.
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "flodl-overlay-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("tempdir creation");
        Self(dir)
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
