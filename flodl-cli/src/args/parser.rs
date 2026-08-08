//! Argv tokenizer + resolver.
//!
//! This is the runtime side of `#[derive(FdlArgs)]`. The derive macro
//! builds an [`ArgsSpec`] from the struct's fields + attributes, calls
//! [`parse`] on `std::env::args()`, then destructures the resulting
//! [`ParsedArgs`] into concrete field values.
//!
//! The parser is opinionated: it does NOT implement every historical
//! convention. What it supports is documented in the test block below,
//! and that set is the contract.

use std::collections::BTreeMap;

// ── Spec (what the CLI declares) ────────────────────────────────────────

/// Declarative spec of what flags and positionals a CLI accepts. Built by
/// the `#[derive(FdlArgs)]` output at runtime, consumed by [`parse`].
#[derive(Debug, Clone, Default)]
pub struct ArgsSpec {
    pub options: Vec<OptionDecl>,
    pub positionals: Vec<PositionalDecl>,
    /// When true, unknown long/short flags are silently skipped (the
    /// token is consumed, no error is raised), and the required-
    /// positional check is disabled. Used by fdl's non-strict tail
    /// validation: the binary re-parses the argv itself, so fdl's job
    /// is to enforce declared contracts (choices on known flags,
    /// positional choices when unambiguous) without blocking
    /// pass-through flags the author chose to allow.
    ///
    /// Defaults to false so derive binaries stay strict by default.
    pub lenient_unknowns: bool,
}

/// Declaration of a single option (long flag, optionally with short alias).
#[derive(Debug, Clone)]
pub struct OptionDecl {
    /// Long name (without `--` prefix).
    pub long: String,
    /// Single-character short alias (without `-` prefix).
    pub short: Option<char>,
    /// True for value-carrying options; false for presence-only flags (bool).
    pub takes_value: bool,
    /// True if bare `--foo` is legal (field type is bool, or a default is
    /// declared for an `Option<T>`). Ignored when `takes_value = false`.
    pub allows_bare: bool,
    /// True for list-typed options (`Vec<T>`): multiple occurrences and
    /// comma-separated values accumulate.
    pub repeatable: bool,
    /// Restrict values to this set (validated at parse time).
    pub choices: Option<Vec<String>>,
}

/// Declaration of a single positional argument.
#[derive(Debug, Clone)]
pub struct PositionalDecl {
    /// Field name (used in error messages).
    pub name: String,
    /// When true, absence is a parse error.
    pub required: bool,
    /// When true, consumes all remaining positionals; must be the last decl.
    pub variadic: bool,
    /// Restrict values to this set.
    pub choices: Option<Vec<String>>,
}

// ── Output (what was passed) ────────────────────────────────────────────

/// Intermediate parsed result, shaped for the derive macro's field
/// extraction. Absence is encoded by a missing map entry.
#[derive(Debug, Default)]
pub struct ParsedArgs {
    /// Keyed by option long name.
    pub options: BTreeMap<String, OptionState>,
    /// Positionals in declaration order; variadic drains the tail.
    pub positionals: Vec<String>,
}

/// What happened to a single option on the command line.
#[derive(Debug, Clone)]
pub enum OptionState {
    /// Flag was passed with no value (bare `--foo` or `-f`).
    BarePresent,
    /// Flag was passed with value(s). Length 1 for scalar, >=1 for list.
    WithValues(Vec<String>),
}

// ── Parse ───────────────────────────────────────────────────────────────

/// Parse argv against a spec. `args[0]` is the program name and is ignored.
///
/// Returns a human-readable error string on failure. The caller prints it
/// to stderr and exits with a non-zero code (see [`super::parse_or_schema`]).
pub fn parse(spec: &ArgsSpec, args: &[String]) -> Result<ParsedArgs, String> {
    let mut out = ParsedArgs::default();
    let mut i = 1usize;
    let mut stop_flags = false;

    while i < args.len() {
        let tok = &args[i];

        if !stop_flags && tok == "--" {
            stop_flags = true;
            i += 1;
            continue;
        }

        if !stop_flags && tok.starts_with("--") {
            // Long flag: `--name` or `--name=value`.
            let rest = &tok[2..];
            let (name, inline_value) = match rest.split_once('=') {
                Some((n, v)) => (n, Some(v.to_string())),
                None => (rest, None),
            };
            match find_long(spec, name) {
                Some(decl) => {
                    i = consume_flag(decl, inline_value, args, i, &mut out)?;
                }
                None if spec.lenient_unknowns => {
                    // Unknown flag tolerated: consume just this token.
                    // We deliberately don't look ahead to consume a
                    // value — fdl has no way to know whether the unknown
                    // flag takes one, and the binary will re-parse the
                    // forwarded tail authoritatively anyway.
                    i += 1;
                }
                None => return Err(unknown_long_error(spec, name)),
            }
            continue;
        }

        if !stop_flags && tok.starts_with('-') && tok.len() >= 2 {
            // Short flag: `-x`, `-xyz` (cluster), `-x=val`, `-xval` rejected.
            let rest = &tok[1..];
            if let Some((head, inline_value)) = rest.split_once('=') {
                // `-x=value` — only valid if head is a single char.
                if head.chars().count() != 1 {
                    return Err(format!(
                        "invalid short-flag syntax `{tok}`: `-x=value` requires a single-letter short"
                    ));
                }
                let c = head.chars().next().unwrap();
                match find_short(spec, c) {
                    Some(decl) => {
                        i = consume_flag(decl, Some(inline_value.to_string()), args, i, &mut out)?;
                    }
                    None if spec.lenient_unknowns => {
                        i += 1;
                    }
                    None => return Err(format!("unknown short flag `-{c}`")),
                }
                continue;
            }
            // Cluster: each char is an independent flag. Only the last
            // may take a value (consumes next arg); all before must be
            // presence-only (takes_value = false).
            let chars: Vec<char> = rest.chars().collect();
            if spec.lenient_unknowns && chars.iter().any(|c| find_short(spec, *c).is_none()) {
                // If any char in the cluster is unknown, we can't
                // safely partition the cluster (unknown `takes_value`
                // makes cluster interpretation ambiguous). Skip the
                // whole token and let the binary handle it.
                i += 1;
                continue;
            }
            for (pos, c) in chars.iter().enumerate() {
                let decl =
                    find_short(spec, *c).ok_or_else(|| format!("unknown short flag `-{c}`"))?;
                let is_last = pos == chars.len() - 1;
                if !is_last && decl.takes_value {
                    return Err(format!(
                        "short `-{c}` takes a value and cannot be clustered mid-token `{tok}`"
                    ));
                }
                if is_last {
                    i = consume_flag(decl, None, args, i, &mut out)?;
                } else {
                    record_option(&mut out, decl, None, spec)?;
                }
            }
            if chars.is_empty() {
                // bare `-` (no flag letter): treat as positional.
                out.positionals.push(tok.clone());
                i += 1;
            }
            continue;
        }

        // Positional.
        out.positionals.push(tok.clone());
        i += 1;
    }

    // Required positional check. Skipped in lenient mode: orphan unknown
    // flags may have been silently dropped, so the collected positionals
    // are an unreliable count of what the user actually wrote. The binary
    // will re-check arity authoritatively.
    if !spec.lenient_unknowns {
        let required_count = spec.positionals.iter().filter(|p| p.required).count();
        if out.positionals.len() < required_count {
            let missing = &spec.positionals[out.positionals.len()].name;
            return Err(format!("missing required argument <{missing}>"));
        }
    }

    // Positional binding + choice validation. A positional with no decl
    // to bind to (beyond the declared list, no variadic) is an error, not
    // silence: `bench -- --model lenet` lands here with `--model` as a
    // positional and the run would otherwise proceed on defaults.
    // Lenient mode keeps tolerating extras — orphan values of dropped
    // unknown flags land as positionals, and the binary re-parses the
    // tail authoritatively.
    for (idx, value) in out.positionals.iter().enumerate() {
        match positional_decl_for(spec, idx) {
            Some(d) => {
                if let Some(choices) = &d.choices
                    && !choices.iter().any(|c| c == value)
                {
                    return Err(format!(
                        "invalid value `{value}` for <{}> -- allowed: {}",
                        d.name,
                        choices.join(", ")
                    ));
                }
            }
            None if spec.lenient_unknowns => {}
            None => {
                let hint = if value.starts_with('-') {
                    "; tokens after a standalone `--` are treated as \
                     positional arguments, not options — pass options \
                     directly, without a `--` separator"
                } else {
                    ""
                };
                return Err(format!("unexpected argument `{value}`{hint}"));
            }
        }
    }

    Ok(out)
}

/// Consume one flag — given the decl and optional inline value — and
/// advance the argv cursor accordingly. Returns the new index.
/// Whether `s` should be treated as a flag when deciding if the token after a
/// value-taking option is that option's value.
///
/// A leading `-` marks a flag EXCEPT for a bare `-`/`--` and for negative
/// numbers: `--lr -0.5` must consume `-0.5` as the value, not read it as a
/// flag. This is unambiguous here because every flodl short flag is alphabetic
/// (`-v`/`-q`/`-h`/`-V`/`-y`) — a `-<number>` token can only be a value.
/// `f64::from_str` also accepts `-inf` / `-nan`, which are fine as values.
fn is_flag_like(s: &str) -> bool {
    s.starts_with('-') && s != "-" && s != "--" && s.parse::<f64>().is_err()
}

fn consume_flag(
    decl: &OptionDecl,
    inline_value: Option<String>,
    args: &[String],
    i: usize,
    out: &mut ParsedArgs,
) -> Result<usize, String> {
    if !decl.takes_value {
        // Presence-only flag: must NOT have an inline value.
        if inline_value.is_some() {
            return Err(format!("flag `--{}` takes no value", decl.long));
        }
        record_option(out, decl, None, &ArgsSpec::default())?;
        return Ok(i + 1);
    }

    // Value-taking option.
    if let Some(v) = inline_value {
        record_option(out, decl, Some(v), &ArgsSpec::default())?;
        return Ok(i + 1);
    }

    // Look at next token: if it exists and is not itself a flag, consume.
    let next_idx = i + 1;
    let next_is_flag = args.get(next_idx).map(|s| is_flag_like(s)).unwrap_or(true); // absent counts as "no value available"

    if !next_is_flag {
        let v = args[next_idx].clone();
        record_option(out, decl, Some(v), &ArgsSpec::default())?;
        return Ok(i + 2);
    }

    // No value available — bare flag. Only valid if the spec allows it.
    if !decl.allows_bare {
        return Err(format!("`--{}` requires a value", decl.long));
    }
    record_option(out, decl, None, &ArgsSpec::default())?;
    Ok(i + 1)
}

/// Record one occurrence of an option. Handles choice validation and
/// repeatable accumulation.
fn record_option(
    out: &mut ParsedArgs,
    decl: &OptionDecl,
    value: Option<String>,
    _spec: &ArgsSpec,
) -> Result<(), String> {
    // Choice validation (only applies when a value is present).
    if let (Some(v), Some(choices)) = (&value, &decl.choices) {
        for part in split_list_value(v) {
            if !choices.iter().any(|c| c == part) {
                return Err(format!(
                    "invalid value `{part}` for `--{}` -- allowed: {}",
                    decl.long,
                    choices.join(", ")
                ));
            }
        }
    }

    let key = decl.long.clone();
    match (value, decl.repeatable) {
        (None, _) => {
            // Bare flag: first occurrence wins (BarePresent).
            out.options.entry(key).or_insert(OptionState::BarePresent);
        }
        (Some(v), false) => {
            // Scalar option: last occurrence wins.
            out.options.insert(key, OptionState::WithValues(vec![v]));
        }
        (Some(v), true) => {
            // List option: accumulate, with comma-split inside each value.
            let parts: Vec<String> = split_list_value(&v).into_iter().map(String::from).collect();
            let entry = out
                .options
                .entry(key)
                .or_insert(OptionState::WithValues(Vec::new()));
            if let OptionState::WithValues(list) = entry {
                list.extend(parts);
            }
        }
    }
    Ok(())
}

/// Split a list value on commas, trimming whitespace around each piece.
/// Empty pieces are dropped (so `--tags a,,b` = `["a", "b"]`).
fn split_list_value(v: &str) -> Vec<&str> {
    v.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

fn find_long<'a>(spec: &'a ArgsSpec, name: &str) -> Option<&'a OptionDecl> {
    spec.options.iter().find(|o| o.long == name)
}

fn find_short(spec: &ArgsSpec, c: char) -> Option<&OptionDecl> {
    spec.options.iter().find(|o| o.short == Some(c))
}

fn positional_decl_for(spec: &ArgsSpec, idx: usize) -> Option<&PositionalDecl> {
    // Direct index up to the variadic; beyond that, re-use the variadic decl.
    if let Some(decl) = spec.positionals.get(idx) {
        return Some(decl);
    }
    spec.positionals.iter().rev().find(|d| d.variadic)
}

/// "did you mean" error for unknown long flags.
fn unknown_long_error(spec: &ArgsSpec, name: &str) -> String {
    let suggestion = spec
        .options
        .iter()
        .filter(|o| similar(&o.long, name))
        .map(|o| format!("--{}", o.long))
        .next();
    match suggestion {
        Some(s) => format!("unknown flag `--{name}`, did you mean `{s}`?"),
        None => format!("unknown flag `--{name}`"),
    }
}

/// "Did you mean" suggestion over a fixed candidate list: returns the
/// first candidate within edit distance 2 of `input`, if any. The
/// flag-level equivalent is folded into `unknown_long_error`; this
/// public entry is for the enum-dispatch codegen, which suggests a
/// subcommand name when the user mistypes one (`bin trian` → `train`).
pub fn suggest(candidates: &[&str], input: &str) -> Option<String> {
    candidates
        .iter()
        .find(|c| similar(c, input))
        .map(|c| (*c).to_string())
}

/// "did you mean" similarity: edit distance ≤ 2 qualifies.
///
/// Simple Levenshtein on char vectors. The input sizes here are tiny
/// (flag names), so an O(n*m) implementation is fine.
fn similar(candidate: &str, target: &str) -> bool {
    if candidate == target {
        return false;
    }
    levenshtein(candidate, target) <= 2
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flag(long: &str, short: Option<char>) -> OptionDecl {
        OptionDecl {
            long: long.into(),
            short,
            takes_value: false,
            allows_bare: true,
            repeatable: false,
            choices: None,
        }
    }

    fn value(long: &str, short: Option<char>, bare_ok: bool) -> OptionDecl {
        OptionDecl {
            long: long.into(),
            short,
            takes_value: true,
            allows_bare: bare_ok,
            repeatable: false,
            choices: None,
        }
    }

    fn list(long: &str, short: Option<char>) -> OptionDecl {
        OptionDecl {
            long: long.into(),
            short,
            takes_value: true,
            allows_bare: false,
            repeatable: true,
            choices: None,
        }
    }

    fn pos(name: &str, required: bool, variadic: bool) -> PositionalDecl {
        PositionalDecl {
            name: name.into(),
            required,
            variadic,
            choices: None,
        }
    }

    fn argv(parts: &[&str]) -> Vec<String> {
        std::iter::once("prog")
            .chain(parts.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn parses_long_flag_with_value() {
        let spec = ArgsSpec {
            options: vec![value("model", None, false)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["--model", "mlp"])).unwrap();
        match out.options.get("model") {
            Some(OptionState::WithValues(v)) => assert_eq!(v, &vec!["mlp".to_string()]),
            other => panic!("expected WithValues, got {:?}", other),
        }
    }

    #[test]
    fn is_flag_like_treats_negative_numbers_as_values() {
        // M22: negative numbers (and bare -/--) are NOT flags for the purpose
        // of value consumption; alphabetic short/long flags are.
        assert!(is_flag_like("--verbose"));
        assert!(is_flag_like("-v"));
        assert!(!is_flag_like("-0.5"));
        assert!(!is_flag_like("-5"));
        assert!(!is_flag_like("-1e-3"));
        assert!(!is_flag_like("-")); // bare dash: value/stdin sentinel
        assert!(!is_flag_like("--")); // separator
        assert!(!is_flag_like("mlp")); // plain positional value
    }

    #[test]
    fn value_flag_consumes_space_separated_negative_number() {
        // M22: `--lr -0.5` reads `-0.5` as the value, not a flag.
        let spec = ArgsSpec {
            options: vec![value("lr", None, false)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["--lr", "-0.5"])).unwrap();
        match out.options.get("lr") {
            Some(OptionState::WithValues(v)) => assert_eq!(v, &vec!["-0.5".to_string()]),
            other => panic!("expected WithValues([-0.5]), got {:?}", other),
        }
    }

    #[test]
    fn value_flag_still_errors_when_next_is_a_real_flag() {
        // A genuine flag after a value-requiring option is NOT consumed.
        let spec = ArgsSpec {
            options: vec![value("lr", None, false), flag("verbose", None)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let err = parse(&spec, &argv(&["--lr", "--verbose"])).unwrap_err();
        assert!(err.contains("requires a value"), "got: {err}");
    }

    #[test]
    fn parses_long_flag_with_equals() {
        let spec = ArgsSpec {
            options: vec![value("model", None, false)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["--model=mlp"])).unwrap();
        match out.options.get("model") {
            Some(OptionState::WithValues(v)) => assert_eq!(v, &vec!["mlp".to_string()]),
            _ => panic!("expected WithValues"),
        }
    }

    #[test]
    fn bare_flag_without_default_errors() {
        let spec = ArgsSpec {
            options: vec![value("report", None, false)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let err = parse(&spec, &argv(&["--report"])).unwrap_err();
        assert!(err.contains("requires a value"), "got: {err}");
    }

    #[test]
    fn bare_flag_with_default_is_present() {
        let spec = ArgsSpec {
            options: vec![value("report", None, true)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["--report"])).unwrap();
        assert!(matches!(
            out.options.get("report"),
            Some(OptionState::BarePresent)
        ));
    }

    #[test]
    fn bool_flag_presence() {
        let spec = ArgsSpec {
            options: vec![flag("validate", None)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["--validate"])).unwrap();
        assert!(matches!(
            out.options.get("validate"),
            Some(OptionState::BarePresent)
        ));
    }

    #[test]
    fn bool_flag_rejects_value() {
        let spec = ArgsSpec {
            options: vec![flag("validate", None)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let err = parse(&spec, &argv(&["--validate=yes"])).unwrap_err();
        assert!(err.contains("takes no value"), "got: {err}");
    }

    #[test]
    fn short_flag() {
        let spec = ArgsSpec {
            options: vec![flag("verbose", Some('v'))],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["-v"])).unwrap();
        assert!(matches!(
            out.options.get("verbose"),
            Some(OptionState::BarePresent)
        ));
    }

    #[test]
    fn short_clustering_for_bool_flags() {
        let spec = ArgsSpec {
            options: vec![flag("a", Some('a')), flag("b", Some('b'))],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["-ab"])).unwrap();
        assert!(out.options.contains_key("a"));
        assert!(out.options.contains_key("b"));
    }

    #[test]
    fn short_cluster_last_may_take_value() {
        let spec = ArgsSpec {
            options: vec![flag("a", Some('a')), value("model", Some('m'), false)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["-am", "mlp"])).unwrap();
        assert!(out.options.contains_key("a"));
        match out.options.get("model") {
            Some(OptionState::WithValues(v)) => assert_eq!(v, &vec!["mlp".to_string()]),
            _ => panic!("expected model value"),
        }
    }

    #[test]
    fn list_option_accumulates_across_repeats_and_commas() {
        let spec = ArgsSpec {
            options: vec![list("tags", Some('t'))],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["--tags", "a,b", "-t", "c"])).unwrap();
        match out.options.get("tags") {
            Some(OptionState::WithValues(v)) => {
                assert_eq!(v, &vec!["a".to_string(), "b".into(), "c".into()]);
            }
            _ => panic!("expected list values"),
        }
    }

    #[test]
    fn positionals_in_order() {
        let spec = ArgsSpec {
            options: vec![],
            positionals: vec![pos("first", true, false), pos("second", false, false)],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["a", "b"])).unwrap();
        assert_eq!(out.positionals, vec!["a".to_string(), "b".into()]);
    }

    #[test]
    fn missing_required_positional_errors() {
        let spec = ArgsSpec {
            options: vec![],
            positionals: vec![pos("first", true, false)],
            ..ArgsSpec::default()
        };
        let err = parse(&spec, &argv(&[])).unwrap_err();
        assert!(err.contains("missing required argument"), "got: {err}");
    }

    #[test]
    fn variadic_positional_absorbs_tail() {
        let spec = ArgsSpec {
            options: vec![],
            positionals: vec![pos("files", false, true)],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["a", "b", "c"])).unwrap();
        assert_eq!(
            out.positionals,
            vec!["a".to_string(), "b".into(), "c".into()]
        );
    }

    #[test]
    fn double_dash_stops_flag_parsing() {
        let spec = ArgsSpec {
            options: vec![flag("verbose", None)],
            positionals: vec![pos("rest", false, true)],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["--", "--verbose", "-x"])).unwrap();
        assert!(!out.options.contains_key("verbose"));
        assert_eq!(out.positionals, vec!["--verbose".to_string(), "-x".into()]);
    }

    #[test]
    fn excess_positional_errors_loudly() {
        // No positionals declared: a stray token must error, not vanish.
        let spec = ArgsSpec {
            options: vec![value("model", None, false)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let err = parse(&spec, &argv(&["stray"])).unwrap_err();
        assert!(err.contains("unexpected argument `stray`"), "got: {err}");
    }

    #[test]
    fn dashdash_forwarded_options_error_with_hint() {
        // The `fdl bench -- --model lenet` footgun: after `--` the
        // options land as positionals; with none declared this must be
        // loud (it used to run silently on defaults).
        let spec = ArgsSpec {
            options: vec![value("model", None, false)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let err = parse(&spec, &argv(&["--", "--model", "lenet"])).unwrap_err();
        assert!(err.contains("unexpected argument `--model`"), "got: {err}");
        assert!(err.contains("without a `--` separator"), "got: {err}");
    }

    #[test]
    fn lenient_mode_tolerates_excess_positionals() {
        // Orphan values of dropped unknown flags land as positionals in
        // lenient mode; the binary re-parses authoritatively, so fdl-side
        // validation must not reject them.
        let spec = ArgsSpec {
            options: vec![],
            positionals: vec![],
            lenient_unknowns: true,
        };
        let out = parse(&spec, &argv(&["--unknown", "orphan-value"])).unwrap();
        assert_eq!(out.positionals, vec!["orphan-value".to_string()]);
    }

    #[test]
    fn unknown_flag_suggests_similar() {
        let spec = ArgsSpec {
            options: vec![value("model", None, false)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let err = parse(&spec, &argv(&["--modl", "mlp"])).unwrap_err();
        assert!(err.contains("did you mean"), "got: {err}");
    }

    #[test]
    fn choices_validated_at_parse_time() {
        let mut model = value("model", None, false);
        model.choices = Some(vec!["mlp".into(), "lenet".into()]);
        let spec = ArgsSpec {
            options: vec![model],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let err = parse(&spec, &argv(&["--model", "foobar"])).unwrap_err();
        assert!(err.contains("allowed"), "got: {err}");
    }

    #[test]
    fn bare_dash_is_positional() {
        let spec = ArgsSpec {
            options: vec![],
            positionals: vec![pos("target", true, false)],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["-"])).unwrap();
        assert_eq!(out.positionals, vec!["-".to_string()]);
    }

    #[test]
    fn scalar_last_write_wins() {
        let spec = ArgsSpec {
            options: vec![value("model", None, false)],
            positionals: vec![],
            ..ArgsSpec::default()
        };
        let out = parse(&spec, &argv(&["--model", "a", "--model", "b"])).unwrap();
        match out.options.get("model") {
            Some(OptionState::WithValues(v)) => assert_eq!(v, &vec!["b".to_string()]),
            _ => panic!("expected last-write-wins"),
        }
    }
}
