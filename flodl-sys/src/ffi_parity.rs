//! FFI boundary parity check: `shim.h` (C declarations) vs `lib.rs`
//! (Rust `extern "C"` bindings).
//!
//! Each `.cpp` includes `helpers.h` -> `shim.h`, so the C++ compiler
//! verifies every shim *definition* matches its *declaration*. But the
//! Rust `extern "C"` block is an UNCHECKED promise: a signature that
//! drifts on one side only (e.g. shim `int` -> `int64_t`, binding still
//! `i32`) compiles clean on both sides and mismatches the ABI at the call
//! boundary -> undefined behavior at runtime, with nothing to catch it.
//!
//! This test parses both `extern` surfaces, normalizes each function to
//! its ABI-relevant signature, and asserts they match 1:1. It runs in the
//! ordinary (CPU) test suite — no libtorch, no libclang, no bindgen — and
//! reads the two files as text (via `include_str!`), so `#[cfg(feature =
//! "cuda")]` / `#ifdef` gating is irrelevant: both texts carry every
//! declaration regardless of build features.
//!
//! ABI normalization (const/mut and pointee type are NOT part of the C
//! ABI, so they are deliberately ignored):
//!   * a pointer of any depth N -> `"ptr{N}"` (pointee collapsed)
//!   * a depth-0 scalar -> a canonical name (`i32`, `i64`, `u64`, `f32`,
//!     `f64`, `bool`, `void`)
//!   * `FlodlTensor` is `void*` (C) / `*mut c_void` (Rust) — a pointer.
//!
//! It fails LOUDLY on any declaration it cannot parse or any type it
//! cannot map, so a new construct or a genuine drift forces a human to
//! reconcile rather than slipping through silently.

use std::collections::BTreeMap;

const SHIM_H: &str = include_str!("../shim.h");
const LIB_RS: &str = include_str!("lib.rs");

/// A function's ABI signature: canonical parameter types + return type.
type Sig = (Vec<String>, String);

// --- comment stripping ----------------------------------------------------

/// Strip `//` line comments and `/* ... */` block comments (covers `///`
/// doc comments too). Good enough for both the C header and the Rust
/// bindings, neither of which puts `//` or `/*` inside a string literal in
/// a declaration.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

// --- canonical type mapping -----------------------------------------------

/// Map a depth-0 C scalar word to its canonical token. Panics (loud) on an
/// unknown word so a newly-introduced type must be reconciled here.
fn canon_c_scalar(word: &str) -> String {
    match word {
        "void" => "void",
        "int" => "i32",
        "int32_t" => "i32",
        "int64_t" => "i64",
        "uint64_t" => "u64",
        "uint8_t" => "u8",
        "float" => "f32",
        "double" => "f64",
        "bool" => "bool",
        other => panic!(
            "ffi_parity: unmapped C scalar type `{other}` — add it to \
             canon_c_scalar (with its canonical ABI token)"
        ),
    }
    .to_string()
}

/// Normalize a C type string (name already stripped) to its ABI token.
fn canon_c(ty: &str) -> String {
    let ty = ty.replace("const", " ");
    let mut depth = ty.matches('*').count();
    let base: String = ty
        .replace('*', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    // FlodlTensor is `typedef void* FlodlTensor` — a pointer itself.
    let base = if base == "FlodlTensor" {
        depth += 1;
        "void".to_string()
    } else {
        base
    };
    if depth > 0 {
        format!("ptr{depth}")
    } else {
        canon_c_scalar(&base)
    }
}

/// Map a depth-0 Rust scalar to its canonical token.
fn canon_rust_scalar(word: &str) -> String {
    match word {
        "i32" => "i32",
        "i64" => "i64",
        "u64" => "u64",
        "u8" => "u8",
        "f32" => "f32",
        "f64" => "f64",
        "bool" => "bool",
        // c_void / c_char never appear at depth 0 (always behind a pointer).
        other => panic!(
            "ffi_parity: unmapped Rust scalar type `{other}` — add it to \
             canon_rust_scalar (with its canonical ABI token)"
        ),
    }
    .to_string()
}

/// Normalize a Rust type string to its ABI token. `()`/empty => void.
fn canon_rust(ty: &str) -> String {
    let ty = ty.trim();
    if ty.is_empty() || ty == "()" {
        return "void".to_string();
    }
    let mut depth = ty.matches("*mut").count() + ty.matches("*const").count();
    let base = ty
        .replace("*mut", " ")
        .replace("*const", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let base = if base == "FlodlTensor" {
        depth += 1;
        "c_void".to_string()
    } else {
        base
    };
    if depth > 0 {
        "ptr".to_string() + &depth.to_string()
    } else {
        canon_rust_scalar(&base)
    }
}

// --- top-level comma split (no nested parens on this FFI surface) ---------

fn split_params(params: &str) -> Vec<String> {
    let p = params.trim();
    if p.is_empty() || p == "void" {
        return Vec::new();
    }
    p.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Split a declaration `... flodl_NAME ( PARAMS )` into (name, ret, params).
/// Returns None if the statement is not a `flodl_` function declaration.
fn split_decl(stmt: &str) -> Option<(String, String, String)> {
    let name_pos = stmt.find("flodl_")?;
    let open = stmt[name_pos..].find('(')? + name_pos;
    let name = stmt[name_pos..open].trim().to_string();
    // Reject anything that isn't a bare identifier (e.g. a comment mention
    // that survived, or a macro) — a real decl has `flodl_x(` adjacency.
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    // Matching close paren (no nested parens on this surface).
    let close = stmt[open..].rfind(')')? + open;
    if close <= open {
        return None;
    }
    let params = stmt[open + 1..close].to_string();
    let ret = stmt[..name_pos].to_string();
    Some((name, ret, params))
}

// --- C parsing ------------------------------------------------------------

fn parse_c(src: &str) -> BTreeMap<String, Sig> {
    let src = strip_comments(src);
    // Drop preprocessor lines and the `extern "C" {` wrapper.
    let src: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("extern \"C\" {", " ");

    let mut map = BTreeMap::new();
    for stmt in src.split(';') {
        let Some((name, ret, params)) = split_decl(stmt) else {
            continue;
        };
        let ret_c = canon_c(ret.trim());
        let param_sigs: Vec<String> = split_params(&params)
            .iter()
            .map(|p| {
                // Strip the trailing parameter name: the last identifier.
                let ty = strip_c_param_name(p);
                canon_c(&ty)
            })
            .collect();
        if map.insert(name.clone(), (param_sigs, ret_c)).is_some() {
            panic!("ffi_parity: duplicate C declaration for `{name}` in shim.h");
        }
    }
    map
}

/// Given a C parameter like `int64_t* shape` or `FlodlTensor result`,
/// return just the type (`int64_t*` / `FlodlTensor`) by dropping the
/// trailing identifier (the parameter name).
fn strip_c_param_name(param: &str) -> String {
    let p = param.trim();
    if p == "void" {
        return p.to_string();
    }
    // Walk back over the trailing identifier (the parameter name);
    // everything before it is the type.
    let bytes = p.as_bytes();
    let mut start = bytes.len();
    while start > 0 {
        let c = bytes[start - 1];
        if c.is_ascii_alphanumeric() || c == b'_' {
            start -= 1;
        } else {
            break;
        }
    }
    // If the whole thing is one token (no separate name — shouldn't happen
    // for named params) keep it as the type.
    if start == 0 {
        return p.to_string();
    }
    p[..start].trim().to_string()
}

// --- Rust parsing ---------------------------------------------------------

fn parse_rust(src: &str) -> BTreeMap<String, Sig> {
    let src = strip_comments(src);
    // Isolate the `extern "C" { ... }` block by brace matching.
    let start = src
        .find("extern \"C\"")
        .expect("ffi_parity: no `extern \"C\"` block in lib.rs");
    let brace = src[start..]
        .find('{')
        .expect("ffi_parity: malformed extern block")
        + start;
    let mut depth = 0i32;
    let mut end = brace;
    for (i, c) in src[brace..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = brace + i;
                    break;
                }
            }
            _ => {}
        }
    }
    let block = &src[brace + 1..end];
    // Drop attribute lines (`#[cfg(...)]`, etc.).
    let block: String = block
        .lines()
        .filter(|l| !l.trim_start().starts_with("#["))
        .collect::<Vec<_>>()
        .join("\n");

    let mut map = BTreeMap::new();
    for stmt in block.split(';') {
        let stmt = stmt.trim();
        if !stmt.contains("fn flodl_") {
            continue;
        }
        let name_pos = stmt.find("flodl_").unwrap();
        let open = stmt[name_pos..]
            .find('(')
            .expect("ffi_parity: rust decl without `(`")
            + name_pos;
        let name = stmt[name_pos..open].trim().to_string();
        // Matching close paren for the arg list (no nested parens here).
        let close = stmt[open..]
            .find(')')
            .expect("ffi_parity: rust decl without `)`")
            + open;
        let params = &stmt[open + 1..close];
        let after = stmt[close + 1..].trim();
        let ret_r = if let Some(rest) = after.strip_prefix("->") {
            canon_rust(rest.trim())
        } else {
            "void".to_string()
        };
        let param_sigs: Vec<String> = split_params(params)
            .iter()
            .map(|p| {
                // `name: type`
                let ty = p.split_once(':').map(|(_, t)| t).unwrap_or(p);
                canon_rust(ty)
            })
            .collect();
        if map.insert(name.clone(), (param_sigs, ret_r)).is_some() {
            panic!("ffi_parity: duplicate Rust binding for `{name}` in lib.rs");
        }
    }
    map
}

// --- the test -------------------------------------------------------------

#[test]
fn ffi_bindings_match_shim_header() {
    let c = parse_c(SHIM_H);
    let r = parse_rust(LIB_RS);

    // Sanity: both sides parsed a substantial, comparable surface. Guards
    // against a parser regression silently seeing ~0 decls and "passing".
    assert!(
        c.len() > 300,
        "ffi_parity: parsed only {} C decls from shim.h — parser regression?",
        c.len()
    );
    assert!(
        r.len() > 300,
        "ffi_parity: parsed only {} Rust bindings from lib.rs — parser regression?",
        r.len()
    );

    // Functions declared in one side but not the other.
    let only_c: Vec<&String> = c.keys().filter(|k| !r.contains_key(*k)).collect();
    let only_r: Vec<&String> = r.keys().filter(|k| !c.contains_key(*k)).collect();
    assert!(
        only_c.is_empty(),
        "ffi_parity: declared in shim.h but MISSING a Rust binding in lib.rs: {only_c:?}"
    );
    assert!(
        only_r.is_empty(),
        "ffi_parity: bound in lib.rs but MISSING a shim.h declaration: {only_r:?}"
    );

    // Signature mismatches on functions present in both.
    let mut mismatches = Vec::new();
    for (name, c_sig) in &c {
        let r_sig = &r[name];
        if c_sig != r_sig {
            mismatches.push(format!(
                "  {name}:\n    shim.h : params={:?} ret={}\n    lib.rs : params={:?} ret={}",
                c_sig.0, c_sig.1, r_sig.0, r_sig.1
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "ffi_parity: {} signature mismatch(es) between shim.h and lib.rs \
         (ABI-normalized: pointer depth + depth-0 scalar; const/mut/pointee ignored):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
