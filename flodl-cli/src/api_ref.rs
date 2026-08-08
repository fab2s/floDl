//! API reference generator: extracts the public API surface from flodl source.
//!
//! Parses Rust source files to find pub structs, constructors, methods, and
//! trait implementations. No external dependencies (string-based parsing).
//!
//! Used by the `/port` agent skill to understand what flodl offers, and by
//! anyone who wants a quick reference without building docs.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::util::system::escape_json;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// A single public function/method signature.
#[derive(Debug)]
struct FnSig {
    name: String,
    signature: String,
}

/// A public type extracted from the source.
#[derive(Debug)]
struct ApiType {
    name: String,
    category: &'static str,
    file: String,
    doc_summary: String,
    doc_examples: Vec<String>,
    constructors: Vec<FnSig>,
    methods: Vec<FnSig>,
    builder_methods: Vec<FnSig>,
    traits: Vec<String>,
}

/// Top-level API reference.
struct ApiRef {
    version: String,
    types: Vec<ApiType>,
}

// ---------------------------------------------------------------------------
// Source locator
// ---------------------------------------------------------------------------

/// Find the flodl source directory. Checks (in order):
/// 1. Explicit path from --path flag
/// 2. ./flodl/src/ (dev checkout, walk up to 5 levels)
/// 3. Cargo registry (~/.cargo/registry/src/*/flodl-*/src/)
/// 4. Cached download (`~/.flodl/api-ref-cache/<tag>/`)
/// 5. Download from latest GitHub release (cached for next time)
pub fn find_flodl_src(explicit: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        let path = PathBuf::from(p);
        if path.is_dir() {
            return Some(path);
        }
    }

    // Dev checkout: walk up from cwd looking for flodl/src/lib.rs
    let mut dir = std::env::current_dir().ok()?;
    for _ in 0..5 {
        let candidate = dir.join("flodl/src");
        if candidate.join("lib.rs").is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }

    // Cargo registry
    if let Some(home) = home_dir() {
        let registry = home.join(".cargo/registry/src");
        if registry.is_dir() {
            // Find the latest flodl version in registry
            if let Ok(entries) = fs::read_dir(&registry) {
                for index_dir in entries.flatten() {
                    if let Ok(crates) = fs::read_dir(index_dir.path()) {
                        let mut best: Option<PathBuf> = None;
                        for entry in crates.flatten() {
                            let name = entry.file_name().to_string_lossy().to_string();
                            if name.starts_with("flodl-")
                                && !name.starts_with("flodl-sys")
                                && !name.starts_with("flodl-cli")
                            {
                                let src = entry.path().join("src");
                                if src.join("lib.rs").is_file() {
                                    best = Some(src);
                                }
                            }
                        }
                        if best.is_some() {
                            return best;
                        }
                    }
                }
            }
        }
    }

    // Check cached downloads
    if let Some(tag) = fetch_latest_tag() {
        if let Some(cache) = cache_dir(&tag)
            && let Some(src) = find_src_in_cache(&cache)
        {
            return Some(src);
        }
        // Download from GitHub
        match download_source(&tag) {
            Ok(src) => return Some(src),
            Err(e) => eprintln!("warning: could not download source: {}", e),
        }
    }

    None
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

// ---------------------------------------------------------------------------
// GitHub source download
// ---------------------------------------------------------------------------

const REPO: &str = "flodl-labs/flodl";

/// Get the latest release tag from GitHub.
fn fetch_latest_tag() -> Option<String> {
    // curl -sI https://github.com/REPO/releases/latest → Location header has the tag
    let output = Command::new("curl")
        .args([
            "-sI",
            &format!("https://github.com/{}/releases/latest", REPO),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let lower = line.to_lowercase();
        if lower.starts_with("location:") {
            // https://github.com/flodl-labs/flodl/releases/tag/0.3.0
            let tag = line.rsplit('/').next()?.trim();
            if !tag.is_empty() {
                return Some(tag.to_string());
            }
        }
    }
    None
}

/// Cache directory for downloaded source: `~/.flodl/api-ref-cache/<tag>/`
fn cache_dir(tag: &str) -> Option<PathBuf> {
    let home = home_dir()?;
    let flodl_home = std::env::var("FLODL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home.join(".flodl"));
    Some(flodl_home.join("api-ref-cache").join(tag))
}

/// Download and extract flodl source from a GitHub release.
/// Returns the path to the flodl/src/ directory inside the cache.
fn download_source(tag: &str) -> Result<PathBuf, String> {
    let cache = cache_dir(tag).ok_or_else(|| "cannot determine home directory".to_string())?;

    // Check if already cached
    let src_dir = find_src_in_cache(&cache);
    if let Some(src) = src_dir {
        return Ok(src);
    }

    eprintln!("Downloading flodl {} source from GitHub...", tag);

    let zip_url = format!("https://github.com/{}/archive/refs/tags/{}.zip", REPO, tag);

    fs::create_dir_all(&cache).map_err(|e| format!("cannot create cache dir: {}", e))?;

    let zip_path = cache.join("source.zip");
    crate::util::http::download_file(&zip_url, &zip_path)?;

    eprintln!("Extracting...");
    crate::util::archive::extract_zip(&zip_path, &cache)?;

    // Clean up zip
    let _ = fs::remove_file(&zip_path);

    find_src_in_cache(&cache)
        .ok_or_else(|| "downloaded archive does not contain flodl/src/lib.rs".to_string())
}

/// Find flodl/src/lib.rs inside a cache directory.
/// GitHub archives extract to `<repo-name>-<tag>/` (e.g. `floDl-0.3.0/`).
fn find_src_in_cache(cache: &Path) -> Option<PathBuf> {
    if !cache.is_dir() {
        return None;
    }
    // Direct check
    let direct = cache.join("flodl/src");
    if direct.join("lib.rs").is_file() {
        return Some(direct);
    }
    // GitHub archive layout: cache/<reponame-tag>/flodl/src/
    if let Ok(entries) = fs::read_dir(cache) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let candidate = path.join("flodl/src");
                if candidate.join("lib.rs").is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Categorize a file path into an API category.
fn categorize(rel_path: &str) -> &'static str {
    if rel_path.contains("loss") {
        "losses"
    } else if rel_path.contains("optim") {
        "optimizers"
    } else if rel_path.contains("scheduler") {
        "schedulers"
    } else if rel_path.contains("nn/") || rel_path.starts_with("nn/") {
        "modules"
    } else if rel_path.starts_with("tensor") {
        "tensor"
    } else if rel_path.starts_with("autograd") {
        "autograd"
    } else if rel_path.starts_with("graph") {
        "graph"
    } else if rel_path.starts_with("distributed") {
        "distributed"
    } else if rel_path.starts_with("data") {
        "data"
    } else {
        "other"
    }
}

/// Extract doc comments above a pub item.
/// Returns (summary_line, code_examples).
fn extract_docs(lines: &[&str], item_line: usize) -> (String, Vec<String>) {
    // Walk backwards from the item line to find /// comments
    let mut doc_lines = Vec::new();
    let mut i = item_line.saturating_sub(1);
    loop {
        let line = lines[i].trim();
        if line.starts_with("///") {
            let text = line.trim_start_matches("///");
            // Keep one leading space if present for indentation
            let text = text.strip_prefix(' ').unwrap_or(text);
            doc_lines.push(text.to_string());
        } else if line.starts_with("#[") || line.is_empty() {
            if !doc_lines.is_empty() && line.is_empty() {
                break;
            }
        } else {
            break;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    doc_lines.reverse();

    let summary = doc_lines.first().cloned().unwrap_or_default();

    // Extract code blocks from doc comments
    let mut examples = Vec::new();
    let mut in_code = false;
    let mut current_block = String::new();

    for line in &doc_lines {
        if line.starts_with("```") {
            if in_code {
                // End of code block
                if !current_block.trim().is_empty() {
                    examples.push(current_block.trim().to_string());
                }
                current_block.clear();
                in_code = false;
            } else {
                in_code = true;
            }
        } else if in_code {
            if !current_block.is_empty() {
                current_block.push('\n');
            }
            current_block.push_str(line);
        }
    }

    (summary, examples)
}

/// Extract a function signature from a line like `pub fn new(a: i64, b: i64) -> Result<Self> {`
fn extract_fn_sig(line: &str) -> Option<String> {
    let trimmed = line.trim();
    // Find the signature between "pub fn" and the opening brace or "where"
    let start = if trimmed.contains("pub fn ") {
        trimmed.find("pub fn ")?
    } else if trimmed.contains("pub const fn ") {
        trimmed.find("pub const fn ")?
    } else {
        return None;
    };

    let sig = &trimmed[start..];
    // Trim trailing { or where
    let sig = sig.trim_end_matches('{').trim_end_matches("where").trim();
    Some(sig.to_string())
}

/// Extract a function name from a signature.
fn extract_fn_name(sig: &str) -> String {
    // "pub fn new(...)" -> "new"
    let after_fn = sig.split("fn ").nth(1).unwrap_or("");
    let name_end = after_fn.find('(').unwrap_or(after_fn.len());
    // Handle generic parameters
    let name_end = name_end.min(after_fn.find('<').unwrap_or(name_end));
    after_fn[..name_end].to_string()
}

/// Parse a single Rust source file and extract pub types and their API.
fn parse_file(src_root: &Path, path: &Path) -> Vec<ApiType> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let rel_path = path
        .strip_prefix(src_root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();

    let category = categorize(&rel_path);
    let lines: Vec<&str> = content.lines().collect();
    let mut types: BTreeMap<String, ApiType> = BTreeMap::new();

    // Pass 1: find all pub struct declarations
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if let Some(after) = trimmed.strip_prefix("pub struct ") {
            let name_end = after
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(after.len());
            let name = after[..name_end].to_string();

            if name.is_empty() || name.starts_with('_') {
                continue;
            }

            // Skip test helper types, internal types
            if name.ends_with("Inner") || name.ends_with("State") && !name.contains("Trained") {
                continue;
            }

            let (doc, examples) = extract_docs(&lines, i);

            types.insert(
                name.clone(),
                ApiType {
                    name,
                    category,
                    file: rel_path.clone(),
                    doc_summary: doc,
                    doc_examples: examples,
                    constructors: Vec::new(),
                    methods: Vec::new(),
                    builder_methods: Vec::new(),
                    traits: Vec::new(),
                },
            );
        }

        // Also capture pub enum
        if let Some(after) = trimmed.strip_prefix("pub enum ") {
            let name_end = after
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(after.len());
            let name = after[..name_end].to_string();
            if !name.is_empty() && !name.starts_with('_') {
                let (doc, examples) = extract_docs(&lines, i);
                types.insert(
                    name.clone(),
                    ApiType {
                        name,
                        category,
                        file: rel_path.clone(),
                        doc_summary: doc,
                        doc_examples: examples,
                        constructors: Vec::new(),
                        methods: Vec::new(),
                        builder_methods: Vec::new(),
                        traits: Vec::new(),
                    },
                );
            }
        }
    }

    // Pass 2: find impl blocks and extract pub methods
    let mut current_impl: Option<(String, Option<String>)> = None; // (type_name, trait_name)
    let mut brace_depth: i32 = 0;
    let mut in_impl = false;
    let mut in_test = false;

    for line in lines.iter() {
        let trimmed = line.trim();

        // Skip test modules
        if trimmed.contains("#[cfg(test)]") {
            in_test = true;
        }
        if in_test {
            if trimmed == "}" && brace_depth <= 1 {
                in_test = false;
            }
            // Count braces even in test to track depth
            for c in trimmed.chars() {
                if c == '{' {
                    brace_depth += 1;
                }
                if c == '}' {
                    brace_depth -= 1;
                }
            }
            continue;
        }

        // Detect impl blocks
        if trimmed.starts_with("impl ") || trimmed.starts_with("impl<") {
            let impl_str = trimmed.to_string();

            // Parse: "impl TypeName {" or "impl TraitName for TypeName {"
            let (type_name, trait_name) = if impl_str.contains(" for ") {
                // impl Trait for Type
                let parts: Vec<&str> = impl_str.split(" for ").collect();
                let trait_part = parts[0]
                    .trim_start_matches("impl ")
                    .trim_start_matches("impl<")
                    .split('>')
                    .next_back()
                    .unwrap_or("")
                    .trim();
                // Remove generic bounds from trait name
                let trait_name = trait_part.split('<').next().unwrap_or(trait_part).trim();
                let type_part = parts.get(1).unwrap_or(&"");
                let type_name = type_part
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("")
                    .trim();
                (type_name.to_string(), Some(trait_name.to_string()))
            } else {
                // impl Type
                let after_impl = impl_str
                    .trim_start_matches("impl<")
                    .split('>')
                    .next_back()
                    .unwrap_or(impl_str.strip_prefix("impl ").unwrap_or(&impl_str));
                let after_impl = after_impl
                    .strip_prefix("impl ")
                    .unwrap_or(after_impl.trim());
                let type_name = after_impl
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("")
                    .trim();
                (type_name.to_string(), None)
            };

            if types.contains_key(&type_name) {
                current_impl = Some((type_name, trait_name));
                in_impl = true;
            }
        }

        // Track brace depth
        for c in trimmed.chars() {
            if c == '{' {
                brace_depth += 1;
            }
            if c == '}' {
                brace_depth -= 1;
                if brace_depth <= 0 && in_impl {
                    in_impl = false;
                    current_impl = None;
                }
            }
        }

        // Extract pub fn inside impl blocks
        if in_impl
            && (trimmed.starts_with("pub fn ") || trimmed.starts_with("pub const fn "))
            && let Some((ref type_name, ref trait_name)) = current_impl
            && let Some(sig) = extract_fn_sig(trimmed)
        {
            let fn_name = extract_fn_name(&sig);
            let fn_sig = FnSig {
                name: fn_name.clone(),
                signature: sig,
            };

            if let Some(api_type) = types.get_mut(type_name) {
                // Record trait implementation
                if let Some(t) = &trait_name
                    && !api_type.traits.contains(t)
                {
                    api_type.traits.push(t.clone());
                }

                // Categorize the method
                if fn_name == "new"
                    || fn_name == "on_device"
                    || fn_name == "no_bias"
                    || fn_name == "no_bias_on_device"
                    || fn_name == "configure"
                    || fn_name == "default"
                {
                    api_type.constructors.push(fn_sig);
                } else if fn_name.starts_with("with_") || fn_name == "done" || fn_name == "build" {
                    api_type.builder_methods.push(fn_sig);
                } else {
                    api_type.methods.push(fn_sig);
                }
            }
        }
    }

    // Pass 3: collect top-level pub fns (not inside impl blocks).
    // These are common for losses, init functions, utility functions.
    let mut free_fns: Vec<FnSig> = Vec::new();
    let mut depth: i32 = 0;
    let mut in_test_block = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if trimmed.contains("#[cfg(test)]") {
            in_test_block = true;
        }

        for c in trimmed.chars() {
            if c == '{' {
                depth += 1;
            }
            if c == '}' {
                depth -= 1;
            }
        }

        if in_test_block {
            if depth <= 0 {
                in_test_block = false;
            }
            continue;
        }

        // Top-level pub fn: depth 0 (module level) or 1 (inside mod block)
        if depth <= 1
            && trimmed.starts_with("pub fn ")
            && let Some(sig) = extract_fn_sig(trimmed)
        {
            let fn_name = extract_fn_name(&sig);
            let (doc, _) = extract_docs(&lines, i);
            free_fns.push(FnSig {
                name: format!("{} -- {}", fn_name, doc),
                signature: sig,
            });
        }
    }

    if !free_fns.is_empty() {
        // Determine a good label from the file name
        let file_stem = std::path::Path::new(&rel_path)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let label = match file_stem.as_str() {
            "mod" => {
                // Use parent directory name
                std::path::Path::new(&rel_path)
                    .parent()
                    .and_then(|p| p.file_name())
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            }
            other => other.to_string(),
        };

        types.insert(
            format!("{}()", label),
            ApiType {
                name: format!("{} (functions)", label),
                category: categorize(&rel_path),
                file: rel_path,
                doc_summary: String::new(),
                doc_examples: Vec::new(),
                constructors: Vec::new(),
                methods: free_fns,
                builder_methods: Vec::new(),
                traits: Vec::new(),
            },
        );
    }

    types.into_values().collect()
}

/// Walk a source tree and parse all .rs files.
fn parse_source_tree(src_root: &Path) -> Vec<ApiType> {
    let mut all_types = Vec::new();
    walk_dir(src_root, src_root, &mut all_types);
    // Sort by category then name
    all_types.sort_by(|a, b| a.category.cmp(b.category).then(a.name.cmp(&b.name)));
    all_types
}

fn walk_dir(root: &Path, dir: &Path, types: &mut Vec<ApiType>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_dir(root, &path, types);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let mut file_types = parse_file(root, &path);
            types.append(&mut file_types);
        }
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn get_version(src_root: &Path) -> String {
    // Try crate Cargo.toml first, then workspace root
    let crate_dir = src_root.parent().unwrap_or(src_root);
    for dir in &[crate_dir, crate_dir.parent().unwrap_or(crate_dir)] {
        let cargo_toml = dir.join("Cargo.toml");
        if let Ok(content) = fs::read_to_string(cargo_toml) {
            // Look for version = "x.y.z" (not version.workspace = true)
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("version")
                    && trimmed.contains('"')
                    && !trimmed.contains("workspace")
                    && let Some(v) = trimmed.split('"').nth(1)
                {
                    return v.to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

fn print_text(api: &ApiRef) {
    println!("flodl API Reference v{}", api.version);
    println!("{}", "=".repeat(40));
    println!();

    let mut by_category: BTreeMap<&str, Vec<&ApiType>> = BTreeMap::new();
    for t in &api.types {
        by_category.entry(t.category).or_default().push(t);
    }

    for (category, types) in &by_category {
        println!("## {}", category_title(category));
        println!();

        for t in types {
            // Skip types with no public API
            if t.constructors.is_empty() && t.methods.is_empty() && t.builder_methods.is_empty() {
                continue;
            }

            print!("### {}", t.name);
            if !t.traits.is_empty() {
                print!("  (implements: {})", t.traits.join(", "));
            }
            println!();

            if !t.doc_summary.is_empty() {
                println!("  {}", t.doc_summary);
            }
            println!("  file: {}", t.file);

            if !t.constructors.is_empty() {
                println!("  constructors:");
                for f in &t.constructors {
                    println!("    {}", f.signature);
                }
            }
            if !t.builder_methods.is_empty() {
                println!("  builder:");
                for f in &t.builder_methods {
                    println!("    .{}()", f.name);
                }
            }
            if !t.methods.is_empty() {
                println!("  methods:");
                for f in &t.methods {
                    println!("    {}", f.signature);
                }
            }
            if !t.doc_examples.is_empty() {
                println!("  examples:");
                for (ei, ex) in t.doc_examples.iter().enumerate() {
                    if ei > 0 {
                        println!();
                    }
                    for line in ex.lines() {
                        println!("    {}", line);
                    }
                }
            }
            println!();
        }
    }
}

fn print_json(api: &ApiRef) {
    print!(
        "{{\"version\":\"{}\",\"types\":[",
        escape_json(&api.version)
    );

    for (i, t) in api.types.iter().enumerate() {
        if t.constructors.is_empty() && t.methods.is_empty() && t.builder_methods.is_empty() {
            continue;
        }

        if i > 0 {
            print!(",");
        }

        print!(
            "{{\"name\":\"{}\",\"category\":\"{}\",\"file\":\"{}\",\"doc\":\"{}\",",
            escape_json(&t.name),
            escape_json(t.category),
            escape_json(&t.file),
            escape_json(&t.doc_summary),
        );

        print!(
            "\"traits\":[{}],",
            t.traits
                .iter()
                .map(|s| format!("\"{}\"", escape_json(s)))
                .collect::<Vec<_>>()
                .join(",")
        );

        print!(
            "\"constructors\":[{}],",
            t.constructors
                .iter()
                .map(|f| format!(
                    "{{\"name\":\"{}\",\"sig\":\"{}\"}}",
                    escape_json(&f.name),
                    escape_json(&f.signature)
                ))
                .collect::<Vec<_>>()
                .join(",")
        );

        print!(
            "\"builder_methods\":[{}],",
            t.builder_methods
                .iter()
                .map(|f| format!("\"{}\"", escape_json(&f.name)))
                .collect::<Vec<_>>()
                .join(",")
        );

        print!(
            "\"methods\":[{}],",
            t.methods
                .iter()
                .map(|f| format!(
                    "{{\"name\":\"{}\",\"sig\":\"{}\"}}",
                    escape_json(&f.name),
                    escape_json(&f.signature)
                ))
                .collect::<Vec<_>>()
                .join(",")
        );

        print!(
            "\"examples\":[{}]",
            t.doc_examples
                .iter()
                .map(|e| format!("\"{}\"", escape_json(e)))
                .collect::<Vec<_>>()
                .join(",")
        );

        print!("}}");
    }

    println!("]}}");
}

fn category_title(cat: &str) -> &str {
    match cat {
        "modules" => "Modules (nn)",
        "losses" => "Losses",
        "optimizers" => "Optimizers",
        "schedulers" => "Schedulers",
        "tensor" => "Tensor",
        "autograd" => "Autograd",
        "graph" => "Graph",
        "distributed" => "Distributed",
        "data" => "Data",
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run(json: bool, path: Option<&str>) -> Result<(), String> {
    let src_root = find_flodl_src(path).ok_or_else(|| {
        "Could not find flodl source. Run from a flodl checkout, \
             or pass --path <flodl/src/>."
            .to_string()
    })?;

    let version = get_version(&src_root);
    let types = parse_source_tree(&src_root);

    let api = ApiRef { version, types };

    if json {
        print_json(&api);
    } else {
        print_text(&api);
    }

    Ok(())
}
