//! Export a HuggingFace-compatible model directory.
//!
//! Two input modes (mutually exclusive, exactly one required):
//!
//! - `--hub <repo_id>`: fetch `config.json` + `model.safetensors` from
//!   the HuggingFace Hub, rebuild the matching flodl graph via
//!   `AutoModel::from_pretrained_for_export`, and write the export.
//!   Pass `--head <kind>` to override the auto-dispatch on
//!   `architectures[0]` (e.g. `--head base` re-exports a pretraining
//!   checkpoint as a feature-extraction encoder).
//! - `--checkpoint <file.fdl>`: load weights from a local flodl
//!   checkpoint, rebuild the matching architecture from the sidecar
//!   `<stem>.config.json` (or an explicit `--config <file>`) and write
//!   the export.
//!
//! In both modes the output is `<out_dir>/model.safetensors` +
//! `<out_dir>/config.json`, ready for HF Python's
//! `AutoModel.from_pretrained("<out_dir>")`.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf export --hub google-bert/bert-base-uncased --out /tmp/bert-exported
//! fdl flodl-hf export --checkpoint /tmp/my.fdl --out /tmp/my-exported
//! # or, inside the dev container directly:
//! cargo run --release --example export_hf --features hub -- --hub google-bert/bert-base-uncased --out /tmp/bert-exported
//! ```
//!
//! Argv parsing + `--help` + `--fdl-schema` are handled by `flodl-cli`'s
//! [`FdlArgs`] derive, so help is identical whether the user types
//! `fdl flodl-hf export -h` (rendered by fdl from the cached schema) or
//! `fdl flodl-hf export` with missing args (rendered by this binary
//! through the same renderer).
//!
//! Supported families: bert, roberta, distilbert, xlm-roberta, albert,
//! deberta-v2 (dispatched via `AutoConfig::from_json_str` on the
//! checkpoint's `model_type`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use flodl::{Device, Graph};
use flodl_cli::{FdlArgs, FdlArgsTrait, parse_or_schema};
use flodl_hf::export::{build_for_export, export_hf_dir};
use flodl_hf::hub::HubExportHead;
use flodl_hf::models::auto::{AutoConfig, AutoModel};
use flodl_hf::safetensors_io::keys_have_pooler;

/// Export a Hub repo or local flodl checkpoint as a HuggingFace-compatible directory (model.safetensors + config.json) using flodl's own writer. Auto-detects family (bert/roberta/distilbert/xlm-roberta/albert/deberta-v2).
#[derive(FdlArgs, Debug)]
struct ExportArgs {
    /// HuggingFace repo id to fetch (e.g. `google-bert/bert-base-uncased`).
    /// Mutex with `--checkpoint`; exactly one is required.
    #[option]
    hub: Option<String>,
    /// Local flodl `.fdl` checkpoint to load. Reads the matching
    /// architecture from the sidecar `<stem>.config.json` (or `--config`).
    /// Mutex with `--hub`; exactly one is required.
    #[option]
    checkpoint: Option<String>,
    /// Output directory; written as
    /// `<out>/model.safetensors` + `<out>/config.json`. Required.
    #[option]
    out: Option<String>,
    /// Override the source `config.json` (checkpoint mode only).
    /// Defaults to reading the sidecar next to `--checkpoint`.
    #[option]
    config: Option<String>,
    /// Overwrite an existing `<out>` directory's `model.safetensors` /
    /// `config.json` without prompting.
    #[option]
    force: bool,
    /// Also write the loaded source config verbatim to
    /// `<out>/config.source.json` alongside the canonical `config.json`
    /// (research / replication provenance — preserves fields the
    /// canonical `to_json_str` normalises away). Checkpoint mode only.
    #[option]
    preserve_source_config: bool,
    /// Force a specific head class instead of dispatching on the
    /// repo's `architectures[0]`. Hub mode only. Values:
    /// `auto` (default) | `base` | `seqcls` | `tokcls` | `qa` | `mlm`.
    /// `base` re-exports the bare backbone even when the upstream
    /// config advertises a head, handy for treating a pretraining
    /// checkpoint as a feature-extraction encoder.
    #[option]
    head: Option<String>,
}

/// Anchor a relative path against `FDL_PROJECT_ROOT` when set.
///
/// `fdl` injects this env var inside docker-compose-managed services so
/// argv paths resolve from the host shell's invocation root regardless
/// of the container's `working_dir`. Absolute paths and host-side runs
/// are unaffected.
fn resolve_path(arg: &str) -> PathBuf {
    let p = Path::new(arg);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    if let Some(root) = std::env::var_os("FDL_PROJECT_ROOT") {
        return PathBuf::from(root).join(p);
    }
    p.to_path_buf()
}

fn main() -> ExitCode {
    let cli: ExportArgs = parse_or_schema();
    match dispatch(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(DispatchError::Usage(msg)) => {
            // Mirror parse_or_schema_from's error UX: print the message
            // followed by the rendered help so the user sees BOTH the
            // specific complaint and the full flag list.
            eprintln!("error: {msg}");
            eprintln!();
            eprintln!("{}", ExportArgs::render_help());
            ExitCode::from(2)
        }
        Err(DispatchError::Runtime(msg)) => {
            // Hub fetch failures, checkpoint IO, family-build errors —
            // the help text wouldn't add value, so just surface the
            // error.
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// Two-tone error type to keep the help-on-usage-error UX surgical:
/// only render the full help when the user's argv shape is wrong, not
/// for every runtime failure mid-export.
enum DispatchError {
    /// Argv-shape problem (missing flag, mutex violation, ...).
    Usage(String),
    /// Runtime failure (network, IO, parse, model build, ...).
    Runtime(String),
}

impl From<flodl::TensorError> for DispatchError {
    fn from(e: flodl::TensorError) -> Self {
        DispatchError::Runtime(format!("{e}"))
    }
}

fn dispatch(cli: &ExportArgs) -> Result<(), DispatchError> {
    // Mutex enforcement: exactly one of --hub / --checkpoint required.
    match (cli.hub.is_some(), cli.checkpoint.is_some()) {
        (true, true) => {
            return Err(DispatchError::Usage(
                "--hub and --checkpoint are mutually exclusive; pass exactly one.".into(),
            ));
        }
        (false, false) => {
            return Err(DispatchError::Usage(
                "missing required input: pass --hub <repo_id> or --checkpoint <file.fdl>.".into(),
            ));
        }
        _ => {}
    }

    if cli.preserve_source_config && cli.hub.is_some() {
        return Err(DispatchError::Usage(
            "--preserve-source-config requires --checkpoint (Hub mode regenerates config from to_json_str)."
                .into(),
        ));
    }
    if cli.config.is_some() && cli.hub.is_some() {
        return Err(DispatchError::Usage(
            "--config is only meaningful with --checkpoint.".into(),
        ));
    }
    if cli.head.is_some() && cli.checkpoint.is_some() {
        return Err(DispatchError::Usage(
            "--head is Hub-mode only (checkpoint mode reads the architecture from the sidecar config)."
                .into(),
        ));
    }
    let head_override = match cli.head.as_deref() {
        None | Some("auto") => None,
        Some(other) => {
            Some(HubExportHead::parse(other).map_err(|e| DispatchError::Usage(e.to_string()))?)
        }
    };

    let out_arg = cli
        .out
        .as_deref()
        .ok_or_else(|| DispatchError::Usage("missing required --out <dir>.".into()))?;
    let out_dir = resolve_path(out_arg);
    if !cli.force {
        let model_path = out_dir.join("model.safetensors");
        let config_path = out_dir.join("config.json");
        let source_path = out_dir.join("config.source.json");
        let preserve_check = cli.preserve_source_config && source_path.exists();
        if model_path.exists() || config_path.exists() || preserve_check {
            return Err(DispatchError::Runtime(format!(
                "{} already contains model.safetensors / config.json (or config.source.json under --preserve-source-config). Pass --force to overwrite.",
                out_dir.display(),
            )));
        }
    }

    if let Some(repo_id) = cli.hub.as_deref() {
        run_hub(repo_id, &out_dir, head_override)?;
    } else if let Some(checkpoint_path) = cli.checkpoint.as_deref() {
        run_checkpoint(
            checkpoint_path,
            cli.config.as_deref(),
            &out_dir,
            cli.preserve_source_config,
        )?;
    }

    Ok(())
}

/// Hub mode: fetch and rebuild via `AutoModel::from_pretrained_for_export`
/// (or the explicit-head variant when `head_override` is set).
fn run_hub(
    repo_id: &str,
    out_dir: &Path,
    head_override: Option<HubExportHead>,
) -> flodl::Result<()> {
    eprintln!("fetching config.json for {repo_id} ...");
    let config = AutoConfig::from_pretrained(repo_id)?;
    eprintln!("detected family: {}", config.model_type());

    eprintln!("loading weights for {repo_id} ...");
    let graph = match head_override {
        Some(head) => {
            eprintln!("forcing head class: {head:?} (overrides architectures[0])");
            AutoModel::from_pretrained_for_export_with_head(repo_id, head)?
        }
        None => AutoModel::from_pretrained_for_export(repo_id)?,
    };

    // Use the graph's source_config (already set by
    // `from_pretrained_for_export` with `architectures` normalised to
    // the base class — `google-bert/bert-base-uncased` advertises
    // `["BertForMaskedLM"]`, but the loader builds the base backbone
    // and drops the head; the normalised config reflects what was
    // actually built). Falling back to `config.to_json_str()` would
    // re-emit the pre-normalised head class and confuse downstream
    // `AutoModelFor*` consumers (they'd look for head keys that aren't
    // there). Fall through is defensive — `from_pretrained_for_export`
    // sets it unconditionally on every supported family.
    let canonical = graph
        .source_config()
        .unwrap_or_else(|| config.to_json_str());

    // Stamp the source repo into config.json so `verify-export` can
    // recover it without an explicit `--hub-source` flag. Canonical
    // `to_json_str()` doesn't emit `_name_or_path`, so this flodl-
    // specific field is the only path back to the source after export.
    // HF's `from_pretrained` ignores unknown top-level keys, so this is
    // forward-compatible with AutoConfig consumers.
    let stamped = inject_source_repo(&canonical, repo_id)?;

    eprintln!("exporting to {} ...", out_dir.display());
    export_hf_dir(&graph, &stamped, out_dir)?;

    println!(
        "exported {repo_id} → {}\n  model.safetensors + config.json ready for AutoModel.from_pretrained",
        out_dir.display(),
    );
    Ok(())
}

/// Insert `flodl_source_repo: <repo_id>` into the canonical config
/// JSON so the matching `verify-export` invocation can recover the
/// Hub source without an explicit flag.
fn inject_source_repo(canonical: &str, repo_id: &str) -> flodl::Result<String> {
    let mut v: serde_json::Value = serde_json::from_str(canonical).map_err(|e| {
        flodl::TensorError::new(&format!("inject_source_repo: parse canonical config: {e}"))
    })?;
    let obj = v.as_object_mut().ok_or_else(|| {
        flodl::TensorError::new("inject_source_repo: canonical config is not a JSON object")
    })?;
    obj.insert(
        "flodl_source_repo".into(),
        serde_json::Value::String(repo_id.to_string()),
    );
    serde_json::to_string_pretty(&v).map_err(|e| {
        flodl::TensorError::new(&format!(
            "inject_source_repo: re-serialize canonical config: {e}"
        ))
    })
}

/// Checkpoint mode: read sidecar (or --config), build matching graph,
/// load `.fdl`, write export.
fn run_checkpoint(
    checkpoint_path: &str,
    config_override: Option<&str>,
    out_dir: &Path,
    preserve_source_config: bool,
) -> flodl::Result<()> {
    let checkpoint_path = resolve_path(checkpoint_path);
    let checkpoint_str = checkpoint_path.to_string_lossy();

    // Resolve config source: --config wins; fall back to sidecar.
    let config_str = if let Some(cfg) = config_override {
        let cfg_path = resolve_path(cfg);
        eprintln!("reading config from {} ...", cfg_path.display());
        std::fs::read_to_string(&cfg_path).map_err(|e| {
            flodl::TensorError::new(&format!("cannot read --config {}: {e}", cfg_path.display()))
        })?
    } else {
        let sidecar = sidecar_for(&checkpoint_path);
        if !sidecar.exists() {
            return Err(flodl::TensorError::new(&format!(
                "no sidecar config at {}; pass --config <file> to override (or save the checkpoint via flodl-hf so the sidecar is emitted automatically)",
                sidecar.display(),
            )));
        }
        eprintln!("reading sidecar from {} ...", sidecar.display());
        std::fs::read_to_string(&sidecar).map_err(|e| {
            flodl::TensorError::new(&format!("cannot read sidecar {}: {e}", sidecar.display()))
        })?
    };

    let config = AutoConfig::from_json_str(&config_str)?;
    eprintln!("detected family: {}", config.model_type());

    // Peek the checkpoint's parameter names to decide pooler-presence.
    let keys = flodl::checkpoint_keys(&checkpoint_str)?;
    let has_pooler = keys_have_pooler(&keys);
    eprintln!(
        "checkpoint declares {} keys, with_pooler={has_pooler}",
        keys.len()
    );

    let graph: Graph = build_for_export(&config, has_pooler, Device::CPU)?;
    let report = graph.load_checkpoint(&checkpoint_str)?;
    eprintln!(
        "loaded {} param(s)/buffer(s); {} skipped, {} missing",
        report.loaded.len(),
        report.skipped.len(),
        report.missing.len(),
    );

    // Canonical config.json always; --preserve-source-config additionally
    // writes the verbatim source as config.source.json (research /
    // replication provenance — keeps the unique fields the canonical
    // form normalises away addressable, without breaking HF Python's
    // AutoConfig path which still reads config.json).
    let canonical_config = config.to_json_str();

    let normalized = source_only_top_level_keys(&config_str, &canonical_config);
    if !normalized.is_empty() {
        eprintln!(
            "note: {} field(s) present in source config not emitted in canonical: {}",
            normalized.len(),
            normalized.join(", "),
        );
    }

    eprintln!("exporting to {} ...", out_dir.display());
    export_hf_dir(&graph, &canonical_config, out_dir)?;

    if preserve_source_config {
        let source_path = out_dir.join("config.source.json");
        std::fs::write(&source_path, &config_str).map_err(|e| {
            flodl::TensorError::new(&format!("write {}: {e}", source_path.display(),))
        })?;
        eprintln!(
            "wrote source config to {} (canonical config.json kept for AutoConfig)",
            source_path.display(),
        );
    }

    let copied = copy_tokenizer_files(&checkpoint_path, out_dir)?;
    if copied == 0 {
        eprintln!(
            "warning: no tokenizer files matched the auto-whitelist next to {}. \
             Copy them into {} manually if HF Python needs them (tokenizer.json, \
             vocab.txt, sentencepiece.bpe.model, ...).",
            checkpoint_path.display(),
            out_dir.display(),
        );
    } else {
        eprintln!(
            "copied {copied} tokenizer file(s) into {}",
            out_dir.display()
        );
    }

    println!(
        "exported {} → {}\n  model.safetensors + config.json ready for AutoModel.from_pretrained",
        checkpoint_path.display(),
        out_dir.display(),
    );
    Ok(())
}

/// Tokenizer files HF Python needs alongside `model.safetensors` /
/// `config.json` to load a model fully. Whitelist is finite and
/// deliberately narrow — it covers every public Hub checkpoint across
/// the families flodl-hf supports without sweeping in training logs,
/// optimizer state, or other run artefacts that happen to live next to
/// the `.fdl` file.
const TOKENIZER_WHITELIST: &[&str] = &[
    // Fast tokenizer + metadata (every modern HF checkpoint ships at least the first).
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    // BERT family WordPiece vocab.
    "vocab.txt",
    // GPT/RoBERTa BPE pair.
    "vocab.json",
    "merges.txt",
    // SentencePiece (XLM-R, ALBERT, DeBERTa-v2).
    "sentencepiece.bpe.model",
    "spm.model",
];

/// Walk the checkpoint's parent directory and copy any whitelisted
/// tokenizer files into `out_dir`. Returns the count of files copied.
///
/// Non-recursive: only files directly next to the `.fdl` checkpoint
/// are considered. Files outside the whitelist (training logs,
/// optimizer state, README, …) are ignored. Returns the raw count so
/// the caller can stderr-warn when zero matched without making it an
/// error — a checkpoint dir without tokenizer files is unusual but
/// not necessarily wrong (the user may copy them in manually, or the
/// downstream consumer may not need them).
fn copy_tokenizer_files(checkpoint_path: &Path, out_dir: &Path) -> flodl::Result<usize> {
    let parent = match checkpoint_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => return Ok(0),
    };
    let mut copied = 0_usize;
    for name in TOKENIZER_WHITELIST {
        let src = parent.join(name);
        if !src.is_file() {
            continue;
        }
        let dst = out_dir.join(name);
        std::fs::copy(&src, &dst).map_err(|e| {
            flodl::TensorError::new(&format!(
                "copy tokenizer file {} -> {}: {e}",
                src.display(),
                dst.display(),
            ))
        })?;
        copied += 1;
    }
    Ok(copied)
}

/// Top-level keys present in `source_json` but absent from
/// `canonical_json` — the fields the family's `to_json_str` normalised
/// away during the canonical re-emit. Returned sorted for stable
/// output. Both inputs are parsed as objects; non-object inputs return
/// an empty list (the surrounding flow has already validated parseable
/// JSON, so this is a defensive no-op).
fn source_only_top_level_keys(source_json: &str, canonical_json: &str) -> Vec<String> {
    let src: serde_json::Value = match serde_json::from_str(source_json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let canon: serde_json::Value = match serde_json::from_str(canonical_json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let (Some(src_obj), Some(canon_obj)) = (src.as_object(), canon.as_object()) else {
        return Vec::new();
    };
    let mut out: Vec<String> = src_obj
        .keys()
        .filter(|k| !canon_obj.contains_key(k.as_str()))
        .cloned()
        .collect();
    out.sort();
    out
}

/// Resolve the sidecar config path for a checkpoint path. Mirrors
/// `flodl::Graph::save_checkpoint`'s sidecar naming (strip `.fdl` and
/// optional `.gz`, then add `.config.json`).
fn sidecar_for(checkpoint: &Path) -> PathBuf {
    let mut p = checkpoint.to_path_buf();
    if p.extension().and_then(|e| e.to_str()) == Some("gz") {
        p.set_extension("");
    }
    p.set_extension("config.json");
    p
}
