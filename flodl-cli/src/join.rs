//! `fdl join` — self-deploy this box as a dial-in worker.
//!
//! The worker-side walk-in for a discovery window (see docs/ddp/01-reference.md):
//! dials the controller's mux port, offers this host's GPUs, and — once
//! admitted — the training binary takes over in agent role (it joins,
//! then spawns and supervises this host's relay and rank children; see
//! flodl's `distributed::launcher::agent`). fdl-cli stays protocol-blind:
//! its whole job is orchestration —
//!
//!   1. resolve settings (flags over the fdl.yml `join:` block),
//!   2. prepare this box ([`crate::prepare`]): the GPU gate, the dataset
//!      source root, the node-local directories the data plane writes —
//!      all of it BEFORE the dial, because admission starts a window
//!      deadline,
//!   3. optionally bring up an ssh `-L` forward of the controller port
//!      (the guardrailed-sshd trust path: reachability = authentication),
//!   4. synthesize the agent bootstrap spec into the binary's
//!      environment (`FLODL_INTERNAL_AGENT_JSON`, hex-encoded JSON — the
//!      same envelope cluster fan-out ships),
//!   5. run + supervise the binary, and in `--persist` mode re-dial
//!      with backoff when it exits (the systemd / golden-image loop).
//!
//! The spec (which may carry the pre-shared session token) rides the
//! child's ENVIRONMENT, never argv — owner-readable via
//! `/proc/<pid>/environ` instead of world-readable via `ps`, the same
//! salt hygiene as the launcher's fan-out.

use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use crate::builtins::JoinArgs;
use crate::config::{self, SshConfig, WorkerJoin, WorkerSource, DEFAULT_CONTROLLER_PORT};
use crate::context::Context;
use crate::prepare::{self, DataSpec, Fail, Prepared, PrepareSpec, SourceSpec};
use crate::style;

/// Agent bootstrap env var — must match flodl's
/// `distributed::launcher::ENV_AGENT_JSON` (the field names in the hex
/// JSON must match `AgentSpec`; locked by `agent_spec_shape_is_the_wire_contract`).
const ENV_AGENT_JSON: &str = "FLODL_INTERNAL_AGENT_JSON";

/// Exit code for a failure retrying cannot fix: no usable GPU, a spec
/// that does not parse, a directory that cannot be created, a missing
/// binary. Distinct from 1 (a transient failure, one-shot) so a
/// fleet can act on the difference without parsing stderr:
///
/// ```ini
/// # /etc/systemd/system/flodl-join.service
/// Restart=always
/// RestartPreventExitStatus=2   # stop hot-looping a misprovisioned box
/// FailureAction=poweroff       # ... and self-deprovision it
/// ```
///
/// fdl deliberately does not power a box off itself: the decision belongs
/// to whatever owns the instance's lifecycle, and 2 is how it hears about it.
pub const EXIT_PERMANENT: i32 = 2;

/// How long the ssh forward gets to come up (auth + local bind).
const TUNNEL_READY_BUDGET: Duration = Duration::from_secs(20);

/// `--persist` re-dial backoff: floor, cap, and the attempt duration
/// past which the backoff resets to the floor (the agent ran a real
/// stint, so the next failure is a fresh incident, not a hot loop).
const BACKOFF_MIN: Duration = Duration::from_secs(5);
const BACKOFF_MAX: Duration = Duration::from_secs(60);
const BACKOFF_RESET_AFTER: Duration = Duration::from_secs(120);

/// Run `fdl join`. `bin_tail` is everything after the command line's
/// standalone `--`: the training binary's own arguments, forwarded
/// verbatim (rank children re-enter the binary with them). `None` =
/// no `--` was given (the config block's `args:` applies); a present
/// but empty tail is an explicit "no arguments".
///
/// Exit code: the agent's own exit code (one-shot); [`EXIT_PERMANENT`]
/// for a failure retrying cannot fix; 1 for a transient one. `--persist`
/// re-dials through transient failures and agent exits, and returns only
/// on a permanent one.
pub fn run(cli: &JoinArgs, bin_tail: Option<&[String]>) -> i32 {
    let (block, project_root) = match load_join_block() {
        Ok(pair) => pair,
        Err(e) => {
            crate::cli_error!("{e}");
            return EXIT_PERMANENT;
        }
    };
    let eff = match resolve_effective(
        cli,
        bin_tail,
        block,
        &crate::cluster::resolve_local_hostname(),
    ) {
        Ok(eff) => eff,
        Err(e) => {
            crate::cli_error!("{e}");
            return EXIT_PERMANENT;
        }
    };
    if eff.controller_defaulted {
        eprintln!(
            "{}",
            style::dim(&format!(
                "fdl join: no controller configured; dialing \
                 127.0.0.1:{DEFAULT_CONTROLLER_PORT} (pass an address or set \
                 `join.controller` in fdl.yml)"
            )),
        );
    }

    // A binary named as a path must exist NOW — a persist loop retrying
    // a missing path forever helps nobody. A binary built from source
    // cannot be checked here: it does not exist until the attempt has
    // fetched and compiled the tree.
    if let BinSource::Given(path) = &eff.bin {
        if !Path::new(path).is_file() {
            crate::cli_error!(
                "training binary not found: {path} — build it first and \
                 point `--bin` (or fdl.yml `join.bin`) at it, or hand this \
                 box a `--source` to build",
            );
            return EXIT_PERMANENT;
        }
    }

    // Local active libtorch (honors FDL_LIBTORCH_CASE), anchored on the
    // project root the config walk found: its lib/ rides
    // LD_LIBRARY_PATH on the child, and its variant label rides the
    // join hello. Absent (fdl running outside a project) the child env
    // is left untouched — the binary may carry an rpath or the caller's
    // environment already provides the libs.
    let libtorch = resolve_local_libtorch(project_root.as_deref());

    let mut backoff = BACKOFF_MIN;
    loop {
        let started = Instant::now();
        // What happened, phrased for the re-dial line. Every branch that
        // is not re-dialable returns from here.
        let outcome = match attempt(&eff, libtorch.as_ref()) {
            Ok(code) => {
                if !eff.persist {
                    return code;
                }
                format!("agent exited with code {code}")
            }
            Err(fail) => {
                crate::cli_error!("{}", fail.message());
                if fail.is_permanent() {
                    // The whole point of the class: a box that cannot be
                    // fixed by waiting must stop, not hot-loop.
                    eprintln!(
                        "{}",
                        style::dim(&format!(
                            "fdl join: not re-dialing — retrying cannot \
                             fix this (exit {EXIT_PERMANENT})"
                        )),
                    );
                    return EXIT_PERMANENT;
                }
                if !eff.persist {
                    return 1;
                }
                "attempt failed".to_string()
            }
        };
        if started.elapsed() > BACKOFF_RESET_AFTER {
            backoff = BACKOFF_MIN;
        }
        eprintln!(
            "fdl join: {outcome} after {}s; re-dialing in {}s (--persist)",
            started.elapsed().as_secs(),
            backoff.as_secs(),
        );
        std::thread::sleep(backoff);
        backoff = (backoff * 2).min(BACKOFF_MAX);
    }
}

// ---------------------------------------------------------------------------
// Settings resolution
// ---------------------------------------------------------------------------

/// The fully resolved join recipe: flags merged over the fdl.yml
/// `join:` block, every default applied.
#[derive(Debug)]
struct Effective {
    /// Controller mux address. Under `ssh` this is the address as seen
    /// FROM the ssh host (the `-L` forward's far end).
    controller_host: String,
    controller_port: u16,
    /// True when neither flags nor config named a controller and the
    /// loopback convention default applied (worth a stderr note).
    controller_defaulted: bool,
    /// Tunnel hop; `None` = direct dial.
    ssh: Option<SshConfig>,
    /// Pre-shared session credential (hex). `None` = open admission.
    token: Option<String>,
    /// How this box gets its training binary: a path to run as given, or
    /// a source to build. Exactly one, checked at resolution.
    bin: BinSource,
    /// libtorch variant to acquire; `None` keeps this box's active one.
    libtorch_spec: Option<String>,
    /// Logical host name in the join hello.
    host: String,
    /// Explicit CUDA device ids; `None` = all GPUs on this host.
    devices: Option<Vec<u8>>,
    persist: bool,
    /// The binary's own arguments.
    bin_args: Vec<String>,
    /// Dataset source root on this box (the mountpoint when
    /// `data_source` is set); `None` ships nothing to the ranks.
    data_path: Option<String>,
    /// Transport that establishes the source root, `<scheme>://<target>`.
    data_source: Option<String>,
}

/// Where this box's training binary comes from. `source:` and `bin:` are
/// mutually exclusive because they answer the same question, and keeping
/// the given-binary case a separate variant rather than a third kind of
/// source spec is what keeps an artifact-versus-source distinction out of
/// the source grammar.
#[derive(Debug, PartialEq, Eq)]
enum BinSource {
    /// A path on this box, run as given.
    Given(String),
    /// Fetched and built here.
    Build(WorkerSource),
}

impl Effective {
    /// Everything [`crate::prepare`] has to settle. The tunnel block
    /// rides along to both artifact specs: the data host and the source
    /// host are the controller box in the shape this exists for, so its
    /// key and options apply to them too.
    fn prepare_spec<'a>(
        &'a self,
        active_libtorch: Option<&'a (PathBuf, String)>,
    ) -> PrepareSpec<'a> {
        PrepareSpec {
            data: DataSpec {
                path: self.data_path.as_deref(),
                source: self.data_source.as_deref(),
                ssh: self.ssh.as_ref(),
            },
            libtorch: self.libtorch_spec.as_deref(),
            active_libtorch,
            source: match &self.bin {
                BinSource::Given(_) => None,
                BinSource::Build(s) => Some(SourceSpec {
                    from: &s.from,
                    cwd: s.cwd.as_deref(),
                    build: s.build.as_deref(),
                    bin: &s.bin,
                    ssh: self.ssh.as_ref(),
                }),
            },
        }
    }
}

/// Merge flags over the config block. Pure — all I/O (hostname, config
/// load) happens in the callers so this stays table-testable.
fn resolve_effective(
    cli: &JoinArgs,
    bin_tail: Option<&[String]>,
    block: Option<WorkerJoin>,
    local_hostname: &str,
) -> Result<Effective, String> {
    let block = block.unwrap_or_default();

    if cli.identity.is_some() && cli.ssh.is_none() && block.ssh.is_none() {
        return Err(
            "--identity is the tunnel's key file — it needs an ssh hop \
             (`--ssh` or fdl.yml `join.ssh`)"
                .to_string(),
        );
    }

    // Tunnel hop: `--ssh [user@]host[:port]` replaces the block's
    // target/user/port but inherits its identity_file/options (not
    // expressible in the compact form); `--identity` wins last. A block
    // without `target:` is an authoring error — ssh needs a host.
    let ssh = match (&cli.ssh, block.ssh) {
        (Some(spec), b) => {
            let mut cfg = parse_ssh_spec(spec)?;
            if let Some(b) = b {
                cfg.identity_file = b.identity_file;
                cfg.options = b.options;
            }
            Some(cfg)
        }
        (None, Some(b)) => {
            if b.target.is_none() {
                return Err(
                    "fdl.yml join.ssh needs a `target:` (the tunnel host)"
                        .to_string(),
                );
            }
            Some(b)
        }
        (None, None) => None,
    };
    let mut ssh = ssh;
    if let (Some(cfg), Some(id)) = (ssh.as_mut(), &cli.identity) {
        cfg.identity_file = Some(id.clone());
    }

    // Controller: flag > block > loopback convention. Through a tunnel
    // the loopback default is THE convention (guardrailed sshd on the
    // controller box), no note needed; a bare loopback default deserves
    // one.
    let named = cli.controller.as_ref().or(block.controller.as_ref());
    let controller_defaulted = named.is_none() && ssh.is_none();
    let (controller_host, controller_port) = match named {
        Some(spec) => parse_host_port(spec)?,
        None => ("127.0.0.1".to_string(), DEFAULT_CONTROLLER_PORT),
    };

    // The binary: a path to run, or a source to build. Flags win over
    // the block on each side, and naming both ways is an authoring error
    // rather than a precedence puzzle — a box that builds its own binary
    // and is also handed one has no defensible answer.
    let bin_path = cli.bin.clone().or(block.bin);
    let source = match (cli.source.clone(), block.source) {
        (Some(from), b) => Some(WorkerSource {
            from,
            // The compact `--source` flag carries only the transport, so
            // the rest keeps coming from the block unless its own flag
            // overrides it (same shape as `--ssh`).
            cwd: cli.source_cwd.clone().or_else(|| b.as_ref().and_then(|b| b.cwd.clone())),
            build: cli.source_build.clone().or_else(|| b.as_ref().and_then(|b| b.build.clone())),
            bin: cli
                .source_bin
                .clone()
                .or_else(|| b.as_ref().map(|b| b.bin.clone()))
                .ok_or_else(|| {
                    "a source needs the artifact it produces — pass \
                     `--source-bin <path relative to the project dir>` or \
                     set `join.source.bin` in fdl.yml"
                        .to_string()
                })?,
        }),
        (None, Some(mut b)) => {
            if let Some(cwd) = cli.source_cwd.clone() {
                b.cwd = Some(cwd);
            }
            if let Some(build) = cli.source_build.clone() {
                b.build = Some(build);
            }
            if let Some(bin) = cli.source_bin.clone() {
                b.bin = bin;
            }
            Some(b)
        }
        (None, None) => None,
    };
    // A source flag with no source to attach to is an authoring error,
    // not something to drop on the floor: the operator meant it to change
    // the run.
    if source.is_none() {
        for (flag, set) in [
            ("--source-cwd", cli.source_cwd.is_some()),
            ("--source-build", cli.source_build.is_some()),
            ("--source-bin", cli.source_bin.is_some()),
        ] {
            if set {
                return Err(format!(
                    "{flag} has no source to apply to — pass `--source \
                     <spec>` too, or set `join.source` in fdl.yml"
                ));
            }
        }
    }

    let bin = match (bin_path, source) {
        (Some(_), Some(_)) => {
            return Err(
                "`bin:` and `source:` both name this box's training binary \
                 — keep the one you mean. `bin:` runs a binary as given; \
                 `source:` fetches and builds one here"
                    .to_string(),
            );
        }
        (Some(path), None) => BinSource::Given(path),
        (None, Some(source)) => BinSource::Build(source),
        (None, None) => {
            return Err(
                "no training binary configured — pass `--bin <path>` (run \
                 it as given) or `--source <spec>` (build it here), or set \
                 `join.bin` / `join.source` in fdl.yml. The binary is the \
                 protocol: it dials, joins, and runs this host's ranks"
                    .to_string(),
            );
        }
    };

    let devices = match &cli.devices {
        Some(spec) => parse_devices(spec)?,
        None => block.devices,
    };

    // A `--` tail — even an empty one — REPLACES the block's args: the
    // args must match the run, so "explicitly none" must be sayable.
    let bin_args = match bin_tail {
        Some(tail) => tail.to_vec(),
        None => block.args,
    };

    Ok(Effective {
        controller_host,
        controller_port,
        controller_defaulted,
        ssh,
        token: cli.token.clone().or(block.token),
        bin,
        host: cli
            .host
            .clone()
            .or(block.host)
            .unwrap_or_else(|| local_hostname.to_string()),
        devices,
        persist: cli.persist || block.persist,
        bin_args,
        libtorch_spec: cli.libtorch.clone().or(block.libtorch),
        data_path: cli.data_path.clone().or(block.data_path),
        data_source: cli.data_source.clone().or(block.data_source),
    })
}

/// Load the top-level `join:` block from the PROJECT config (base
/// fdl.yml merged with the active env overlay when one is selected),
/// plus the directory it lives in (the project root — where libtorch/
/// is anchored). The walk steps over command-level fdl.ymls
/// ([`config::find_project_config`]): `fdl join` typically runs from
/// the command dir the training binary expects as cwd (e.g.
/// `ddp-bench/`), whose own fdl.yml is a command config that neither
/// carries a `join:` block nor marks the libtorch root. `Ok(None)`
/// root/block when there is no project at all — flags carry everything
/// then; a present-but-broken project config is a loud error, not a
/// silent fallback (the operator may be relying on `join.bin`).
fn load_join_block() -> Result<(Option<WorkerJoin>, Option<PathBuf>), String> {
    let cwd = std::env::current_dir()
        .map_err(|e| format!("cannot read the current directory: {e}"))?;
    let Some(config_path) = config::find_project_config(&cwd) else {
        return Ok((None, None));
    };
    let env_name = std::env::var("FDL_ENV").ok().filter(|s| !s.trim().is_empty());
    let project = config::load_project_with_env(&config_path, env_name.as_deref())
        .map_err(|e| format!("cannot load {}: {e}", config_path.display()))?;
    let root = config_path.parent().map(Path::to_path_buf);
    Ok((project.join, root))
}

/// Parse `host[:port]`, default port [`DEFAULT_CONTROLLER_PORT`] —
/// same convention as `fdl status --addr`.
fn parse_host_port(spec: &str) -> Result<(String, u16), String> {
    match spec.rsplit_once(':') {
        Some((host, port)) => {
            let port = port.parse::<u16>().map_err(|_| {
                format!("invalid controller address `{spec}` — expected host[:port]")
            })?;
            if host.is_empty() {
                return Err(format!(
                    "invalid controller address `{spec}` — expected host[:port]"
                ));
            }
            Ok((host.to_string(), port))
        }
        None => Ok((spec.to_string(), DEFAULT_CONTROLLER_PORT)),
    }
}

/// Parse the compact tunnel spec `[user@]host[:port]` into an
/// [`SshConfig`] (target/user/port only; identity/options come from
/// the config block or `--identity`).
fn parse_ssh_spec(spec: &str) -> Result<SshConfig, String> {
    let (user, rest) = match spec.split_once('@') {
        Some((u, r)) if !u.is_empty() => (Some(u.to_string()), r),
        Some(_) => {
            return Err(format!("invalid --ssh `{spec}` — empty user before `@`"));
        }
        None => (None, spec),
    };
    let (host, port) = match rest.rsplit_once(':') {
        Some((h, p)) => {
            let port = p.parse::<u16>().map_err(|_| {
                format!("invalid --ssh `{spec}` — expected [user@]host[:port]")
            })?;
            (h, Some(port))
        }
        None => (rest, None),
    };
    if host.is_empty() {
        return Err(format!("invalid --ssh `{spec}` — expected [user@]host[:port]"));
    }
    Ok(SshConfig {
        target: Some(host.to_string()),
        port,
        user,
        identity_file: None,
        options: Vec::new(),
    })
}

/// Parse `--devices`: comma-separated CUDA ids, or `all` for
/// every GPU on this host (= unset).
fn parse_devices(spec: &str) -> Result<Option<Vec<u8>>, String> {
    if spec.trim().eq_ignore_ascii_case("all") {
        return Ok(None);
    }
    spec.split(',')
        .map(|s| {
            s.trim().parse::<u8>().map_err(|_| {
                format!("invalid --devices `{spec}` — expected e.g. `0,1` or `all`")
            })
        })
        .collect::<Result<Vec<u8>, String>>()
        .map(Some)
}

// ---------------------------------------------------------------------------
// Agent spec synthesis
// ---------------------------------------------------------------------------

/// Build the hex-encoded JSON payload for [`ENV_AGENT_JSON`]. Field
/// names ARE the wire contract with flodl's `AgentSpec` deserializer;
/// optional fields are omitted (serde defaults fill them).
///
/// The prepared data path travels this way rather than in the join hello
/// because the controller has nothing to say about it: it never
/// configured this host, and only this box knows where its source root
/// actually ended up. flodl's agent inserts it into the envelope its
/// rank children read, beside the same-shaped rewrite it already does
/// for the controller address.
fn agent_spec_hex(
    eff: &Effective,
    dial: (&str, u16),
    libtorch_label: &str,
    prepared: &Prepared,
) -> String {
    let mut spec = serde_json::json!({
        "host": eff.host,
        "controller_host": dial.0,
        "controller_port": dial.1,
        "libtorch": libtorch_label,
    });
    if let Some(token) = &eff.token {
        spec["salt_hex"] = serde_json::json!(token);
    }
    if let Some(devices) = &eff.devices {
        spec["local_devices"] = serde_json::json!(devices);
    }
    if let Some(data) = &prepared.data_path {
        spec["data_path"] = serde_json::json!(data.display().to_string());
    }
    hex_encode(spec.to_string().as_bytes())
}

/// Lowercase hex — flodl's `cluster::hex_decode` counterpart.
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Active libtorch of this box: `(variant directory, variant label for
/// the hello)`. Anchored on the project root when the config walk found
/// one (a command-dir cwd's `Context::resolve` would stop a level too
/// low); the plain context fallback covers project-less setups
/// (`~/.flodl`).
///
/// The directory rather than its `lib/`: the build wants `LIBTORCH_PATH`
/// (headers included) and the child wants `lib/`, so one value serves
/// both and neither caller has to guess which it was handed.
fn resolve_local_libtorch(project_root: Option<&Path>) -> Option<(PathBuf, String)> {
    let root = match project_root {
        Some(r) => r.to_path_buf(),
        None => Context::resolve().root,
    };
    let info = crate::libtorch::detect::read_active(&root)?;
    let dir = root.join("libtorch").join(&info.path);
    dir.join("lib").is_dir().then_some((dir, info.path))
}

/// `LD_LIBRARY_PATH` for the training binary, with this box's inherited
/// value appended.
///
/// The ordering inside is not ours to choose: on ROCm the system runtime
/// must precede libtorch's own lib dir, because libtorch-rocm bundles the
/// whole userspace ROCm stack and a bundle that disagrees with the host's
/// amdkfd driver segfaults the rank at its first GPU op. Prepending
/// unconditionally, which is what this did before, is the segfault
/// configuration on an AMD box.
fn child_ld_library_path(libtorch_dir: &Path, variant: &str) -> String {
    let lib = libtorch_dir.join("lib").display().to_string();
    let vendor = crate::libtorch::detect::variant_vendor(variant);
    let value = crate::libtorch::detect::ld_library_path_value(
        vendor,
        &lib,
        &std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string()),
    );
    match std::env::var("LD_LIBRARY_PATH") {
        Ok(cur) if !cur.is_empty() => format!("{value}:{cur}"),
        _ => value,
    }
}

// ---------------------------------------------------------------------------
// One attempt: tunnel up (optional), agent run, teardown
// ---------------------------------------------------------------------------

/// One full join attempt: prepare, tunnel, dial, supervise. Returns the
/// agent's exit code, or the classed reason preparation/orchestration
/// stopped.
///
/// Preparation comes first and every attempt re-runs it: admission
/// starts a window deadline, so a mount established after the dial burns
/// it, and re-running is how `--persist` becomes a provisioning loop.
///
/// The tunnel — when one is configured — lives exactly as long as the
/// attempt: rebuilt fresh each re-dial, so a half-dead forward can never
/// outlive the run it served.
fn attempt(
    eff: &Effective,
    active_libtorch: Option<&(PathBuf, String)>,
) -> Result<i32, Fail> {
    let mut notes = Vec::new();
    let prepared = prepare::prepare(&eff.prepare_spec(active_libtorch), &mut notes);
    prepare::print_notes(&notes);
    let prepared = prepared?;

    let mut tunnel: Option<Child> = None;
    let dial: (String, u16) = match &eff.ssh {
        Some(ssh) => {
            let local_port = pick_local_port().map_err(Fail::Transient)?;
            let argv = build_tunnel_argv(
                ssh,
                local_port,
                &eff.controller_host,
                eff.controller_port,
            );
            eprintln!(
                "fdl join: opening tunnel {} -> {}:{} (local port {local_port})",
                ssh.target.as_deref().unwrap_or("?"),
                eff.controller_host,
                eff.controller_port,
            );
            let mut child = Command::new(&argv[0])
                .args(&argv[1..])
                .stdin(Stdio::null())
                .spawn()
                .map_err(|e| {
                    // No ssh on the box is a provisioning fact, not a
                    // passing condition.
                    Fail::Permanent(format!("spawn ssh tunnel: {e}"))
                })?;
            // Auth failure and an unreachable host are both possible
            // here and ssh does not let us tell them apart, so this
            // stays re-dialable: a wrong key hits the backoff cap and
            // keeps saying so, once a minute, loudly.
            if let Err(e) = wait_tunnel_ready(&mut child, local_port) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Fail::Transient(e));
            }
            tunnel = Some(child);
            ("127.0.0.1".to_string(), local_port)
        }
        None => (eff.controller_host.clone(), eff.controller_port),
    };

    // Preparation is the authority on libtorch: it either acquired a
    // variant or carried the active one through, and the label it settles
    // on is what the join hello announces.
    let libtorch_label = prepared.libtorch.as_ref().map(|(_, l)| l.as_str()).unwrap_or("");
    let spec_hex = agent_spec_hex(eff, (&dial.0, dial.1), libtorch_label, &prepared);

    // A built binary runs in the project directory inside the fetched
    // tree, which is where its own relative paths resolve; a given one
    // keeps fdl's cwd, exactly as before.
    let (bin, bin_cwd) = match (&eff.bin, &prepared.bin) {
        (BinSource::Build(_), Some(built)) => (built.bin.clone(), Some(built.cwd.clone())),
        (BinSource::Given(path), None) => (PathBuf::from(path), None),
        // Neither combination is reachable — preparation returns a binary
        // exactly when it was given a source — so say so rather than
        // quietly preferring one and hiding a wiring inversion.
        (kind, built) => {
            return Err(Fail::Permanent(format!(
                "internal: preparation and the resolved binary disagree \
                 ({}, built={})",
                match kind {
                    BinSource::Given(_) => "a path was given",
                    BinSource::Build(_) => "a source was given",
                },
                built.is_some(),
            )));
        }
    };

    let mut cmd = Command::new(&bin);
    cmd.args(&eff.bin_args)
        .env(ENV_AGENT_JSON, &spec_hex)
        // Children report under the logical roster name even when it
        // differs from `hostname` — same override fan-out applies.
        .env(crate::cluster::ENV_HOST_OVERRIDE, &eff.host)
        .stdin(Stdio::null());
    if let Some(cwd) = &bin_cwd {
        cmd.current_dir(cwd);
    }
    if let Some((dir, variant)) = &prepared.libtorch {
        cmd.env("LD_LIBRARY_PATH", child_ld_library_path(dir, variant));
    }

    // A given path was checked before the loop and a built one was just
    // written, so a spawn failure here is the file itself: not
    // executable, wrong architecture, bad interpreter.
    let status = cmd.status().map_err(|e| {
        Fail::Permanent(format!("run {}: {e}", bin.display()))
    });
    if let Some(mut t) = tunnel.take() {
        let _ = t.kill();
        let _ = t.wait();
    }
    Ok(status?.code().unwrap_or(1))
}

/// Reserve a loopback port for the tunnel's local end: bind :0, read
/// the assignment, release. The tiny bind-to-ssh race is absorbed by
/// the retry loop around each attempt.
fn pick_local_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("reserve local tunnel port: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("reserve local tunnel port: {e}"))?
        .port();
    Ok(port)
}

/// Assemble the tunnel command: `ssh -N -T` + user options first (they
/// win — OpenSSH takes the first value it sees per key) + flodl's
/// non-interactive defaults + the `-L` forward. Returned as argv for
/// testability.
fn build_tunnel_argv(
    ssh: &SshConfig,
    local_port: u16,
    controller_host: &str,
    controller_port: u16,
) -> Vec<String> {
    let mut argv: Vec<String> = vec!["ssh".into(), "-N".into(), "-T".into()];
    if let Some(warning) = crate::cluster::batchmode_override_warning(
        &ssh.options,
        ssh.target.as_deref().unwrap_or("?"),
    ) {
        eprintln!("{warning}");
    }
    for opt in &ssh.options {
        argv.push("-o".into());
        argv.push(opt.clone());
    }
    if let Some(port) = ssh.port {
        argv.push("-p".into());
        argv.push(port.to_string());
    }
    if let Some(user) = ssh.user.as_deref() {
        argv.push("-l".into());
        argv.push(user.to_string());
    }
    if let Some(id) = ssh.identity_file.as_deref() {
        argv.push("-i".into());
        argv.push(id.to_string());
    }
    // BatchMode: never hang on a prompt (a passphrase prompt inside a
    // systemd unit wedges forever). ExitOnForwardFailure: a forward the
    // remote refuses (permitopen mismatch) must kill ssh, not leave a
    // tunnel that black-holes the dial. ServerAlive: a silently dead
    // link tears the agent down instead of hanging the run.
    argv.push("-o".into());
    argv.push("BatchMode=yes".into());
    argv.push("-o".into());
    argv.push("ExitOnForwardFailure=yes".into());
    argv.push("-o".into());
    argv.push("ServerAliveInterval=30".into());
    argv.push("-L".into());
    argv.push(format!(
        "127.0.0.1:{local_port}:{controller_host}:{controller_port}"
    ));
    argv.push(ssh.target.clone().unwrap_or_default());
    argv
}

/// Block until ssh's local forward accepts (auth done, listener bound)
/// or the budget runs out. A probe connection that reaches the far mux
/// and immediately EOFs is by-design harmless (the dispatcher drops
/// pre-magic EOFs and keeps serving). An early ssh exit is the loud
/// path: auth or forward failure, with ssh's own stderr right above.
fn wait_tunnel_ready(child: &mut Child, local_port: u16) -> Result<(), String> {
    let deadline = Instant::now() + TUNNEL_READY_BUDGET;
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], local_port));
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!(
                "ssh tunnel exited ({status}) before the forward came up — \
                 see its output above (auth failure, or the remote refused \
                 the forward)"
            ));
        }
        if let Ok(probe) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) {
            drop(probe);
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "ssh tunnel did not come up within {}s (local port \
                 {local_port} never accepted)",
                TUNNEL_READY_BUDGET.as_secs(),
            ));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_flags() -> JoinArgs {
        JoinArgs {
            controller: None,
            ssh: None,
            identity: None,
            token: None,
            bin: None,
            source: None,
            source_cwd: None,
            source_build: None,
            source_bin: None,
            libtorch: None,
            host: None,
            devices: None,
            persist: false,
            data_path: None,
            data_source: None,
        }
    }

    fn full_block() -> WorkerJoin {
        WorkerJoin {
            controller: Some("10.0.0.9:9000".into()),
            ssh: Some(SshConfig {
                target: Some("bastion".into()),
                port: Some(2222),
                user: Some("join-user".into()),
                identity_file: Some("/etc/flodl/join_key".into()),
                options: vec!["StrictHostKeyChecking=accept-new".into()],
            }),
            token: Some("aa".repeat(16)),
            bin: Some("target/release/train".into()),
            source: None,
            libtorch: Some("auto".into()),
            host: Some("worker-7".into()),
            devices: Some(vec![0, 1]),
            persist: true,
            args: vec!["--model".into(), "lenet".into()],
            data_path: Some("/flodl/data".into()),
            data_source: Some("sshfs://flodl@ctrl:/srv/data".into()),
        }
    }

    /// A block that builds its binary instead of naming one.
    fn source_block() -> WorkerJoin {
        WorkerJoin {
            source: Some(WorkerSource {
                from: "rsync://exa:/home/op/rdl".into(),
                cwd: Some("ddp-bench".into()),
                build: Some("cargo build --release --bin ddp-bench".into()),
                bin: "target/release/ddp-bench".into(),
            }),
            bin: None,
            ..full_block()
        }
    }

    #[test]
    fn flags_win_over_the_config_block() {
        let cli = JoinArgs {
            controller: Some("exa".into()),
            ssh: Some("op@front:22".into()),
            identity: Some("/tmp/id".into()),
            token: Some("bb".repeat(16)),
            bin: Some("other/bin".into()),
            libtorch: Some("cu128".into()),
            host: Some("pascal".into()),
            devices: Some("2".into()),
            persist: false,
            data_path: Some("/mnt/corpus".into()),
            data_source: Some("sshfs://exa/mnt/corpus".into()),
            ..no_flags()
        };
        let tail: Vec<String> = vec!["--epochs".into(), "3".into()];
        let eff = resolve_effective(
            &cli,
            Some(&tail),
            Some(full_block()),
            "localbox",
        )
        .unwrap();
        assert_eq!(eff.controller_host, "exa");
        assert_eq!(eff.controller_port, DEFAULT_CONTROLLER_PORT);
        assert!(!eff.controller_defaulted);
        let ssh = eff.ssh.as_ref().unwrap();
        assert_eq!(ssh.target.as_deref(), Some("front"));
        assert_eq!(ssh.user.as_deref(), Some("op"));
        assert_eq!(ssh.port, Some(22));
        // Compact --ssh keeps the block's options; --identity wins last.
        assert_eq!(ssh.identity_file.as_deref(), Some("/tmp/id"));
        assert_eq!(ssh.options, vec!["StrictHostKeyChecking=accept-new".to_string()]);
        assert_eq!(eff.token.as_deref(), Some("bb".repeat(16).as_str()));
        assert_eq!(eff.bin, BinSource::Given("other/bin".into()));
        assert_eq!(eff.libtorch_spec.as_deref(), Some("cu128"));
        assert_eq!(eff.host, "pascal");
        assert_eq!(eff.devices, Some(vec![2]));
        // persist: block `true` sticks (the flag can only turn it on).
        assert!(eff.persist);
        // A `--` tail replaces the block's args.
        assert_eq!(eff.bin_args, vec!["--epochs".to_string(), "3".into()]);
        assert_eq!(eff.data_path.as_deref(), Some("/mnt/corpus"));
        assert_eq!(eff.data_source.as_deref(), Some("sshfs://exa/mnt/corpus"));
    }

    #[test]
    fn block_fills_everything_the_flags_left_unset() {
        let eff =
            resolve_effective(&no_flags(), None, Some(full_block()), "localbox").unwrap();
        assert_eq!(eff.controller_host, "10.0.0.9");
        assert_eq!(eff.controller_port, 9000);
        let ssh = eff.ssh.as_ref().unwrap();
        assert_eq!(ssh.target.as_deref(), Some("bastion"));
        assert_eq!(ssh.identity_file.as_deref(), Some("/etc/flodl/join_key"));
        assert_eq!(eff.bin, BinSource::Given("target/release/train".into()));
        assert_eq!(eff.libtorch_spec.as_deref(), Some("auto"));
        assert_eq!(eff.host, "worker-7");
        assert_eq!(eff.devices, Some(vec![0, 1]));
        assert!(eff.persist);
        assert_eq!(eff.bin_args, vec!["--model".to_string(), "lenet".into()]);
        assert_eq!(eff.data_path.as_deref(), Some("/flodl/data"));
        assert_eq!(
            eff.data_source.as_deref(),
            Some("sshfs://flodl@ctrl:/srv/data"),
        );
        // The tunnel block's key and options carry to the data mount:
        // same box, same key (which that key must permit — see
        // `prepare::DataSpec::ssh`).
        let spec = eff.prepare_spec(None);
        assert_eq!(
            spec.data.ssh.and_then(|s| s.identity_file.as_deref()),
            Some("/etc/flodl/join_key"),
        );
    }

    #[test]
    fn a_source_block_becomes_a_source_spec_carrying_the_same_key() {
        let eff = resolve_effective(&no_flags(), None, Some(source_block()), "localbox").unwrap();
        let spec = eff.prepare_spec(None);
        let source = spec.source.expect("a source block yields a source spec");
        assert_eq!(source.from, "rsync://exa:/home/op/rdl");
        assert_eq!(source.cwd, Some("ddp-bench"));
        assert_eq!(source.bin, "target/release/ddp-bench");
        // The pull runs over the same hop the tunnel uses.
        assert_eq!(
            source.ssh.and_then(|s| s.identity_file.as_deref()),
            Some("/etc/flodl/join_key"),
        );
    }

    #[test]
    fn naming_both_a_binary_and_a_source_is_a_loud_error() {
        // Not a precedence puzzle: a box handed both has no defensible
        // answer, so it must be told rather than guessed at.
        let block = WorkerJoin { bin: Some("target/release/train".into()), ..source_block() };
        let err = resolve_effective(&no_flags(), None, Some(block), "x").unwrap_err();
        assert!(err.contains("both name"), "got: {err}");
    }

    #[test]
    fn a_source_flag_keeps_the_blocks_other_source_fields() {
        // Same shape as the compact `--ssh`: the flag carries the
        // transport, the block still answers for the rest.
        let cli = JoinArgs { source: Some("file:///mnt/rdl".into()), ..no_flags() };
        let eff = resolve_effective(&cli, None, Some(source_block()), "x").unwrap();
        assert_eq!(
            eff.bin,
            BinSource::Build(WorkerSource {
                from: "file:///mnt/rdl".into(),
                cwd: Some("ddp-bench".into()),
                build: Some("cargo build --release --bin ddp-bench".into()),
                bin: "target/release/ddp-bench".into(),
            }),
        );
    }

    #[test]
    fn a_source_flag_with_no_artifact_anywhere_is_a_loud_error() {
        let cli = JoinArgs { source: Some("file:///mnt/rdl".into()), ..no_flags() };
        let err = resolve_effective(&cli, None, None, "x").unwrap_err();
        assert!(err.contains("--source-bin"), "got: {err}");
    }

    #[test]
    fn a_source_detail_flag_with_no_source_is_a_loud_error() {
        // Silently dropping it would leave the operator with a run that
        // ignored what they typed, which is the failure `--`-forwarded
        // options already taught this CLI once.
        let cli = JoinArgs {
            bin: Some("t/bin".into()),
            source_cwd: Some("ddp-bench".into()),
            ..no_flags()
        };
        let err = resolve_effective(&cli, None, None, "x").unwrap_err();
        assert!(err.contains("--source-cwd"), "got: {err}");
        assert!(err.contains("no source"), "got: {err}");
    }

    #[test]
    fn defaults_are_loopback_hostname_and_all_devices() {
        let cli = JoinArgs { bin: Some("t/bin".into()), ..no_flags() };
        let eff = resolve_effective(&cli, None, None, "localbox").unwrap();
        assert_eq!(eff.controller_host, "127.0.0.1");
        assert_eq!(eff.controller_port, DEFAULT_CONTROLLER_PORT);
        assert!(eff.controller_defaulted);
        assert!(eff.ssh.is_none());
        assert!(eff.token.is_none());
        assert_eq!(eff.host, "localbox");
        assert_eq!(eff.devices, None);
        assert!(!eff.persist);
        assert!(eff.bin_args.is_empty());
        // No data fields: prepare checks nothing and ships nothing, so
        // the training binary keeps its own default.
        assert!(eff.data_path.is_none());
        assert!(eff.data_source.is_none());
    }

    #[test]
    fn an_explicit_empty_tail_clears_the_block_args() {
        // `fdl join --` = "this run takes no arguments" — it must
        // replace the block's list, not fall back to it.
        let eff = resolve_effective(
            &no_flags(),
            Some(&[]),
            Some(full_block()),
            "localbox",
        )
        .unwrap();
        assert!(eff.bin_args.is_empty());
    }

    #[test]
    fn identity_without_an_ssh_hop_is_a_loud_error() {
        let cli = JoinArgs {
            identity: Some("/tmp/id".into()),
            bin: Some("t/bin".into()),
            ..no_flags()
        };
        let err = resolve_effective(&cli, None, None, "x").unwrap_err();
        assert!(err.contains("ssh hop"), "got: {err}");
    }

    #[test]
    fn missing_bin_is_a_loud_error() {
        let err = resolve_effective(&no_flags(), None, None, "x").unwrap_err();
        assert!(err.contains("--bin"), "got: {err}");
        assert!(err.contains("join.bin"), "got: {err}");
    }

    #[test]
    fn ssh_implies_the_loopback_controller_without_a_note() {
        let cli = JoinArgs {
            ssh: Some("join@ctrl".into()),
            bin: Some("t/bin".into()),
            ..no_flags()
        };
        let eff = resolve_effective(&cli, None, None, "x").unwrap();
        assert_eq!(eff.controller_host, "127.0.0.1");
        assert_eq!(eff.controller_port, DEFAULT_CONTROLLER_PORT);
        assert!(!eff.controller_defaulted, "tunnel loopback is the convention");
    }

    #[test]
    fn block_ssh_without_target_is_a_loud_error() {
        let block = WorkerJoin {
            ssh: Some(SshConfig::default()),
            bin: Some("t/bin".into()),
            ..WorkerJoin::default()
        };
        let err = resolve_effective(&no_flags(), None, Some(block), "x").unwrap_err();
        assert!(err.contains("target"), "got: {err}");
    }

    #[test]
    fn spec_parsers_cover_their_shapes() {
        assert_eq!(
            parse_host_port("exa").unwrap(),
            ("exa".to_string(), DEFAULT_CONTROLLER_PORT),
        );
        assert_eq!(parse_host_port("exa:9000").unwrap(), ("exa".to_string(), 9000));
        assert!(parse_host_port(":9000").is_err());
        assert!(parse_host_port("exa:banana").is_err());

        let ssh = parse_ssh_spec("join@ctrl:2222").unwrap();
        assert_eq!(ssh.target.as_deref(), Some("ctrl"));
        assert_eq!(ssh.user.as_deref(), Some("join"));
        assert_eq!(ssh.port, Some(2222));
        let bare = parse_ssh_spec("ctrl").unwrap();
        assert_eq!(bare.target.as_deref(), Some("ctrl"));
        assert_eq!(bare.user, None);
        assert_eq!(bare.port, None);
        assert!(parse_ssh_spec("@ctrl").is_err());
        assert!(parse_ssh_spec("join@").is_err());
        assert!(parse_ssh_spec("ctrl:pear").is_err());

        assert_eq!(parse_devices("0,1").unwrap(), Some(vec![0, 1]));
        assert_eq!(parse_devices(" 2 ").unwrap(), Some(vec![2]));
        assert_eq!(parse_devices("all").unwrap(), None);
        assert!(parse_devices("0,x").is_err());
    }

    /// The JSON field names are flodl's `AgentSpec` wire contract —
    /// this test IS the cross-crate compatibility lock (flodl-cli is
    /// zero-dep on flodl by design, so the shape is asserted literally;
    /// flodl's `agent_spec_round_trips_through_hex` holds the other end).
    #[test]
    fn agent_spec_shape_is_the_wire_contract() {
        let cli = JoinArgs {
            token: Some("ab".repeat(16)),
            bin: Some("t/bin".into()),
            host: Some("pascal".into()),
            devices: Some("0,1".into()),
            ..no_flags()
        };
        let eff = resolve_effective(&cli, None, None, "x").unwrap();
        let prepared = Prepared {
            data_path: Some(PathBuf::from("/flodl/data")),
            ..Prepared::default()
        };
        let hex =
            agent_spec_hex(&eff, ("127.0.0.1", 40123), "builds/sm61-sm120", &prepared);
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let spec: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(spec["host"], "pascal");
        assert_eq!(spec["controller_host"], "127.0.0.1");
        assert_eq!(spec["controller_port"], 40123);
        assert_eq!(spec["salt_hex"], "ab".repeat(16));
        assert_eq!(spec["local_devices"], serde_json::json!([0, 1]));
        assert_eq!(spec["libtorch"], "builds/sm61-sm120");
        assert_eq!(spec["data_path"], "/flodl/data");
        // Optional fields are OMITTED when unset, never null — flodl's
        // serde defaults own the fallbacks.
        let open = {
            let cli = JoinArgs { bin: Some("t/bin".into()), ..no_flags() };
            let eff = resolve_effective(&cli, None, None, "cloud-1").unwrap();
            agent_spec_hex(&eff, ("10.0.0.1", 1337), "", &Prepared::default())
        };
        let bytes: Vec<u8> = (0..open.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&open[i..i + 2], 16).unwrap())
            .collect();
        let spec: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(spec.get("salt_hex").is_none());
        assert!(spec.get("local_devices").is_none());
        assert!(spec.get("dataset_sig_hex").is_none());
        // A box that declares no source root must ship no key at all:
        // an empty string here would point every rank at the process cwd.
        assert!(spec.get("data_path").is_none());
    }

    #[test]
    fn tunnel_argv_orders_user_options_before_the_defaults() {
        let ssh = SshConfig {
            target: Some("ctrl".into()),
            port: Some(2222),
            user: Some("join-user".into()),
            identity_file: Some("/etc/flodl/join_key".into()),
            options: vec!["ServerAliveInterval=5".into()],
        };
        let argv = build_tunnel_argv(&ssh, 40123, "127.0.0.1", 1337);
        assert_eq!(argv[0], "ssh");
        assert!(argv.contains(&"-N".to_string()));
        assert!(argv.contains(&"BatchMode=yes".to_string()));
        assert!(argv.contains(&"ExitOnForwardFailure=yes".to_string()));
        // First -o value wins in OpenSSH: the user's override must
        // appear before flodl's default of the same key.
        let user_pos = argv.iter().position(|a| a == "ServerAliveInterval=5").unwrap();
        let default_pos =
            argv.iter().position(|a| a == "ServerAliveInterval=30").unwrap();
        assert!(user_pos < default_pos);
        assert!(argv.contains(&"127.0.0.1:40123:127.0.0.1:1337".to_string()));
        assert_eq!(argv.last().map(String::as_str), Some("ctrl"));
        let p = argv.iter().position(|a| a == "-p").unwrap();
        assert_eq!(argv[p + 1], "2222");
        let l = argv.iter().position(|a| a == "-l").unwrap();
        assert_eq!(argv[l + 1], "join-user");
        let i = argv.iter().position(|a| a == "-i").unwrap();
        assert_eq!(argv[i + 1], "/etc/flodl/join_key");
    }

    #[test]
    fn wait_tunnel_ready_sees_a_live_listener_and_a_dead_child() {
        // A child that exits immediately stands in for a failed ssh.
        let mut dead = Command::new("true").spawn().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let err = wait_tunnel_ready(&mut dead, 1).unwrap_err();
        assert!(err.contains("before the forward came up"), "got: {err}");

        // A live listener on the reserved port = ready, child untouched.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut slow = Command::new("sleep").arg("5").spawn().unwrap();
        assert!(wait_tunnel_ready(&mut slow, port).is_ok());
        let _ = slow.kill();
        let _ = slow.wait();
    }

    #[test]
    fn hex_encode_is_lowercase_bytewise() {
        assert_eq!(hex_encode(b"\x00\xff\x10"), "00ff10");
        assert_eq!(hex_encode(b"{}"), "7b7d");
    }
}
