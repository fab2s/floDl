use super::authorized_keys::{
    UpsertOutcome, install_authorized_line, key_material, upsert_authorized_line,
};
use super::cloud_init::render_cloud_init;
use super::credentials::{find_token_line, recover_shape, replace_token_line, validate_label};
use super::list::{enumerate_farms, render_farm_list};
use super::publish_recipe::{
    common_ancestor, declares_gpu_features, derive_publish, flodl_path_dep, freshness_report,
    normalize, package_name,
};
use super::render::{
    authorized_keys_line, render_overlay_scaffold, render_sshd_conf, render_worker_yml,
};
use super::wizard::wizard_at;
use super::*;

fn tempdir() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "fdl-join-config-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0),
    ));
    fs::create_dir_all(&d).unwrap();
    d
}

// ── Doors and endpoints ─────────────────────────────────────────────────

#[test]
fn door_defaults_to_b_and_refuses_c_by_name() {
    assert_eq!(Door::parse(None).unwrap(), Door::B);
    assert_eq!(Door::parse(Some("a")).unwrap(), Door::A);
    assert_eq!(Door::parse(Some("nologin")).unwrap(), Door::Nologin);
    let err = Door::parse(Some("c")).unwrap_err();
    assert!(err.contains("second, source-only key"), "got: {err}");
    let err = Door::parse(Some("z")).unwrap_err();
    assert!(err.contains("unknown door"), "got: {err}");
}

#[test]
fn endpoint_parses_the_compact_spec() {
    let e = Endpoint::parse(Some("flodl-join@ctrl.example.com:2222")).unwrap();
    assert_eq!(e.user, "flodl-join");
    assert_eq!(e.host, "ctrl.example.com");
    assert_eq!(e.port, 2222);
    let e = Endpoint::parse(Some("ctrl")).unwrap();
    assert_eq!(e.host, "ctrl");
    assert_eq!(e.port, 22);
    assert!(Endpoint::parse(Some("user@:22")).is_err());
    assert!(Endpoint::parse(Some("host:notaport")).is_err());
}

#[test]
fn labels_are_filename_safe() {
    assert!(validate_label("b300").is_ok());
    assert!(validate_label("farm_a-2").is_ok());
    assert!(validate_label("").is_err());
    assert!(validate_label("a/b").is_err());
    assert!(validate_label("a b").is_err());
}

// ── Token line surgery ──────────────────────────────────────────────────

#[test]
fn token_surgery_preserves_every_other_byte() {
    let yml = "\
# a farm, mostly comments
cluster:
  controller:
    join:
      # the credential
      token: aaaabbbbccccddddaaaabbbbccccdddd
      start: manual
";
    assert_eq!(
        find_token_line(yml).as_deref(),
        Some("aaaabbbbccccddddaaaabbbbccccdddd"),
    );
    let new = replace_token_line(yml, "ffff0000ffff0000ffff0000ffff0000").unwrap();
    assert!(new.contains("token: ffff0000ffff0000ffff0000ffff0000"));
    // Everything else is byte-identical: comments, indentation, order.
    assert!(new.contains("# a farm, mostly comments"));
    assert!(new.contains("      # the credential"));
    assert!(new.contains("      start: manual"));
    // A commented token line is not a token.
    assert_eq!(find_token_line("# token: dead\n"), None);
}

// ── Manifest scanning ───────────────────────────────────────────────────

#[test]
fn manifest_scan_reads_name_path_dep_and_features() {
    let m = "\
[package]
name = \"my-train\"
version = \"0.1.0\"

[dependencies]
flodl = { path = \"../rdl/flodl\", features = [\"x\"] }
serde = \"1\"

[features]
cuda = [\"flodl/cuda\"]
rocm = [\"flodl/rocm\"]
";
    assert_eq!(package_name(m).as_deref(), Some("my-train"));
    assert_eq!(flodl_path_dep(m).as_deref(), Some("../rdl/flodl"));
    assert!(declares_gpu_features(m));

    let table_form = "\
[package]
name = \"t\"

[dependencies.flodl]
path = \"../flodl\"
";
    assert_eq!(flodl_path_dep(table_form).as_deref(), Some("../flodl"));

    let registry = "[package]\nname = \"t\"\n\n[dependencies]\nflodl = \"=0.7.0\"\n";
    assert_eq!(flodl_path_dep(registry), None);
    assert!(!declares_gpu_features(registry));
}

#[test]
fn path_dep_walks_up_to_the_dep_root() {
    // <tmp>/proj/train crate depends on <tmp>/proj/rdl/flodl: the
    // fetched root must be <tmp>/proj, with cwd pointing back down.
    let tmp = tempdir();
    let train = tmp.join("proj/train");
    fs::create_dir_all(&train).unwrap();
    fs::create_dir_all(tmp.join("proj/rdl/flodl")).unwrap();
    fs::write(
        train.join("Cargo.toml"),
        "[package]\nname = \"train\"\n\n[dependencies]\nflodl = { path = \"../rdl/flodl\" }\n",
    )
    .unwrap();
    let d = derive_publish(&train).unwrap().expect("a crate");
    assert_eq!(d.from_root, tmp.join("proj").canonicalize().unwrap());
    assert_eq!(d.cwd_rel.as_deref(), Some("train"));
    assert_eq!(d.bin, "target/release/train");
    assert!(!d.build.contains("FDL_GPU_FEATURE"), "no features declared");
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn registry_dep_ships_the_crate_dir_alone() {
    let tmp = tempdir();
    let train = tmp.join("train");
    fs::create_dir_all(&train).unwrap();
    fs::write(
        train.join("Cargo.toml"),
        "[package]\nname = \"train\"\n\n[dependencies]\nflodl = \"=0.7.0\"\n\n\
         [features]\ncuda = [\"flodl/cuda\"]\nrocm = [\"flodl/rocm\"]\n",
    )
    .unwrap();
    let d = derive_publish(&train).unwrap().expect("a crate");
    assert_eq!(d.from_root, train.canonicalize().unwrap());
    assert_eq!(d.cwd_rel, None);
    assert!(
        d.build.contains("--features \"$FDL_GPU_FEATURE\""),
        "declared vendor features must ride the recipe: {}",
        d.build,
    );
    assert!(d.bin_caveat.is_none());
    // No crate at all: a note-shaped absence, not an error.
    assert!(derive_publish(&tmp.join("nowhere")).unwrap().is_none());
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn a_workspace_above_the_crate_earns_the_bin_caveat() {
    let tmp = tempdir();
    let member = tmp.join("ws/member");
    fs::create_dir_all(&member).unwrap();
    fs::create_dir_all(tmp.join("ws/flodl")).unwrap();
    fs::write(
        tmp.join("ws/Cargo.toml"),
        "[workspace]\nmembers = [\"member\"]\n",
    )
    .unwrap();
    fs::write(
        member.join("Cargo.toml"),
        "[package]\nname = \"member\"\n\n[dependencies]\nflodl = { path = \"../flodl\" }\n",
    )
    .unwrap();
    let d = derive_publish(&member).unwrap().expect("a crate");
    let caveat = d.bin_caveat.expect("the workspace must be flagged");
    assert!(caveat.contains("WORKSPACE target/"), "got: {caveat}");
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn normalize_and_common_ancestor_do_pure_path_math() {
    assert_eq!(
        normalize(Path::new("/a/b/c/../../d")),
        PathBuf::from("/a/d"),
    );
    assert_eq!(
        common_ancestor(Path::new("/a/b/c"), Path::new("/a/b/d/e")),
        PathBuf::from("/a/b"),
    );
}

// ── Freshness ───────────────────────────────────────────────────────────

#[test]
fn freshness_flags_a_stale_lockfile() {
    let tmp = tempdir();
    fs::write(tmp.join("Cargo.lock"), "x").unwrap();
    // No lockfile case first.
    let empty = tempdir();
    assert!(freshness_report(&empty).contains("no Cargo.lock"));
    // A source file newer than the lock (mtimes are second-resolution
    // on some filesystems, so set the lock visibly old).
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(600);
    let lock = fs::File::options()
        .write(true)
        .open(tmp.join("Cargo.lock"))
        .unwrap();
    lock.set_modified(old).unwrap();
    fs::write(tmp.join("main.rs"), "fn main() {}").unwrap();
    let report = freshness_report(&tmp);
    assert!(report.contains("predates"), "got: {report}");
    let _ = fs::remove_dir_all(&tmp);
    let _ = fs::remove_dir_all(&empty);
}

// ── Renders ─────────────────────────────────────────────────────────────

fn no_flags() -> JoinConfigArgs {
    JoinConfigArgs {
        label: None,
        controller: None,
        door: None,
        crate_dir: None,
        data_path: None,
        gpu_ram_share: None,
        regen: false,
        install_key: false,
        no_install_key: false,
        authorized_keys: None,
        cloud_init: false,
        cloud_init_user: None,
        yes: false,
        list: false,
        dry_run: false,
        json: false,
    }
}

#[test]
fn cloud_init_embeds_the_artifacts_and_the_failure_taxonomy() {
    let yml = "join:\n  token: t\n  persist: true\n";
    let key = "-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n-----END OPENSSH PRIVATE KEY-----\n";
    let ci = render_cloud_init("b300", "ubuntu", Door::B, yml, key);
    assert!(ci.starts_with("#cloud-config\n"));
    assert!(ci.contains("SECRET ARTIFACT"));
    // Both payloads land indented under their write_files entries.
    assert!(
        ci.contains("      -----BEGIN OPENSSH PRIVATE KEY-----"),
        "got:\n{ci}"
    );
    assert!(ci.contains("      join:"), "got:\n{ci}");
    assert!(ci.contains("path: /home/ubuntu/.ssh/flodl-join"));
    assert!(ci.contains("permissions: \"0600\""));
    // The failure taxonomy rides the unit: re-dial transient, stop on
    // permanent, halt.
    assert!(ci.contains("Restart=always"));
    assert!(ci.contains("RestartPreventExitStatus=2"));
    assert!(ci.contains("FailureAction=poweroff"));
    assert!(ci.contains("User=ubuntu"));
    assert!(ci.contains("systemctl enable --now flodl-join.service"));
    // A halt is not a deprovision everywhere, and the file has to say so
    // where the bill keeps running.
    assert!(ci.contains("NOT the meter"), "got:\n{ci}");
}

#[test]
fn cloud_init_installs_what_the_instance_does_not_have() {
    let yml = "join:\n  token: t\n";
    let key = "-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n-----END OPENSSH PRIVATE KEY-----\n";
    let ci = render_cloud_init("b300", "ubuntu", Door::B, yml, key);
    // fdl is not on a stock cloud image, and the unit starts in the same
    // boot: fetch it first, and let a baked-in one win.
    assert!(ci.contains("command -v fdl >/dev/null ||"), "got:\n{ci}");
    assert!(ci.contains("https://flodl.dev/fdl"));
    let fdl_at = ci.find("command -v fdl").unwrap();
    let enable_at = ci.find("systemctl enable --now").unwrap();
    assert!(
        fdl_at < enable_at,
        "fdl must be installed before the unit starts"
    );
}

#[test]
fn cloud_init_provisions_only_what_the_door_reaches_for() {
    let yml = "join:\n  token: t\n";
    let key = "k\n";
    let b = render_cloud_init("b300", "ubuntu", Door::B, yml, key);
    // Door `b` fetches a source tree and builds it on the box.
    assert!(b.contains("command -v cargo >/dev/null ||"), "got:\n{b}");
    assert!(b.contains(" - build-essential\n"), "got:\n{b}");
    assert!(b.contains(" - rsync\n"));
    // Installed as the service user, or cargo cannot write its registry.
    assert!(b.contains("su -l ubuntu -c"), "got:\n{b}");
    assert!(
        b.contains("Environment=PATH=/home/ubuntu/.cargo/bin:"),
        "got:\n{b}"
    );

    // Door `a` mounts the data root instead; a missing sshfs is classed
    // permanent, which under this unit means exit 2 and a halt.
    let a = render_cloud_init("b300", "ubuntu", Door::A, yml, key);
    assert!(a.contains(" - sshfs\n"), "got:\n{a}");
    assert!(!a.contains("cargo"), "door `a` builds nothing");

    let n = render_cloud_init("b300", "ubuntu", Door::Nologin, yml, key);
    assert!(!n.contains("cargo"));
    assert!(!n.contains("sshfs"));
    assert!(n.contains(" - curl\n"), "every door still fetches fdl");
}

#[test]
fn a_root_instance_gets_root_s_actual_home() {
    let yml = "join:\n  token: t\n";
    let key = "k\n";
    let ci = render_cloud_init("b300", "root", Door::B, yml, key);
    // /home/root exists on no image, so composing the path from the name
    // alone puts the key where sshd will never look for it.
    assert!(ci.contains("path: /root/.ssh/flodl-join"), "got:\n{ci}");
    assert!(ci.contains("path: /root/training/fdl.yml"));
    assert!(ci.contains("WorkingDirectory=/root/training"));
    assert!(ci.contains("Environment=PATH=/root/.cargo/bin:"));
    assert!(!ci.contains("/home/root"), "got:\n{ci}");
}

#[test]
fn the_worker_yml_speaks_each_doors_dialect() {
    let ep = Endpoint {
        user: "flodl-join".into(),
        host: "ctrl".into(),
        port: 2222,
    };
    let cli = no_flags();

    let b = render_worker_yml("b300", &ep, "aa".repeat(16).as_str(), Door::B, &cli);
    assert!(
        b.contains("from: rsync://flodl-join@ctrl:/tree"),
        "got:\n{b}"
    );
    assert!(b.contains(&format!("token: {}", "aa".repeat(16))));
    assert!(b.contains("port: 2222"));
    assert!(b.contains("identity_file: ~/.ssh/flodl-join"));
    assert!(b.contains("libtorch: auto"));
    assert!(b.contains("persist: true"));

    let a = render_worker_yml("b300", &ep, "tok", Door::A, &cli);
    assert!(
        a.contains("data_source: sshfs://flodl-join@ctrl:/flodl/data"),
        "got:\n{a}"
    );
    assert!(!a.contains("from: rsync"), "door `a` cannot pull a source");

    let n = render_worker_yml("b300", &ep, "tok", Door::Nologin, &cli);
    assert!(!n.contains("data_source:"));
    assert!(!n.contains("from: rsync"));

    let mut cli = no_flags();
    cli.gpu_ram_share = Some(0.5);
    let apu = render_worker_yml("b300", &ep, "tok", Door::B, &cli);
    assert!(apu.contains("gpu_ram_share: 0.5"), "got:\n{apu}");
}

#[test]
fn the_authorized_line_composes_restrictions_and_the_doors_command() {
    let pub_line = "ssh-ed25519 AAAAtest flodl-join-b300";
    let served = PathBuf::from("/home/op/.flodl/run");
    let cli = no_flags();

    let b = authorized_keys_line(Door::B, &served, &cli, pub_line);
    assert!(b.starts_with("restrict,port-forwarding,permitopen=\"127.0.0.1:1337\","));
    assert!(
        b.contains("command=\"rrsync -ro /home/op/.flodl/run\""),
        "got: {b}"
    );
    assert!(b.ends_with(pub_line));

    let a = authorized_keys_line(Door::A, &served, &cli, pub_line);
    assert!(
        a.contains("command=\"internal-sftp -R -d /flodl/data\""),
        "got: {a}"
    );

    let n = authorized_keys_line(Door::Nologin, &served, &cli, pub_line);
    assert!(n.contains("command=\"/usr/sbin/nologin\""), "got: {n}");
}

/// The scaffold is not just plausible yml: it must load through the
/// REAL config loader as an overlay and surface the token where the
/// launcher reads it.
#[test]
fn the_scaffolded_overlay_loads_through_the_real_config_path() {
    let tmp = tempdir();
    let base = tmp.join("fdl.yml");
    fs::write(&base, "# base\n").unwrap();
    let token = fresh_token().unwrap();
    fs::write(
        tmp.join("fdl.b300.yml"),
        render_overlay_scaffold("b300", &token, &tmp, Some("trainer")),
    )
    .unwrap();
    let project = crate::config::load_project_with_env(&base, Some("b300")).unwrap();
    let cluster = project
        .cluster
        .expect("the overlay carries a cluster block");
    let join = cluster.controller.join.expect("a join block");
    assert_eq!(join.token.as_deref(), Some(token.as_str()));
    assert_eq!(join.discovery, Some(true));
    assert_eq!(join.start.as_deref(), Some("manual"));
    // Without a launcher-mode command the farm cannot host a run: the
    // command resolves against the base and executes locally.
    let cmd = project
        .commands
        .get("trainer")
        .expect("the scaffold wires the named command");
    assert_eq!(cmd.cluster, Some(true));
    assert!(cluster.workers.is_empty(), "walk-ins fill the roster");
    let _ = fs::remove_dir_all(&tmp);
}

// ── authorized_keys upsert ──────────────────────────────────────────────

const OUR_LINE: &str = "restrict,port-forwarding,permitopen=\"127.0.0.1:1337\",\
    command=\"rrsync -ro /srv/run\" ssh-ed25519 AAAAour flodl-join-b300";

#[test]
fn key_material_skips_quote_aware_options() {
    // The options field carries quoted spaces AND commas; the scan must
    // still land on the key type.
    assert_eq!(key_material(OUR_LINE), Some(("ssh-ed25519", "AAAAour")),);
    // A bare line (no options) and other key types.
    assert_eq!(
        key_material("ssh-rsa AAAAbare user@host"),
        Some(("ssh-rsa", "AAAAbare")),
    );
    assert_eq!(
        key_material("sk-ecdsa-sha2-nistp256@openssh.com AAAAsk c"),
        Some(("sk-ecdsa-sha2-nistp256@openssh.com", "AAAAsk")),
    );
    // Comments, blanks, and garbage are not keys.
    assert_eq!(key_material("# a comment"), None);
    assert_eq!(key_material(""), None);
    assert_eq!(key_material("options-only-no-key"), None);
}

#[test]
fn upsert_appends_replaces_or_leaves_identical() {
    // Empty file: appended, trailing newline included.
    let (out, o) = upsert_authorized_line("", OUR_LINE).unwrap();
    assert_eq!(o, UpsertOutcome::Appended);
    assert_eq!(out, format!("{OUR_LINE}\n"));

    // Foreign lines are preserved byte for byte, ours appended after.
    let foreign = "ssh-ed25519 AAAAforeign someone@laptop\n# a comment\n";
    let (out, o) = upsert_authorized_line(foreign, OUR_LINE).unwrap();
    assert_eq!(o, UpsertOutcome::Appended);
    assert!(
        out.starts_with(foreign),
        "foreign content must be untouched"
    );
    assert!(out.ends_with(&format!("{OUR_LINE}\n")));

    // The same key under DIFFERENT options: replaced in place, the
    // neighbours untouched.
    let mixed = "ssh-ed25519 AAAAforeign a@b\nssh-ed25519 AAAAour old-comment\n# tail\n";
    let (out, o) = upsert_authorized_line(mixed, OUR_LINE).unwrap();
    assert_eq!(o, UpsertOutcome::Replaced);
    assert!(out.contains("ssh-ed25519 AAAAforeign a@b\n"));
    assert!(out.contains(&format!("{OUR_LINE}\n")));
    assert!(!out.contains("old-comment"));
    assert!(out.ends_with("# tail\n"));

    // Byte-identical line already present: a no-op that says so.
    let installed = format!("{OUR_LINE}\n");
    let (out, o) = upsert_authorized_line(&installed, OUR_LINE).unwrap();
    assert_eq!(o, UpsertOutcome::Identical);
    assert_eq!(out, installed);
}

/// The non-interactive contract: without explicit consent the install
/// is a skip that says why, never a silent mutation; the two explicit
/// flags contradict loudly.
#[test]
fn install_needs_explicit_consent() {
    let mut changes = Changes::new(false);
    let mut cli = no_flags();
    cli.yes = true; // --yes is NOT consent for a security mutation
    let line = OUR_LINE;
    match install_authorized_line(&cli, &mut changes, line, 1).unwrap() {
        InstallAction::Skipped(why) => {
            assert!(why.contains("--install-key"), "got: {why}")
        }
        other => panic!("--yes alone must not install, got {other:?}"),
    }
    let mut cli = no_flags();
    cli.no_install_key = true;
    assert!(matches!(
        install_authorized_line(&cli, &mut changes, line, 1).unwrap(),
        InstallAction::Skipped(_),
    ));
    let mut cli = no_flags();
    cli.install_key = true;
    cli.no_install_key = true;
    assert!(install_authorized_line(&cli, &mut changes, line, 1).is_err());
    // Nothing above got as far as touching the file.
    assert!(changes.entries.is_empty());
}

#[test]
fn fresh_tokens_are_32_hex_and_unique() {
    let a = fresh_token().unwrap();
    let b = fresh_token().unwrap();
    assert_eq!(a.len(), 32);
    assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    assert_ne!(a, b);
}

// ── Farm shape recovery ─────────────────────────────────────────────────

/// Every door's worker yml must read back as the door that wrote it.
/// The wizard reuses credentials across runs, so before this the second
/// run silently re-rendered the farm from flag defaults: the door fell
/// back to `b` and the controller to this box's hostname, leaving the
/// printed authorized_keys line describing a farm that no longer matched
/// the one on disk.
#[test]
fn a_farms_door_and_controller_survive_a_flagless_rerun() {
    for door in [Door::B, Door::A, Door::Nologin] {
        let farm = tempdir();
        fs::create_dir_all(&farm).unwrap();
        let ctrl = Endpoint::parse(Some("op@ctrl.example:2222")).unwrap();
        let yml = render_worker_yml("f", &ctrl, "tok", door, &no_flags());
        fs::write(farm.join("worker.yml"), yml).unwrap();

        let (got_door, got_ctrl) = recover_shape(&farm).expect("the farm reads back");
        assert_eq!(got_door, door, "door for {door:?}");
        let round = Endpoint::parse(Some(&got_ctrl)).unwrap();
        assert_eq!(round.host, "ctrl.example");
        assert_eq!(round.port, 2222);
        assert_eq!(round.user, "op");
        let _ = fs::remove_dir_all(&farm);
    }
}

#[test]
fn a_farm_dir_without_a_worker_yml_recovers_nothing() {
    // Nothing to recover must stay None rather than a guess: a first run
    // has to keep taking the flag defaults.
    let farm = tempdir();
    fs::create_dir_all(&farm).unwrap();
    assert!(recover_shape(&farm).is_none());
    let _ = fs::remove_dir_all(&farm);
}

/// `--authorized-keys` exists for doors the default cannot reach, but
/// the promise it must not break is that the wizard installs door keys
/// and never edits system sshd configuration.
#[test]
fn an_authorized_keys_path_under_etc_ssh_is_refused() {
    let mut cli = no_flags();
    cli.install_key = true;
    cli.authorized_keys = Some("/etc/ssh/authorized_keys.d/op".to_string());
    let mut changes = Changes::new(false);
    let err = install_authorized_line(&cli, &mut changes, "ssh-ed25519 AAAA test", 22).unwrap_err();
    assert!(err.contains("system sshd configuration"), "got: {err}");
    assert!(err.contains("by hand"), "names the way out: {err}");
    // The dry half keeps the same promise.
    cli.dry_run = true;
    let err = install_authorized_line(&cli, &mut changes, "ssh-ed25519 AAAA test", 22).unwrap_err();
    assert!(err.contains("system sshd configuration"), "got: {err}");
}

// ── Generated sshd drop-in ──────────────────────────────────────────────

/// `ForceCommand` belongs only to the tunnel-only door. Doors a and b
/// carry their command in the key line, and a daemon-level forced
/// command would override it — the tunnel would keep working while the
/// mount or the source pull failed, which is the confusing half-failure.
#[test]
fn the_drop_in_forces_a_command_only_for_the_tunnel_only_door() {
    let deb = crate::util::platform::Platform::Debian;
    let nologin = render_sshd_conf("f", Door::Nologin, 2022, deb);
    assert!(
        nologin.contains("ForceCommand /usr/sbin/nologin"),
        "{nologin}"
    );
    for door in [Door::A, Door::B] {
        let conf = render_sshd_conf("f", door, 2022, deb);
        assert!(
            !conf.contains("ForceCommand"),
            "{door:?} must not force one:\n{conf}"
        );
    }
}

/// The guardrail is bound to the exposed port, not to a user: that is
/// what leaves ordinary logins on 22 alone while confining every key
/// that arrives on the door, including ones added later.
#[test]
fn the_drop_in_scopes_the_guardrail_to_the_port() {
    let conf = render_sshd_conf(
        "f",
        Door::Nologin,
        2022,
        crate::util::platform::Platform::Debian,
    );
    assert!(conf.contains("Match LocalPort 2022"), "{conf}");
    assert!(conf.contains("PermitOpen 127.0.0.1:1337"), "{conf}");
    // 22 must still be served, or the operator locks themselves out of
    // the box the moment the drop-in lands.
    assert!(conf.contains("\nPort 22\n"), "{conf}");
    assert!(
        !conf.contains("Match User"),
        "user-scoped defeats the purpose: {conf}"
    );
}

/// Each family's trap is named in the file it would bite.
#[test]
fn the_drop_in_names_the_per_platform_trap() {
    use crate::util::platform::Platform;
    let deb = render_sshd_conf("f", Door::Nologin, 2022, Platform::Debian);
    assert!(
        deb.contains("ssh.socket"),
        "Debian must warn about socket activation: {deb}"
    );
    let rhel = render_sshd_conf("f", Door::Nologin, 2022, Platform::Rhel);
    assert!(
        rhel.contains("SELinux"),
        "RHEL must warn about the port label: {rhel}"
    );
    // On the default port neither trap applies, so neither is mentioned.
    let plain = render_sshd_conf("f", Door::Nologin, 22, Platform::Debian);
    assert!(!plain.contains("ssh.socket"), "{plain}");
}

/// A scaffolded overlay must never cost the project its commands.
///
/// The placeholder form is the trap: a `commands:` key whose only
/// children are comments is YAML *null*, and the overlay merge applies
/// that null over the base's command map — deleting every command in
/// the project whenever the farm overlay is active. Caught after
/// shipping it; the key has to be commented out along with them.
#[test]
fn a_scaffolded_overlay_never_deletes_the_projects_commands() {
    for hint in [None, Some("trainer")] {
        let tmp = tempdir();
        let base = tmp.join("fdl.yml");
        fs::write(&base, "commands:\n  build:\n    run: echo hi\n").unwrap();
        let token = fresh_token().unwrap();
        fs::write(
            tmp.join("fdl.f.yml"),
            render_overlay_scaffold("f", &token, &tmp, hint),
        )
        .unwrap();
        let merged = crate::config::load_project_with_env(&base, Some("f")).unwrap();
        assert!(
            merged.commands.contains_key("build"),
            "hint={hint:?}: the base command vanished under the overlay",
        );
        match hint {
            Some(name) => assert_eq!(
                merged.commands.get(name).and_then(|c| c.cluster),
                Some(true),
                "a named command must be wired for launcher mode",
            ),
            // Nothing named means nothing added, not a wiped map.
            None => assert_eq!(merged.commands.len(), 1),
        }
        let _ = fs::remove_dir_all(&tmp);
    }
}

// ── The write-through change recorder ───────────────────────────────────

#[test]
fn the_recorder_classifies_and_a_dry_one_withholds_the_write() {
    let tmp = tempdir();
    let path = tmp.join("artifact.txt");

    let mut real = Changes::new(false);
    assert_eq!(
        real.write(&path, "v1\n", "artifact").unwrap(),
        ChangeKind::Create
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "v1\n");
    // Identical content is never rewritten.
    assert_eq!(
        real.write(&path, "v1\n", "artifact").unwrap(),
        ChangeKind::Unchanged
    );
    assert_eq!(
        real.write(&path, "v2\n", "artifact").unwrap(),
        ChangeKind::Update
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "v2\n");

    let missing = tmp.join("never-written.txt");
    let mut dry = Changes::new(true);
    assert_eq!(
        dry.write(&missing, "x\n", "artifact").unwrap(),
        ChangeKind::Create
    );
    assert!(!missing.exists(), "a dry run must not write");
    // ... but it still classifies against what IS on disk.
    assert_eq!(
        dry.write(&path, "v3\n", "artifact").unwrap(),
        ChangeKind::Update
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "v2\n");
    let _ = fs::remove_dir_all(&tmp);
}

// ── --dry-run ───────────────────────────────────────────────────────────

/// A dry first pass in an empty directory: the full report, placeholder
/// credentials, a change list of creates, and NOT ONE byte on disk.
#[test]
fn a_dry_run_in_an_empty_dir_writes_nothing_and_plans_everything() {
    let tmp = tempdir();
    let mut cli = no_flags();
    cli.label = Some("dryfarm".to_string());
    cli.dry_run = true;

    let report = wizard_at(&cli, &tmp).unwrap();

    assert!(report.dry_run);
    assert!(matches!(report.overlay_action, OverlayAction::Scaffolded));
    assert!(matches!(report.key_action, KeyAction::Generated));
    // Credentials an apply would mint appear as placeholders, never as
    // values the apply will not reproduce.
    assert!(report.authorized_line.contains(PLACEHOLDER_PUB));
    assert!(report.worker_yml.contains(PLACEHOLDER_TOKEN));
    // Consent is read from flags, never prompted for.
    assert!(matches!(report.install, InstallAction::Skipped(_)));
    // Everything an apply would create is in the plan.
    assert!(report.changes.iter().all(|c| c.kind == ChangeKind::Create));
    let planned: Vec<&str> = report.changes.iter().map(|c| c.what).collect();
    for what in [
        "minimal base fdl.yml",
        ".fdl self-gitignore",
        "join key pair",
        "farm overlay",
        "worker fdl.yml",
        "install notes",
        "sshd drop-in",
    ] {
        assert!(planned.contains(&what), "missing {what} in {planned:?}");
    }
    // And the directory is untouched: no base yml, no overlay, no .fdl.
    let left: Vec<_> = fs::read_dir(&tmp).unwrap().flatten().collect();
    assert!(left.is_empty(), "dry run left files behind: {left:?}");

    // The JSON twin carries the same facts.
    let json = report.to_json();
    assert_eq!(json["dry_run"], serde_json::json!(true));
    assert!(!json["changes"].as_array().unwrap().is_empty());
    let _ = fs::remove_dir_all(&tmp);
}

/// A dry pass over a farm already in shape reports reuse and changes
/// nothing — the idle re-run a GUI issues before offering an apply.
#[test]
fn a_dry_run_over_an_existing_farm_reports_reuse_and_keeps_content() {
    let tmp = tempdir();
    fs::write(tmp.join("fdl.yml"), "# base\n").unwrap();
    let label = "shaped";
    let farm = tmp.join(".fdl").join(label);
    fs::create_dir_all(farm.join("keys")).unwrap();
    fs::write(farm.join("keys/flodl-join"), "PRIVATE\n").unwrap();
    fs::write(
        farm.join("keys/flodl-join.pub"),
        "ssh-ed25519 AAAAexisting flodl-join-shaped\n",
    )
    .unwrap();
    let token = "aaaabbbbccccddddaaaabbbbccccdddd";
    fs::write(
        tmp.join(format!("fdl.{label}.yml")),
        format!("cluster:\n  controller:\n    join:\n      token: {token}\n"),
    )
    .unwrap();
    let ctrl = Endpoint::parse(Some("op@ctrl.example:2222")).unwrap();
    let mut cli = no_flags();
    cli.label = Some(label.to_string());
    let worker = render_worker_yml(label, &ctrl, token, Door::B, &cli);
    fs::write(farm.join("worker.yml"), &worker).unwrap();

    cli.dry_run = true;
    let report = wizard_at(&cli, &tmp).unwrap();

    assert!(matches!(report.key_action, KeyAction::Reused));
    assert!(matches!(report.overlay_action, OverlayAction::TokenReused));
    // The recovered shape rendered the same worker yml, so it is not a
    // change; notes + drop-in do not exist yet, so they are.
    let kind_of = |what: &str| {
        report
            .changes
            .iter()
            .find(|c| c.what == what)
            .map(|c| c.kind)
    };
    assert_eq!(kind_of("worker fdl.yml"), Some(ChangeKind::Unchanged));
    assert_eq!(kind_of("join key pair"), Some(ChangeKind::Unchanged));
    assert_eq!(kind_of("farm overlay"), Some(ChangeKind::Unchanged));
    assert_eq!(kind_of("install notes"), Some(ChangeKind::Create));
    assert_eq!(kind_of("sshd drop-in"), Some(ChangeKind::Create));
    // The real key and token flow into the report (nothing is minted,
    // so nothing is a placeholder).
    assert!(report.authorized_line.contains("AAAAexisting"));
    assert!(report.worker_yml.contains(token));
    // And disk is exactly as staged: no notes, no drop-in, same worker.
    assert!(!farm.join("install-notes.md").exists());
    assert!(!farm.join(format!("sshd-{label}.conf")).exists());
    assert_eq!(fs::read_to_string(farm.join("worker.yml")).unwrap(), worker);
    let _ = fs::remove_dir_all(&tmp);
}

/// `--dry-run --regen` plans the credential swap without touching it.
#[test]
fn a_dry_regen_promises_new_credentials_without_minting_them() {
    let tmp = tempdir();
    fs::write(tmp.join("fdl.yml"), "# base\n").unwrap();
    let label = "regenfarm";
    let farm = tmp.join(".fdl").join(label);
    fs::create_dir_all(farm.join("keys")).unwrap();
    fs::write(farm.join("keys/flodl-join"), "PRIVATE\n").unwrap();
    fs::write(farm.join("keys/flodl-join.pub"), "ssh-ed25519 AAAAold c\n").unwrap();
    let overlay = format!(
        "cluster:\n  controller:\n    join:\n      token: {}\n",
        "a".repeat(32)
    );
    fs::write(tmp.join(format!("fdl.{label}.yml")), &overlay).unwrap();

    let mut cli = no_flags();
    cli.label = Some(label.to_string());
    cli.dry_run = true;
    cli.regen = true;
    let report = wizard_at(&cli, &tmp).unwrap();

    assert!(matches!(report.key_action, KeyAction::Regenerated));
    assert!(matches!(
        report.overlay_action,
        OverlayAction::TokenReplaced
    ));
    assert!(report.authorized_line.contains(PLACEHOLDER_PUB));
    assert!(report.worker_yml.contains(PLACEHOLDER_TOKEN));
    // The old credentials are still exactly in place.
    assert_eq!(
        fs::read_to_string(farm.join("keys/flodl-join.pub")).unwrap(),
        "ssh-ed25519 AAAAold c\n",
    );
    assert_eq!(
        fs::read_to_string(tmp.join(format!("fdl.{label}.yml"))).unwrap(),
        overlay,
    );
    let _ = fs::remove_dir_all(&tmp);
}

/// The dry install verdict classifies the file without touching it.
#[test]
fn a_dry_install_verdict_reads_the_file_and_never_writes_it() {
    let tmp = tempdir();
    let ak = tmp.join("authorized_keys");
    fs::write(&ak, format!("{OUR_LINE}\n")).unwrap();
    let before = fs::read_to_string(&ak).unwrap();

    let mut cli = no_flags();
    cli.dry_run = true;
    cli.install_key = true;
    cli.authorized_keys = Some(ak.display().to_string());

    let mut changes = Changes::new(true);
    // Identical line → already present.
    assert!(matches!(
        install_authorized_line(&cli, &mut changes, OUR_LINE, 1).unwrap(),
        InstallAction::AlreadyPresent,
    ));
    // Same key, different options → would replace.
    let other_opts = OUR_LINE.replace("restrict,", "");
    assert!(matches!(
        install_authorized_line(&cli, &mut changes, &other_opts, 1).unwrap(),
        InstallAction::Replaced,
    ));
    // A placeholder key (an apply would mint it) → would append.
    let placeholder_line = format!("restrict {PLACEHOLDER_PUB}");
    assert!(matches!(
        install_authorized_line(&cli, &mut changes, &placeholder_line, 1).unwrap(),
        InstallAction::Installed,
    ));
    // Without consent flags the dry verdict is the same skip as ever.
    cli.install_key = false;
    assert!(matches!(
        install_authorized_line(&cli, &mut changes, OUR_LINE, 1).unwrap(),
        InstallAction::Skipped(_),
    ));
    assert_eq!(fs::read_to_string(&ak).unwrap(), before, "dry run wrote");
    let _ = fs::remove_dir_all(&tmp);
}

// ── --list ──────────────────────────────────────────────────────────────

/// The union rule: overlays that are farms, farm dirs without overlays,
/// and env overlays that are not farms — each classified, none dressed
/// as another. `.fdl/` state that is not a farm (schema caches) never
/// lists.
#[test]
fn farm_enumeration_unions_overlays_and_dirs_and_skips_non_farms() {
    let tmp = tempdir();
    fs::write(tmp.join("fdl.yml"), "# base\n").unwrap();

    // A full farm: overlay with token + wizard-shaped dir.
    let full = tmp.join(".fdl").join("full");
    fs::create_dir_all(full.join("keys")).unwrap();
    fs::write(full.join("keys/flodl-join"), "PRIVATE\n").unwrap();
    fs::write(full.join("keys/flodl-join.pub"), "ssh-ed25519 AAAA c\n").unwrap();
    let ctrl = Endpoint::parse(Some("op@ctrl.example:2222")).unwrap();
    fs::write(
        full.join("worker.yml"),
        render_worker_yml("full", &ctrl, "tok", Door::A, &no_flags()),
    )
    .unwrap();
    fs::write(
        tmp.join("fdl.full.yml"),
        format!(
            "cluster:\n  controller:\n    join:\n      token: {}\n",
            "b".repeat(32)
        ),
    )
    .unwrap();

    // A half farm: keys only, overlay deleted.
    let half = tmp.join(".fdl").join("half");
    fs::create_dir_all(half.join("keys")).unwrap();

    // An env overlay that is no farm at all.
    fs::write(tmp.join("fdl.cluster.yml"), "cluster: {}\n").unwrap();

    // Non-farm .fdl state must not masquerade as a farm.
    fs::create_dir_all(tmp.join(".fdl").join("schema-cache")).unwrap();

    let (farms, other_envs) = enumerate_farms(&tmp);
    let labels: Vec<&str> = farms.iter().map(|f| f.label.as_str()).collect();
    assert_eq!(labels, vec!["full", "half"]);
    assert_eq!(other_envs, vec!["cluster".to_string()]);

    let full_info = &farms[0];
    assert!(full_info.overlay_exists && full_info.has_token && full_info.key_present);
    assert!(full_info.worker_yml && !full_info.cloud_init);
    assert_eq!(full_info.door, Some(Door::A));
    assert_eq!(
        full_info.controller.as_deref(),
        Some("op@ctrl.example:2222")
    );

    let half_info = &farms[1];
    assert!(!half_info.overlay_exists && !half_info.has_token && !half_info.key_present);
    assert_eq!(half_info.door, None);

    // The JSON twin says the same.
    let json = full_info.to_json();
    assert_eq!(json["door"], serde_json::json!("a"));
    assert_eq!(json["overlay"]["token"], serde_json::json!(true));

    // Human render: farms present, MISSING called out, non-farms aside.
    let text = render_farm_list(&tmp, &farms, &other_envs);
    assert!(text.contains("full"), "{text}");
    assert!(text.contains("MISSING"), "{text}");
    assert!(text.contains("not farms: cluster"), "{text}");
    let _ = fs::remove_dir_all(&tmp);
}
