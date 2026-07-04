    use super::*;
    use super::spawn::{
        build_remote_bash_command, build_remote_relay_bash_command, build_slim_envelope_for,
        shell_quote, supervise_children, PerHostPrebuild,
    };
    use serde_json::json;
    use std::process::Command;

    fn sample_relay_spec() -> RelaySpec {
        RelaySpec {
            host: "pascal".into(),
            controller_host: "192.168.122.1".into(),
            controller_port: 1337,
            ranks: vec![1, 2],
            salt_hex: "0123456789abcdef0123456789abcdef".into(),
            world_size: 3,
            data_channel: true,
        }
    }

    #[test]
    fn relay_spec_hex_json_round_trips() {
        let spec = sample_relay_spec();
        let hex = crate::distributed::cluster::hex_encode(
            serde_json::to_string(&spec).unwrap().as_bytes(),
        );
        let bytes = crate::distributed::cluster::hex_decode(&hex).unwrap();
        let back: RelaySpec = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, spec);
    }

    #[test]
    fn remote_relay_bash_command_exports_relay_env_only() {
        let spec_hex = "deadbeef";
        let cmd = build_remote_relay_bash_command(
            "/opt/flodl",
            spec_hex,
            "ddp-bench",
            &["--model".into(), "resnet-graph".into()],
            &std::collections::BTreeMap::new(),
            &std::collections::BTreeMap::new(),
            None,
        );
        // Exports the relay spec and runs the binary...
        assert!(cmd.contains("FLODL_RELAY_JSON="), "missing relay env: {cmd}");
        assert!(cmd.contains("cd '/opt/flodl'"), "missing cd: {cmd}");
        assert!(cmd.contains("fdl 'ddp-bench'"), "missing fdl cmd: {cmd}");
        assert!(cmd.contains("--model") && cmd.contains("resnet-graph"));
        // ...but never the rank-role env vars or CUDA scoping.
        assert!(!cmd.contains("FLODL_CLUSTER_JSON="), "leaked rank envelope: {cmd}");
        assert!(!cmd.contains("FLODL_LOCAL_RANK="), "leaked rank slot: {cmd}");
        assert!(!cmd.contains("CUDA_VISIBLE_DEVICES="), "relay must not scope CUDA: {cmd}");
        // Trap wrapper for clean signal forwarding.
        assert!(cmd.contains("trap ") && cmd.contains("__flodl_pid"), "missing trap: {cmd}");
    }

    fn canonical_full_json() -> serde_json::Value {
        json!({
            "controller": {
                "host": "192.168.122.1",
                "port": 29500,
                "path": "/opt/flodl"
            },
            "workers": [
                {
                    "host": "host-a",
                    "ranks": [0],
                    "local_devices": [0],
                    "nccl_socket_ifname": "virbr0",
                    "path": "/opt/flodl",
                    "arch": "precompiled/cu128"
                },
                {
                    "host": "host-b",
                    "ssh": { "target": "host-b" },
                    "ranks": [1, 2],
                    "local_devices": "all",
                    "nccl_socket_ifname": "enp1s0",
                    "path": "/srv/flodl"
                }
            ]
        })
    }

    #[test]
    fn parses_full_topology() {
        let c = FullCluster::from_value(&canonical_full_json()).unwrap();
        assert_eq!(c.controller.host, "192.168.122.1");
        assert_eq!(c.controller.port, 29500);
        assert_eq!(c.world_size(), 3);
        assert!(c.spans_multiple_workers());

        assert_eq!(c.workers.len(), 2);
        assert_eq!(c.workers[0].host, "host-a");
        assert_eq!(c.workers[0].ranks, vec![0]);
        assert_eq!(c.workers[0].local_devices, Some(vec![0]));
        assert_eq!(c.workers[0].ssh, None);

        assert_eq!(c.workers[1].host, "host-b");
        assert_eq!(c.workers[1].ranks, vec![1, 2]);
        // "all" stays unresolved at launcher-parse time; each host resolves
        // its own at startup.
        assert_eq!(c.workers[1].local_devices, None);
        assert_eq!(c.workers[1].ssh_target(), "host-b");
    }

    #[test]
    fn rejects_empty_workers() {
        let mut v = canonical_full_json();
        v["workers"] = json!([]);
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("workers must be non-empty"), "got: {err}");
    }

    #[test]
    fn rejects_rank_gap_across_hosts() {
        let mut v = canonical_full_json();
        v["workers"][1]["ranks"] = json!([2, 3]); // gap: 0 + (2,3) misses rank 1
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(
            err.to_string().contains("duplicates or gaps"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_duplicate_ranks() {
        let mut v = canonical_full_json();
        v["workers"][1]["ranks"] = json!([0, 1]); // collides with host-a's [0]
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(
            err.to_string().contains("duplicates or gaps"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_local_devices_length_mismatch_for_explicit() {
        let mut v = canonical_full_json();
        v["workers"][1]["local_devices"] = json!([0]); // ranks: [1, 2] needs 2 devices
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("length mismatch"), "got: {err}");
    }

    #[test]
    fn accepts_local_devices_all_at_launcher_parse_time() {
        // "all" stays symbolic; resolution is deferred to startup on the
        // host that ends up parsing the slim envelope.
        let mut v = canonical_full_json();
        v["workers"][0]["local_devices"] = json!("all");
        let c = FullCluster::from_value(&v).unwrap();
        assert_eq!(c.workers[0].local_devices, None);
    }

    #[test]
    fn rejects_unknown_local_devices_string() {
        let mut v = canonical_full_json();
        v["workers"][0]["local_devices"] = json!("every");
        let err = FullCluster::from_value(&v).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("local_devices") && msg.contains("every"),
            "got: {msg}"
        );
    }

    #[test]
    fn rejects_controller_port_overflow() {
        let mut v = canonical_full_json();
        v["controller"]["port"] = json!(100_000);
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("u16"), "got: {err}");
    }

    #[test]
    fn slim_envelope_strips_ssh_carries_metadata() {
        // Direct test of the build_slim_envelope_for helper: the slim
        // shape must round-trip through LocalCluster::from_env on the
        // rank side, so it has to match that parser's expectations
        // (controller/world_size/num_workers/worker with no ssh field).
        let full = FullCluster::from_value(&canonical_full_json()).unwrap();
        let worker = full.workers.iter().find(|h| h.host == "host-b").unwrap();
        let env = build_slim_envelope_for(&full, worker);

        assert_eq!(env["controller"]["host"], "192.168.122.1");
        assert_eq!(env["controller"]["port"], 29500);
        assert_eq!(env["world_size"], 3);
        assert_eq!(env["num_workers"], 2);
        assert_eq!(env["worker"]["host"], "host-b");
        assert_eq!(env["worker"]["ranks"], serde_json::json!([1, 2]));
        assert_eq!(env["worker"]["local_devices"], serde_json::json!("all"));
        assert_eq!(env["worker"]["nccl_socket_ifname"], "enp1s0");
        // ssh: stripped (launcher-only field; slim envelope is rank-side).
        assert!(env["worker"].get("ssh").is_none(), "ssh must be stripped");
    }

    #[test]
    fn slim_envelope_emits_explicit_local_devices_when_present() {
        let full = FullCluster::from_value(&canonical_full_json()).unwrap();
        let host_a = full.workers.iter().find(|h| h.host == "host-a").unwrap();
        let env = build_slim_envelope_for(&full, host_a);
        assert_eq!(env["worker"]["local_devices"], serde_json::json!([0]));
    }

    #[test]
    fn shell_quote_simple() {
        assert_eq!(shell_quote("foo"), "'foo'");
    }

    #[test]
    fn shell_quote_with_spaces() {
        assert_eq!(shell_quote("foo bar"), "'foo bar'");
    }

    #[test]
    fn shell_quote_escapes_internal_quotes() {
        assert_eq!(shell_quote("don't"), "'don'\\''t'");
    }

    fn empty_env() -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::new()
    }

    #[test]
    fn build_remote_bash_command_shape() {
        let cluster_env = empty_env();
        let host_env = empty_env();
        let s = build_remote_bash_command(
            "/srv/flodl",
            "abcd1234",
            "host-b",
            0,
            Some("cluster"),
            "train",
            &["--epochs".to_string(), "10".to_string()],
            &cluster_env,
            &host_env,
            None,
            None,
        );
        assert!(s.starts_with("cd '/srv/flodl' && "));
        assert!(s.contains("FLODL_CLUSTER_JSON='abcd1234'"));
        assert!(s.contains("FLODL_HOST_NAME='host-b'"));
        assert!(s.contains("FLODL_LOCAL_RANK=0"));
        assert!(s.contains("FDL_ENV='cluster'"));
        assert!(s.contains("fdl 'train' '--epochs' '10' &\n"));
        // The trap forwards TERM, then escalates to KILL after a grace
        // period (a rank stuck in an uninterruptible CUDA ioctl ignores
        // TERM forever; nothing else on the remote escalates once the
        // launcher is gone).
        assert!(s.contains("trap 'kill -TERM \"$__flodl_pid\" 2>/dev/null;"));
        assert!(s.contains("kill -KILL \"$__flodl_pid\""));
        assert!(s.contains("wait \"$__flodl_pid\""));
        assert!(s.ends_with("exit $?\n"));
    }

    #[test]
    fn build_remote_bash_command_omits_fdl_env_when_none() {
        let cluster_env = empty_env();
        let host_env = empty_env();
        let s = build_remote_bash_command(
            "/srv/flodl",
            "abcd",
            "worker",
            0,
            None,
            "train",
            &[],
            &cluster_env,
            &host_env,
            None,
            None,
        );
        assert!(
            !s.contains("FDL_ENV"),
            "FDL_ENV must be absent when overlay_env is None; got: {s}"
        );
    }

    #[test]
    fn build_remote_bash_command_uses_trap_wrapper() {
        // The trap wrapper is load-bearing: it keeps a bash process
        // alive on the remote after launch so that a connection-drop
        // SIGHUP from sshd reaches a shell that can signal the binary,
        // instead of being lost to a bare `exec`'d binary that ignores
        // SIGHUP. Without this, every cluster smoke leaves an orphan
        // ddp-bench on the remote until manual pkill.
        let cluster_env = empty_env();
        let host_env = empty_env();
        let s = build_remote_bash_command(
            "/srv", "ff", "w", 0, None, "train", &[],
            &cluster_env, &host_env, None, None,
        );
        assert!(s.contains(" fdl "), "missing `fdl` invocation: {s}");
        assert!(s.contains(" &\n"), "missing background `&`: {s}");
        assert!(
            s.contains("__flodl_pid=$!"),
            "missing `__flodl_pid=$!`: {s}"
        );
        assert!(
            s.contains("trap 'kill -TERM \"$__flodl_pid\""),
            "missing trap line: {s}"
        );
        assert!(s.contains("wait \"$__flodl_pid\""), "missing wait: {s}");
        assert!(s.ends_with("exit $?\n"), "missing exit prop: {s}");
    }

    #[test]
    fn build_remote_bash_command_quotes_dangerous_path() {
        // Single quotes in the path must round-trip through the
        // single-quote-escape idiom.
        let cluster_env = empty_env();
        let host_env = empty_env();
        let s = build_remote_bash_command(
            "/srv/it's", "ff", "w", 0, None, "train", &[],
            &cluster_env, &host_env, None, None,
        );
        assert!(
            s.contains("cd '/srv/it'\\''s'"),
            "path with single quote not properly escaped: {s}"
        );
    }

    #[test]
    fn build_remote_bash_command_uses_prebuild_binary_and_ld_path() {
        // When the prebuild envelope provides an entry for this host,
        // the remote dispatch must (a) emit LD_LIBRARY_PATH, (b) launch
        // the binary directly via `<bin>` (no `fdl` re-entry), and (c)
        // close with the trap wrapper so the binary can be cleaned up
        // via SIGHUP on connection drop.
        let cluster_env = empty_env();
        let host_env = empty_env();
        let pb = PerHostPrebuild {
            bin: "target/cluster/worker/release/ddp-bench".into(),
            ld_library_path: "/opt/libtorch/lib".into(),
            cwd_subpath: "ddp-bench".into(),
        };
        let s = build_remote_bash_command(
            "/srv/flodl",
            "abcd",
            "worker",
            0,
            None,
            "ddp-bench",
            &["--mode".into(), "nccl-sync".into()],
            &cluster_env,
            &host_env,
            None,
            Some(&pb),
        );
        assert!(
            s.contains("LD_LIBRARY_PATH='/opt/libtorch/lib'"),
            "missing prebuild LD_LIBRARY_PATH: {s}",
        );
        assert!(
            s.contains("cd '/srv/flodl/ddp-bench'"),
            "remote cwd must cd into <host.path>/<cwd_subpath>: {s}",
        );
        assert!(
            s.contains(" '/srv/flodl/target/cluster/worker/release/ddp-bench'"),
            "binary path must be absolute (independent of cwd offset): {s}",
        );
        assert!(
            !s.contains("fdl 'ddp-bench'"),
            "prebuild path must NOT re-enter fdl on remote: {s}",
        );
        assert!(
            s.contains("'--mode' 'nccl-sync' &\n"),
            "user args must be appended ahead of the trap wrapper: {s}",
        );
        assert!(s.ends_with("exit $?\n"), "trap wrapper must end the cmd: {s}");
    }

    #[test]
    fn build_remote_bash_command_prebuild_yields_to_host_env_ld_path() {
        // If the user sets LD_LIBRARY_PATH via host.env, the
        // auto-derived prebuild LD_LIBRARY_PATH must yield (the user's
        // value is the source of truth; e.g. they need extra paths
        // for bare-metal libnccl alongside libtorch).
        let cluster_env = empty_env();
        let mut host_env = empty_env();
        host_env.insert(
            "LD_LIBRARY_PATH".into(),
            "/opt/libtorch/lib:/usr/local/lib".into(),
        );
        let pb = PerHostPrebuild {
            bin: "target/cluster/worker/release/ddp-bench".into(),
            ld_library_path: "/opt/libtorch/lib".into(),
            cwd_subpath: String::new(),
        };
        let s = build_remote_bash_command(
            "/srv", "ff", "w", 0, None, "ddp-bench", &[],
            &cluster_env, &host_env, None, Some(&pb),
        );
        // Only the host_env value should be present; the auto-derived
        // prebuild-only LD_LIBRARY_PATH must be suppressed.
        let host_pos = s.find("LD_LIBRARY_PATH='/opt/libtorch/lib:/usr/local/lib'").unwrap();
        // The auto-derived entry would have emitted exactly this
        // substring; assert it's absent.
        assert!(
            !s.contains(" LD_LIBRARY_PATH='/opt/libtorch/lib' "),
            "auto-derived LD_LIBRARY_PATH should yield to host_env: {s}",
        );
        let _ = host_pos;
    }

    #[test]
    fn build_remote_bash_command_exports_cluster_and_host_env() {
        // Cluster-scope and host-scope env vars round-trip into the
        // exported shell command. Host overrides cluster on key
        // collisions.
        let mut cluster_env = empty_env();
        cluster_env.insert("NCCL_P2P_DISABLE".into(), "1".into());
        cluster_env.insert("SHARED_FLAG".into(), "cluster-wins".into());
        let mut host_env = empty_env();
        host_env.insert("HOST_FLAG".into(), "host-val".into());
        host_env.insert("SHARED_FLAG".into(), "host-wins".into());
        let s = build_remote_bash_command(
            "/srv", "ff", "w", 0, None, "train", &[],
            &cluster_env, &host_env, Some(1), None,
        );
        assert!(s.contains("NCCL_P2P_DISABLE='1'"));
        assert!(s.contains("HOST_FLAG='host-val'"));
        // Host SHARED_FLAG export comes after cluster's; the shell
        // takes the last value when env vars are assigned multiple
        // times in a `K=V K=V ...` prefix.
        let cluster_pos = s.find("SHARED_FLAG='cluster-wins'").unwrap();
        let host_pos = s.find("SHARED_FLAG='host-wins'").unwrap();
        assert!(cluster_pos < host_pos, "host env must export after cluster env");
        assert!(s.contains("CUDA_VISIBLE_DEVICES=1"));
    }

    #[test]
    fn slim_envelope_round_trips_through_local_cluster_parser() {
        // Smoke test: the slim envelope built by the launcher must parse
        // cleanly via the rank-side LocalCluster::from_value. Same wire
        // contract, validated end-to-end.
        let full = FullCluster::from_value(&canonical_full_json()).unwrap();
        let host_a = full.workers.iter().find(|h| h.host == "host-a").unwrap();
        let env = build_slim_envelope_for(&full, host_a);
        let parsed = crate::distributed::cluster::LocalCluster::from_value(&env)
            .expect("slim envelope must parse via LocalCluster::from_value");
        assert_eq!(parsed.world_size(), 3);
        assert_eq!(parsed.controller.host, "192.168.122.1");
        assert_eq!(parsed.worker.host, "host-a");
        assert_eq!(parsed.worker.ranks, vec![0]);
        assert_eq!(parsed.worker.local_devices, vec![0]);
        // FullCluster::from_value defaults salt to zeros; the envelope
        // carries that, and the rank-side parser reads it back.
        assert_eq!(parsed.salt, [0u8; crate::distributed::wire::SESSION_SALT_BYTES]);
    }

    #[test]
    fn slim_envelope_propagates_session_salt() {
        // The launcher generates a fresh salt and stamps it onto
        // FullCluster; every slim envelope it builds must carry that
        // salt unchanged through to the rank-side LocalCluster.
        let mut full = FullCluster::from_value(&canonical_full_json()).unwrap();
        let salt: crate::distributed::wire::SessionSalt = [
            0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04,
            0xfe, 0xed, 0xfa, 0xce, 0x05, 0x06, 0x07, 0x08,
        ];
        full = full.with_session_salt(salt);
        let host_a = full.workers.iter().find(|h| h.host == "host-a").unwrap();
        let env = build_slim_envelope_for(&full, host_a);
        // Salt field must be present as a 32-char lowercase hex string.
        let hex = env
            .get("salt")
            .and_then(|v| v.as_str())
            .expect("envelope.salt is a string");
        assert_eq!(hex.len(), 32);
        let parsed = crate::distributed::cluster::LocalCluster::from_value(&env).unwrap();
        assert_eq!(parsed.salt, salt);
    }

    #[test]
    fn supervise_children_clean_exit_returns_none() {
        // Both children exit cleanly. supervise_children should return
        // None without sending any kill signals.
        let mut children: Vec<super::spawn::SupervisedChild> = Vec::new();
        for lr in 0..2 {
            let child = Command::new("true")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn `true`");
            children.push(("host".to_string(), lr, vec![lr], child, Vec::new()));
        }
        assert!(supervise_children(children, None).is_none());
    }

    #[test]
    fn supervise_children_failure_terminates_peers() {
        // One child exits immediately with status 1; the other would
        // sleep for 60s. Concurrent supervision must detect the failure
        // and SIGTERM the sleeper so the call returns promptly. The
        // assertion is the wall-clock budget: significantly less than
        // the sleeper's 60s argument.
        let fail_child = Command::new("sh")
            .args(["-c", "exit 1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn `sh -c 'exit 1'`");
        let sleep_child = Command::new("sleep")
            .arg("60")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn `sleep 60`");
        let children = vec![
            ("host-fail".to_string(), 0, vec![0], fail_child, Vec::new()),
            ("host-sleep".to_string(), 1, vec![1], sleep_child, Vec::new()),
        ];

        let start = std::time::Instant::now();
        let err = supervise_children(children, None).expect("expected failure attribution");
        let elapsed = start.elapsed();

        assert!(
            err.to_string().contains("host-fail"),
            "attribution should name the first failed rank: {err}"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(15),
            "SIGTERM-on-failure must reap the sleeper well before its 60s budget; took {elapsed:?}"
        );
    }

    // ---- role-env promotion gate -------------------------------------------

    /// Clear all four role vars. Callers hold `cluster::ENV_MUTEX`.
    fn clear_role_env() {
        unsafe {
            std::env::remove_var(ENV_FULL_CLUSTER_JSON);
            std::env::remove_var(ENV_RELAY_JSON);
            std::env::remove_var(crate::distributed::cluster::ENV_CLUSTER_JSON);
            std::env::remove_var(crate::distributed::cluster::ENV_LOCAL_RANK);
        }
    }

    #[test]
    fn role_env_pristine_matches_dispatch_truth_table() {
        let _guard = crate::distributed::cluster::ENV_MUTEX.lock().unwrap();
        clear_role_env();
        assert!(role_env_pristine());

        // Each solo role var flips the gate off — including the relay's,
        // which the old hand-rolled auto-promote check missed (a relay
        // child on a multi-GPU host re-promoted and died at dispatch).
        for k in [ENV_FULL_CLUSTER_JSON, ENV_RELAY_JSON] {
            unsafe { std::env::set_var(k, "deadbeef") };
            assert!(!role_env_pristine(), "{k} set must not be pristine");
            clear_role_env();
        }
        // Rank-child shape: slim envelope + slot.
        unsafe {
            std::env::set_var(crate::distributed::cluster::ENV_CLUSTER_JSON, "deadbeef");
            std::env::set_var(crate::distributed::cluster::ENV_LOCAL_RANK, "0");
        }
        assert!(!role_env_pristine());
        // Inconsistent env (full + slim + slot) is not pristine either:
        // promotion is skipped and the launch-path dispatch errors loudly.
        unsafe { std::env::set_var(ENV_FULL_CLUSTER_JSON, "deadbeef") };
        assert!(!role_env_pristine());
        clear_role_env();
    }

    #[test]
    fn programmatic_promotion_is_role_gated() {
        let _guard = crate::distributed::cluster::ENV_MUTEX.lock().unwrap();
        clear_role_env();
        let full = FullCluster::from_value(&canonical_full_json()).unwrap();

        // Rank child: slim + slot set, FULL stripped by the launcher. The
        // child re-enters user code whose config still carries `cluster:` —
        // it must NOT re-promote (it did: every rank child died at dispatch
        // with "inconsistent env", breaking programmatic clusters end to
        // end).
        unsafe {
            std::env::set_var(crate::distributed::cluster::ENV_CLUSTER_JSON, "deadbeef");
            std::env::set_var(crate::distributed::cluster::ENV_LOCAL_RANK, "0");
        }
        assert!(!promote_programmatic_cluster(&full));
        assert!(std::env::var_os(ENV_FULL_CLUSTER_JSON).is_none());
        clear_role_env();

        // Relay child: only FLODL_RELAY_JSON set.
        unsafe { std::env::set_var(ENV_RELAY_JSON, "deadbeef") };
        assert!(!promote_programmatic_cluster(&full));
        assert!(std::env::var_os(ENV_FULL_CLUSTER_JSON).is_none());
        clear_role_env();

        // Pristine process: promotes, and the envelope round-trips through
        // the same path `DdpHandle::launch`'s dispatch consumes.
        assert!(promote_programmatic_cluster(&full));
        assert!(matches!(dispatch(), Ok(Role::Launcher)));
        let back = FullCluster::from_env().unwrap();
        assert_eq!(back.controller.host, full.controller.host);
        assert_eq!(back.controller.port, full.controller.port);
        assert_eq!(back.world_size(), full.world_size());
        // Already-promoted (launcher) process: fdl-cli's envelope wins,
        // no overwrite.
        assert!(!promote_programmatic_cluster(&full));
        clear_role_env();
    }

    // ---- env-block validation ----------------------------------------------

    /// Reserved keys are rejected at parse: user env is applied
    /// last-write-wins over the launcher's built-ins on both spawn
    /// mediums, so a reserved key reaching the spawn paths would
    /// silently clobber rank↔device identity.
    #[test]
    fn env_block_rejects_reserved_and_malformed_keys() {
        let with_env = |k: &str, v: &str| {
            let mut val = canonical_full_json();
            val["env"] = serde_json::json!({ k: v });
            FullCluster::from_value(&val)
        };
        for reserved in ["CUDA_VISIBLE_DEVICES", "FLODL_LOCAL_RANK", "FLODL_ANYTHING"] {
            let err = with_env(reserved, "x").unwrap_err();
            assert!(
                err.to_string().contains("reserved"),
                "{reserved}: expected reserved-key rejection, got: {err}"
            );
        }
        for bad in ["HAS SPACE", "1LEADING_DIGIT", "DASH-ED", ""] {
            let err = with_env(bad, "x").unwrap_err();
            assert!(
                err.to_string().contains("valid env var name"),
                "{bad:?}: expected charset rejection, got: {err}"
            );
        }
        // The whole point of the env block stays available.
        for ok in ["NCCL_DEBUG", "LD_PRELOAD", "LD_LIBRARY_PATH", "_UNDER"] {
            assert!(
                with_env(ok, "x").is_ok(),
                "{ok}: legitimate tuning key must be accepted"
            );
        }
    }

    /// Per-worker env blocks ride the same chokepoint.
    #[test]
    fn worker_env_block_rejects_reserved_keys() {
        let mut val = canonical_full_json();
        val["workers"][0]["env"] =
            serde_json::json!({ "FLODL_CLUSTER_JSON": "deadbeef" });
        let err = FullCluster::from_value(&val).unwrap_err();
        assert!(err.to_string().contains("reserved"), "got: {err}");
    }

    // ---- elastic supervision ------------------------------------------------

    /// Post-formation, a child failure is a membership event: its global
    /// ranks land in the reported-deaths queue, peers are NOT terminated,
    /// and a within-tolerance run returns success (degraded, not failed).
    #[test]
    fn supervise_elastic_tolerates_death_and_reports_ranks() {
        use crate::distributed::cluster_coordinator::ReportedDeaths;
        let fail_child = Command::new("sh")
            .args(["-c", "exit 7"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn failing child");
        // The survivor exits cleanly after a short beat — under
        // kill-all it would be SIGTERMed instead (exit code != 0 →
        // this test would then trip the verdict below).
        let survivor = Command::new("sh")
            .args(["-c", "sleep 1; exit 0"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn survivor");
        let children = vec![
            ("host-a".to_string(), 0, vec![0], fail_child, Vec::new()),
            ("host-b".to_string(), 1, vec![1], survivor, Vec::new()),
        ];
        let queue: ReportedDeaths =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let dead = crate::distributed::controller::DeadRanks::new(2);
        let elastic = ElasticSupervision {
            reported_deaths: std::sync::Arc::clone(&queue),
            dead_ranks: std::sync::Arc::clone(&dead),
            max_failure: Some(
                crate::distributed::max_failure::MaxFailureThreshold::Absolute(2),
            ),
            world_size: 2,
            cohort_formed: std::sync::Arc::new(
                std::sync::atomic::AtomicBool::new(true),
            ),
        };
        let verdict = supervise_children(children, Some(elastic));
        assert!(
            verdict.is_none(),
            "within-tolerance death must not fail the run: {verdict:?}"
        );
        assert_eq!(
            queue.lock().unwrap().as_slice(),
            &[0],
            "failed child's global ranks must be reported to the coordinator"
        );
    }

    /// Pre-formation the legacy kill-all stands even with an elastic
    /// context attached — a half-formed NCCL cohort cannot absorb a
    /// death.
    #[test]
    fn supervise_elastic_pre_formation_kills_all() {
        let fail_child = Command::new("sh")
            .args(["-c", "exit 1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn failing child");
        let sleeper = Command::new("sleep")
            .arg("60")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleeper");
        let children = vec![
            ("host-a".to_string(), 0, vec![0], fail_child, Vec::new()),
            ("host-b".to_string(), 1, vec![1], sleeper, Vec::new()),
        ];
        let elastic = ElasticSupervision {
            reported_deaths: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            dead_ranks: crate::distributed::controller::DeadRanks::new(2),
            max_failure: None,
            world_size: 2,
            cohort_formed: std::sync::Arc::new(
                std::sync::atomic::AtomicBool::new(false),
            ),
        };
        let start = std::time::Instant::now();
        let err = supervise_children(children, Some(elastic))
            .expect("pre-formation failure must fail the run");
        assert!(err.to_string().contains("host-a"), "got: {err}");
        assert!(
            start.elapsed() < std::time::Duration::from_secs(15),
            "sleeper must be SIGTERMed promptly pre-formation"
        );
    }

    /// Deaths at or past max_failure fail the run even when every child
    /// eventually exited cleanly (the coordinator's ShutdownWithSave
    /// path produces clean exits).
    #[test]
    fn supervise_elastic_verdict_fails_past_threshold() {
        let c0 = Command::new("sh")
            .args(["-c", "exit 3"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        let children = vec![("host-a".to_string(), 0, vec![0], c0, Vec::new())];
        let dead = crate::distributed::controller::DeadRanks::new(3);
        // Simulate the coordinator having processed reports up to the
        // threshold (supervision reads the LEDGER for its verdict).
        dead.declare_dead(0);
        dead.declare_dead(1);
        let elastic = ElasticSupervision {
            reported_deaths: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            dead_ranks: std::sync::Arc::clone(&dead),
            max_failure: Some(
                crate::distributed::max_failure::MaxFailureThreshold::Absolute(2),
            ),
            world_size: 3,
            cohort_formed: std::sync::Arc::new(
                std::sync::atomic::AtomicBool::new(true),
            ),
        };
        let err = supervise_children(children, Some(elastic))
            .expect("threshold breach must fail the run");
        assert!(err.to_string().contains("max_failure exceeded"), "got: {err}");
    }
