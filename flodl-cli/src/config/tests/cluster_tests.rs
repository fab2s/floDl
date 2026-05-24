//! Cluster-config tests: overlay parsing, ClusterConfig::validate
//! invariants, LocalDevices Marker/Explicit serialization, canonical
//! JSON round-trip, and `resolve_cluster_dispatch` /
//! `cluster_dispatch_enabled` chain semantics.

use super::*;

    #[test]
    fn cluster_overlay_parses() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).expect("parse cluster overlay");
        let cluster = cfg.cluster.as_ref().expect("cluster: block present");
        assert_eq!(cluster.controller.host, "192.168.122.1");
        assert_eq!(cluster.controller.port, 29500);
        assert_eq!(cluster.world_size(), 3);
        assert_eq!(cluster.workers.len(), 2);
        assert!(cluster.spans_multiple_hosts());

        let worker = &cluster.workers[1];
        assert_eq!(worker.host, "worker-host");
        assert_eq!(worker.ranks, vec![1, 2]);
        assert_eq!(
            worker.local_devices,
            LocalDevices::Explicit(vec![0, 1])
        );
        assert_eq!(worker.nccl_socket_ifname, "enp1s0");
        assert_eq!(worker.path, "/srv/flodl");
        assert_eq!(
            worker.ssh.as_ref().and_then(|s| s.target.as_deref()),
            Some("worker-host"),
        );

        // CommandSpec cluster: true survives the custom Deserialize.
        let test_cmd = cfg.commands.get("cuda-test").expect("cuda-test command");
        assert_eq!(test_cmd.cluster, Some(true));
        let train_cmd = cfg.commands.get("train").expect("train command");
        assert_eq!(train_cmd.cluster, Some(true));
        assert_eq!(train_cmd.run.as_deref(), Some("cargo run --release --bin my-training-app"));
    }

    #[test]
    fn cluster_block_optional() {
        let yaml = "commands: { foo: { run: \"echo hi\" } }\n";
        let cfg: ProjectConfig = serde_yaml::from_str(yaml).expect("parse without cluster");
        assert!(cfg.cluster.is_none());
        // CommandSpec cluster defaults to None.
        assert_eq!(cfg.commands.get("foo").and_then(|c| c.cluster), None);
    }

    #[test]
    fn cluster_overlay_merges_via_deep_merge() {
        // The canonical author pattern: `cluster:` lives in `fdl.vm.yml`
        // (the overlay), not the base `fdl.yml`. deep_merge stitches them.
        let base_yaml = "commands:\n  train:\n    run: cargo run --release\n";
        let overlay_yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /tmp/test-solo
  workers:
    - host: solo
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /tmp/test-solo
commands:
  train:
    cluster: true
";
        let base: serde_yaml::Value = serde_yaml::from_str(base_yaml).unwrap();
        let overlay: serde_yaml::Value = serde_yaml::from_str(overlay_yaml).unwrap();
        let merged = crate::overlay::deep_merge(base, overlay);
        let merged_yaml = serde_yaml::to_string(&merged).unwrap();
        let cfg: ProjectConfig =
            serde_yaml::from_str(&merged_yaml).expect("merged config parses");
        let cluster = cfg.cluster.as_ref().expect("cluster: from overlay");
        assert_eq!(cluster.world_size(), 1);
        assert_eq!(cluster.workers[0].host, "solo");
        // Overlay added `cluster: true` to the train command; the base's
        // `run:` survived the merge.
        let train = cfg.commands.get("train").expect("train command");
        assert_eq!(train.cluster, Some(true));
        assert_eq!(train.run.as_deref(), Some("cargo run --release"));
    }

    #[test]
    fn validate_rejects_duplicate_ranks() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().workers[1].ranks = vec![1, 1];
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("duplicates or gaps"), "got: {err}");
    }

    #[test]
    fn validate_rejects_rank_gap() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().workers[1].ranks = vec![2, 3];
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("duplicates or gaps"), "got: {err}");
    }

    #[test]
    fn validate_rejects_len_mismatch() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().workers[1].local_devices =
            LocalDevices::Explicit(vec![0]);
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("length mismatch"), "got: {err}");
    }

    #[test]
    fn validate_rejects_empty_hosts() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().workers.clear();
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("non-empty"), "got: {err}");
    }

    #[test]
    fn validate_rejects_missing_socket_ifname_when_multi_host() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().workers[0].nccl_socket_ifname = String::new();
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("nccl_socket_ifname"), "got: {err}");
        assert!(err.contains("multiple workers"), "got: {err}");
    }

    #[test]
    fn validate_rejects_empty_path() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().workers[0].path = String::new();
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("path"), "got: {err}");
        assert!(err.contains("master-host"), "got: {err}");
    }

    #[test]
    fn validate_allows_empty_socket_ifname_when_single_host() {
        // Single-host clusters don't go through NCCL TCP, so the ifname
        // requirement doesn't apply.
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /tmp/test-solo
  workers:
    - host: solo
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: \"\"
      path: /tmp/test-solo
";
        let cfg: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.cluster.as_ref().unwrap().validate().expect("single-host with empty ifname must pass");
    }

    #[test]
    fn validate_passes_canonical() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_ref().unwrap().validate().expect("canonical topology must validate");
    }

    #[test]
    fn canonical_json_stable_and_round_trips() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        let cluster = cfg.cluster.as_ref().unwrap();
        let json = cluster.canonical_json().expect("serialize");

        // Stability: serialization twice produces identical output.
        let json2 = cluster.canonical_json().unwrap();
        assert_eq!(json, json2);

        // Round-trip: parsing back gives the same observable state.
        let parsed: ClusterConfig = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(parsed.controller.host, cluster.controller.host);
        assert_eq!(parsed.controller.port, cluster.controller.port);
        assert_eq!(parsed.workers.len(), cluster.workers.len());
        assert_eq!(parsed.world_size(), cluster.world_size());
        assert_eq!(
            parsed.workers[1].ssh.as_ref().and_then(|s| s.target.as_deref()),
            Some("worker-host"),
        );
    }

    #[test]
    fn local_devices_all_yaml_parses_as_marker() {
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /tmp/solo
  workers:
    - host: solo
      ranks: [0, 1]
      local_devices: all
      nccl_socket_ifname: lo
      path: /tmp/solo
";
        let cfg: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        let host = &cfg.cluster.unwrap().workers[0];
        assert_eq!(host.local_devices, LocalDevices::All);
        assert!(host.local_devices.is_all());
        assert!(host.local_devices.as_explicit().is_none());
    }

    #[test]
    fn local_devices_explicit_parses_as_explicit() {
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /tmp/solo
  workers:
    - host: solo
      ranks: [0]
      local_devices: [3]
      nccl_socket_ifname: lo
      path: /tmp/solo
";
        let cfg: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        let host = &cfg.cluster.unwrap().workers[0];
        assert_eq!(host.local_devices, LocalDevices::Explicit(vec![3]));
        assert!(!host.local_devices.is_all());
        assert_eq!(host.local_devices.as_explicit(), Some(&[3u8][..]));
    }

    #[test]
    fn local_devices_all_skips_length_check_in_validate() {
        // With `all`, ranks.len() vs local_devices length is NOT checked at
        // validate-time (resolution is deferred to startup on the target
        // host). Verifies the controller-side validate passes a config with
        // 5 ranks + local_devices: all even though no explicit list exists.
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /tmp/solo
  workers:
    - host: solo
      ranks: [0, 1, 2, 3, 4]
      local_devices: all
      nccl_socket_ifname: lo
      path: /tmp/solo
";
        let cfg: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.cluster
            .as_ref()
            .unwrap()
            .validate()
            .expect("validate must pass for local_devices: all");
    }

    #[test]
    fn local_envelope_for_all_emits_all_string() {
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /tmp/solo
  workers:
    - host: solo
      ranks: [0]
      local_devices: all
      nccl_socket_ifname: lo
      path: /tmp/solo
";
        let cfg: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        let cluster = cfg.cluster.unwrap();
        let env = cluster.local_envelope_for(&cluster.workers[0]);
        assert_eq!(env["worker"]["local_devices"], serde_json::json!("all"));
    }

    #[test]
    fn local_devices_all_round_trips_through_canonical_json() {
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /tmp/solo
  workers:
    - host: solo
      ranks: [0]
      local_devices: all
      nccl_socket_ifname: lo
      path: /tmp/solo
";
        let cfg: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        let cluster = cfg.cluster.unwrap();
        let json = cluster.canonical_json().unwrap();
        let parsed: ClusterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.workers[0].local_devices, LocalDevices::All);
    }

    #[test]
    fn local_devices_rejects_unknown_string() {
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /tmp/solo
  workers:
    - host: solo
      ranks: [0]
      local_devices: every
      nccl_socket_ifname: lo
      path: /tmp/solo
";
        let err = serde_yaml::from_str::<ProjectConfig>(yaml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("local_devices") && msg.contains("every"),
            "expected loud error mentioning the bad value, got: {msg}"
        );
    }

    #[test]
    fn local_envelope_strips_ssh_adds_world_metadata() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        let cluster = cfg.cluster.as_ref().unwrap();
        let worker = &cluster.workers[1];
        let env = cluster.local_envelope_for(worker);

        // Top-level: controller coords + world metadata.
        assert_eq!(env["controller"]["host"], "192.168.122.1");
        assert_eq!(env["controller"]["port"], 29500);
        assert_eq!(env["world_size"], 3); // 1 + 2 ranks
        assert_eq!(env["num_workers"], 2);

        // Worker slice: only this worker's fields.
        let w = &env["worker"];
        assert_eq!(w["host"], "worker-host");
        assert_eq!(w["ranks"], serde_json::json!([1, 2]));
        assert_eq!(w["local_devices"], serde_json::json!([0, 1]));
        assert_eq!(w["nccl_socket_ifname"], "enp1s0");
        assert_eq!(w["path"], "/srv/flodl");
        assert_eq!(w["arch"], "builds/sm61-sm120");

        // ssh: stripped (launcher-only).
        assert!(w.get("ssh").is_none(), "ssh must not appear in envelope");
    }

    #[test]
    fn local_envelope_omits_optional_arch() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        let mut cluster = cfg.cluster.unwrap();
        cluster.workers[0].arch = None;
        let env = cluster.local_envelope_for(&cluster.workers[0]);
        assert!(
            env["worker"].get("arch").is_none(),
            "arch should be omitted when None"
        );
    }

    #[test]
    fn local_envelope_master_host_carries_rank_zero() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        let cluster = cfg.cluster.as_ref().unwrap();
        let master = &cluster.workers[0];
        let env = cluster.local_envelope_for(master);
        let ranks = env["worker"]["ranks"].as_array().unwrap();
        assert!(
            ranks.iter().any(|r| r.as_u64() == Some(0)),
            "master worker's envelope must include rank 0"
        );
    }

    #[test]
    fn ssh_target_defaults_to_name() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        let cluster = cfg.cluster.as_ref().unwrap();
        assert_eq!(cluster.ssh_target(&cluster.workers[0]), "master-host"); // no ssh: → name
        assert_eq!(cluster.ssh_target(&cluster.workers[1]), "worker-host"); // explicit
    }

    // ── Path-inheritance cluster-dispatch resolver ──────────────────


    #[test]
    fn cluster_dispatch_empty_chain_is_false() {
        // No command directives anywhere → default false.
        assert!(!resolve_cluster_dispatch(&[]));
    }

    #[test]
    fn cluster_dispatch_all_unset_is_false() {
        // Every level along the path declined to set cluster: → false.
        assert!(!resolve_cluster_dispatch(&[None, None, None]));
    }

    #[test]
    fn cluster_dispatch_leaf_true_wins_over_unset_ancestors() {
        // root unset, leaf cluster: true → fan out.
        assert!(resolve_cluster_dispatch(&[None, Some(true)]));
    }

    #[test]
    fn cluster_dispatch_leaf_false_wins_over_unset_ancestors() {
        // root unset, leaf cluster: false → explicit local.
        assert!(!resolve_cluster_dispatch(&[None, Some(false)]));
    }

    #[test]
    fn cluster_dispatch_inherits_from_path_command_at_root() {
        // commands.ddp-bench in root: cluster: true. ddp-bench/fdl.yml's
        // leaf command leaves it unset → inherits parent's true.
        assert!(resolve_cluster_dispatch(&[Some(true), None]));
    }

    #[test]
    fn cluster_dispatch_leaf_overrides_ancestor_true_to_false() {
        // root: ddp-bench cluster: true. ddp-bench/cuda: cluster: false
        // (e.g. for a single-host quick smoke test) → stays local despite
        // ancestor saying yes.
        assert!(!resolve_cluster_dispatch(&[Some(true), Some(false)]));
    }

    #[test]
    fn cluster_dispatch_leaf_overrides_ancestor_false_to_true() {
        // root: research/ cluster: false (everything stays local).
        // research/multi-host-experiment: cluster: true (opt back in).
        assert!(resolve_cluster_dispatch(&[Some(false), Some(true)]));
    }

    #[test]
    fn cluster_dispatch_intermediate_override_wins_at_sub_sub_depth() {
        // 4-level chain: root → ddp-bench → sweep → experiment.
        // root true, ddp-bench false (turns ddp-bench off wholesale),
        // sweep unset, experiment unset → effective false. The deepest
        // explicit value (ddp-bench's false) outranks root's true; sweep
        // and experiment inherit from there.
        assert!(!resolve_cluster_dispatch(&[Some(true), Some(false), None, None]));
    }

    #[test]
    fn cluster_dispatch_deepest_override_wins_through_chain() {
        // Same 4-level chain but the experiment opts back in: effective true.
        // Resolution walks leaf-first, finds Some(true) immediately.
        assert!(resolve_cluster_dispatch(&[Some(true), Some(false), None, Some(true)]));
    }

    #[test]
    fn cluster_dispatch_enabled_requires_cluster_block() {
        // Even with leaf cluster: true, dispatching is disabled when no
        // cluster topology is declared at root.
        let no_cluster: ProjectConfig = serde_yaml::from_str(
            "commands: { foo: { run: \"echo hi\" } }\n",
        )
        .unwrap();
        assert!(!cluster_dispatch_enabled(&no_cluster, &[Some(true)]));
        assert!(!cluster_dispatch_enabled(&no_cluster, &[Some(true), Some(true)]));
    }

    #[test]
    fn cluster_dispatch_enabled_when_block_and_chain_agree() {
        let with_cluster: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        assert!(cluster_dispatch_enabled(&with_cluster, &[Some(true)]));
        assert!(cluster_dispatch_enabled(&with_cluster, &[Some(true), None]));
        // Cluster block present but chain says false → not enabled.
        assert!(!cluster_dispatch_enabled(&with_cluster, &[Some(false)]));
        assert!(!cluster_dispatch_enabled(&with_cluster, &[None, Some(false)]));
        // Cluster block present but no directives anywhere → not enabled.
        assert!(!cluster_dispatch_enabled(&with_cluster, &[]));
        assert!(!cluster_dispatch_enabled(&with_cluster, &[None, None]));
    }
