    use super::*;
    use serde_json::json;

    fn worker_envelope() -> Value {
        json!({
            "controller": { "host": "192.168.122.1", "port": 29500 },
            "world_size": 3,
            "num_workers": 2,
            "worker": {
                "host": "host-b",
                "ranks": [1, 2],
                "local_devices": [0, 1],
                "nccl_socket_ifname": "enp1s0",
                "path": "/srv/flodl",
                "arch": "builds/sm61-sm120"
            }
        })
    }

    fn master_envelope() -> Value {
        json!({
            "controller": { "host": "192.168.122.1", "port": 29500 },
            "world_size": 3,
            "num_workers": 2,
            "worker": {
                "host": "host-a",
                "ranks": [0],
                "local_devices": [0],
                "nccl_socket_ifname": "virbr0",
                "path": "/opt/flodl",
                "arch": "precompiled/cu128"
            }
        })
    }

    #[test]
    fn parses_canonical_envelope() {
        let c = LocalCluster::from_value(&worker_envelope()).expect("parse");
        assert_eq!(c.controller.host, "192.168.122.1");
        assert_eq!(c.controller.port, 29500);
        assert_eq!(c.world_size(), 3);
        assert_eq!(c.num_workers, 2);
        assert!(c.spans_multiple_workers());

        assert_eq!(c.worker.host, "host-b");
        assert_eq!(c.worker.ranks, vec![1, 2]);
        assert_eq!(c.worker.local_devices, vec![0, 1]);
        assert_eq!(c.worker.nccl_socket_ifname, "enp1s0");
        assert_eq!(c.worker.path, "/srv/flodl");
        assert_eq!(c.worker.arch.as_deref(), Some("builds/sm61-sm120"));
    }

    #[test]
    fn rejects_local_devices_all_when_cuda_devices_insufficient() {
        // `local_devices: "all"` resolves on the host that receives the
        // envelope, using cuda_device_count(). With ranks of length > visible
        // CUDA devices (always true in CPU test mode, where count == 0), the
        // resolver must error loudly mentioning the count.
        let mut v = worker_envelope();
        v["worker"]["local_devices"] = json!("all");
        let err = LocalCluster::from_value(&v).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("local_devices") && msg.contains("\"all\""),
            "expected loud error mentioning local_devices: all, got: {msg}"
        );
    }

    #[test]
    fn parses_local_devices_all_when_cuda_sufficient() {
        // Success path: only meaningful when CUDA is available with enough
        // devices for the rank count. Self-skip in CPU mode.
        let avail = crate::tensor::cuda_device_count();
        if avail < 1 {
            eprintln!("cuda_device_count() = {avail}; skipping all-shorthand success test");
            return;
        }
        // Build an envelope with ranks_len = 1 (always satisfiable when at
        // least one CUDA device is visible).
        let mut v = master_envelope();
        v["worker"]["local_devices"] = json!("all");
        let c = LocalCluster::from_value(&v).expect("parse with local_devices: all");
        // Resolved indices are 0..ranks_len; ranks_len = 1 in master_envelope.
        assert_eq!(c.worker.local_devices, vec![0u8]);
    }

    #[test]
    fn rejects_local_devices_unknown_string() {
        let mut v = worker_envelope();
        v["worker"]["local_devices"] = json!("every");
        let err = LocalCluster::from_value(&v).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("local_devices") && msg.contains("every"),
            "expected loud error mentioning the bad value, got: {msg}"
        );
    }

    #[test]
    fn rejects_missing_path() {
        let mut v = worker_envelope();
        v["worker"].as_object_mut().unwrap().remove("path");
        let err = LocalCluster::from_value(&v).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("path"), "got: {msg}");
        assert!(msg.contains("host-b"), "got: {msg}");
    }

    #[test]
    fn rejects_ranks_devices_len_mismatch() {
        let mut v = worker_envelope();
        v["worker"]["ranks"] = json!([1]);
        v["worker"]["local_devices"] = json!([0, 1]);
        let err = LocalCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("length mismatch"), "got: {err}");
    }

    #[test]
    fn rejects_rank_out_of_bounds() {
        let mut v = worker_envelope();
        v["worker"]["ranks"] = json!([1, 5]);
        v["worker"]["local_devices"] = json!([0, 1]);
        let err = LocalCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("out of bounds"), "got: {err}");
    }

    #[test]
    fn rejects_zero_world_size() {
        let mut v = worker_envelope();
        v["world_size"] = json!(0);
        let err = LocalCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("world_size"), "got: {err}");
    }

    #[test]
    fn rejects_num_workers_exceeding_world_size() {
        let mut v = worker_envelope();
        v["num_workers"] = json!(99);
        let err = LocalCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("num_workers"), "got: {err}");
    }

    #[test]
    fn rejects_controller_port_overflow() {
        let mut v = worker_envelope();
        v["controller"]["port"] = json!(100_000); // > u16::MAX
        let err = LocalCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("u16"), "got: {err}");
    }

    #[test]
    fn this_worker_matches_envelope() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: ENV_MUTEX serializes env-mutating tests in this module.
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-b");
        }
        let h = c.this_worker().expect("hostname matches");
        assert_eq!(h.host, "host-b");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
    }

    #[test]
    fn this_worker_loud_error_on_mismatch() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "wrong-host");
        }
        let err = c.this_worker().unwrap_err();
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
        let msg = err.to_string();
        assert!(msg.contains("wrong-host"), "got: {msg}");
        assert!(msg.contains("host-b"), "got: {msg}");
        assert!(msg.contains(ENV_HOST_OVERRIDE), "got: {msg}");
    }

    #[test]
    fn thread_local_override_beats_env_var() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "wrong-host");
        }
        // Thread-local takes precedence; even though env says "wrong-host",
        // the thread-local says "host-b" which matches the envelope.
        set_thread_hostname_override(Some("host-b"));
        let h = c.this_worker().expect("thread-local wins");
        assert_eq!(h.host, "host-b");
        set_thread_hostname_override(None);
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
    }

    #[test]
    fn from_env_returns_none_when_unset() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::remove_var(ENV_CLUSTER_JSON);
        }
        assert!(LocalCluster::from_env().unwrap().is_none());
    }

    #[test]
    fn from_env_round_trips_hex() {
        let v = worker_envelope();
        let bytes = serde_json::to_vec(&v).unwrap();
        let hex = hex_encode(&bytes);

        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_CLUSTER_JSON, &hex);
        }
        let c = LocalCluster::from_env().expect("decode ok").expect("Some");
        unsafe {
            env::remove_var(ENV_CLUSTER_JSON);
        }
        assert_eq!(c.worker.host, "host-b");
        assert_eq!(c.world_size, 3);
        assert_eq!(c.num_workers, 2);
    }

    #[test]
    fn from_env_rejects_bad_hex() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_CLUSTER_JSON, "not-valid-hex-zz");
        }
        let err = LocalCluster::from_env().unwrap_err();
        unsafe {
            env::remove_var(ENV_CLUSTER_JSON);
        }
        assert!(err.to_string().contains("hex-decode"), "got: {err}");
    }

    #[test]
    fn hex_round_trip() {
        let data = b"\x00\x0fhello\xff\xab";
        let h = hex_encode(data);
        assert_eq!(h, "000f68656c6c6fffab");
        let back = hex_decode(&h).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn hex_decode_uppercase() {
        let back = hex_decode("FF0A").unwrap();
        assert_eq!(back, vec![0xFF, 0x0A]);
    }

    #[test]
    fn hex_decode_rejects_odd_length() {
        assert!(hex_decode("abc").is_err());
    }

    #[test]
    fn my_rank_picks_first_slot() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-b");
            env::set_var(ENV_LOCAL_RANK, "0");
        }
        let (global_rank, device) = c.my_rank().expect("my_rank ok");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        // worker_envelope: ranks=[1,2], local_devices=[0,1]. Index 0 -> (1, CUDA(0)).
        assert_eq!(global_rank, 1);
        assert_eq!(device, Device::CUDA(0));
    }

    #[test]
    fn my_rank_picks_second_slot() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-b");
            env::set_var(ENV_LOCAL_RANK, "1");
        }
        let (global_rank, device) = c.my_rank().expect("my_rank ok");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        // worker_envelope: index 1 -> (2, CUDA(1)).
        assert_eq!(global_rank, 2);
        assert_eq!(device, Device::CUDA(1));
    }

    #[test]
    fn my_rank_loud_error_when_env_unset() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-b");
            env::remove_var(ENV_LOCAL_RANK);
        }
        let err = c.my_rank().unwrap_err();
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
        let msg = err.to_string();
        assert!(msg.contains(ENV_LOCAL_RANK), "got: {msg}");
        assert!(msg.contains("not set"), "got: {msg}");
    }

    #[test]
    fn my_rank_loud_error_on_unparseable_value() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-b");
            env::set_var(ENV_LOCAL_RANK, "not-a-number");
        }
        let err = c.my_rank().unwrap_err();
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        let msg = err.to_string();
        assert!(msg.contains(ENV_LOCAL_RANK), "got: {msg}");
        assert!(msg.contains("not-a-number"), "got: {msg}");
    }

    #[test]
    fn my_rank_loud_error_on_oob_index() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-b");
            env::set_var(ENV_LOCAL_RANK, "5"); // host owns 2 ranks (indexes 0,1)
        }
        let err = c.my_rank().unwrap_err();
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        let msg = err.to_string();
        assert!(msg.contains("out of bounds"), "got: {msg}");
        assert!(msg.contains("host-b"), "got: {msg}");
        // The error names the valid range so the user can fix the launcher.
        assert!(msg.contains("0..2"), "got: {msg}");
    }

    #[test]
    fn my_rank_accepts_whitespace_padded_value() {
        // The launcher emits unquoted numeric, but defending against
        // user-set values with stray whitespace is cheap.
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-b");
            env::set_var(ENV_LOCAL_RANK, "  1  ");
        }
        let (gr, _) = c.my_rank().expect("trimmed parse ok");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        assert_eq!(gr, 2);
    }

    #[test]
    fn my_rank_thread_local_override_beats_env() {
        // Threaded multi-rank tests set distinct thread-local rank overrides
        // per thread; env vars are process-wide and would conflict. The
        // thread-local must win.
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-b");
            env::set_var(ENV_LOCAL_RANK, "0"); // env says 0, override says 1
        }
        set_thread_local_rank_override(Some(1));
        let (global_rank, device) = c.my_rank().expect("my_rank ok");
        set_thread_local_rank_override(None);
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        // worker_envelope: index 1 -> (2, CUDA(1)). Override wins over env=0.
        assert_eq!(global_rank, 2);
        assert_eq!(device, Device::CUDA(1));
    }

    #[test]
    fn my_rank_thread_local_override_works_without_env() {
        // In threaded tests, env var is typically unset; the override is the
        // sole source of the local-rank index.
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-b");
            env::remove_var(ENV_LOCAL_RANK);
        }
        set_thread_local_rank_override(Some(0));
        let (global_rank, device) = c.my_rank().expect("my_rank ok");
        set_thread_local_rank_override(None);
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
        assert_eq!(global_rank, 1);
        assert_eq!(device, Device::CUDA(0));
    }

    #[test]
    fn my_rank_thread_local_override_clears_back_to_env() {
        // After clearing the override (passing None), the env path takes
        // over -- and absent env should produce the canonical loud error.
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-b");
            env::remove_var(ENV_LOCAL_RANK);
        }
        set_thread_local_rank_override(Some(0));
        let _ = c.my_rank().expect("override path ok");
        set_thread_local_rank_override(None);
        let err = c.my_rank().unwrap_err();
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
        // No override, no env -> loud error mentioning the env var.
        assert!(err.to_string().contains(ENV_LOCAL_RANK), "got: {err}");
        assert!(err.to_string().contains("not set"), "got: {err}");
    }

    #[test]
    fn my_rank_thread_local_override_oob_still_bounds_checked() {
        // The bounds check runs regardless of source: an out-of-bounds
        // override index produces the same loud error as an out-of-bounds
        // env value. Catches test bugs (wrong index passed to the helper).
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-b");
            env::remove_var(ENV_LOCAL_RANK);
        }
        set_thread_local_rank_override(Some(99)); // host owns 2 ranks
        let err = c.my_rank().unwrap_err();
        set_thread_local_rank_override(None);
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
        let msg = err.to_string();
        assert!(msg.contains("out of bounds"), "got: {msg}");
        assert!(msg.contains("0..2"), "got: {msg}");
    }

    #[test]
    fn my_rank_single_rank_host() {
        // master_envelope: ranks=[0], local_devices=[0]. Single-rank host.
        // FLODL_LOCAL_RANK=0 still must be set per the explicit-overlay
        // contract -- launcher always injects it.
        let c = LocalCluster::from_value(&master_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "host-a");
            env::set_var(ENV_LOCAL_RANK, "0");
        }
        let (global_rank, device) = c.my_rank().expect("single-rank ok");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        assert_eq!(global_rank, 0);
        assert_eq!(device, Device::CUDA(0));
    }

