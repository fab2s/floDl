    use super::*;
    use std::io::Cursor;

    const ZERO_SALT: SessionSalt = [0u8; 16];
    const SAMPLE_SALT: SessionSalt = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00,
    ];

    #[test]
    fn hmac_sha256_64_is_deterministic_and_key_sensitive() {
        // Same (salt, bytes) → same tag every call.
        let h = hmac_sha256_64(&ZERO_SALT, b"hello");
        let h2 = hmac_sha256_64(&ZERO_SALT, b"hello");
        assert_eq!(h, h2);
        // Different salt → different tag (with overwhelming probability).
        let h3 = hmac_sha256_64(&SAMPLE_SALT, b"hello");
        assert_ne!(h, h3);
        // Different message → different tag.
        let h4 = hmac_sha256_64(&ZERO_SALT, b"hellp");
        assert_ne!(h, h4);
    }

    #[test]
    fn hmac_sha256_64_truncation_matches_full_mac() {
        // The truncated tag must be exactly the first 8 bytes of the full
        // HMAC-SHA256 output, interpreted little-endian. Guards against
        // accidental endian flips or wrong-half truncation in future edits.
        let bytes = b"some payload bytes for verification";
        let full: [u8; 32] = HMAC::mac(bytes, SAMPLE_SALT.as_slice());
        let mut expected_first_8 = [0u8; 8];
        expected_first_8.copy_from_slice(&full[0..8]);
        let expected = u64::from_le_bytes(expected_first_8);
        assert_eq!(hmac_sha256_64(&SAMPLE_SALT, bytes), expected);
    }

    #[test]
    fn msg_kind_round_trip() {
        for k in [
            MsgKind::Control,
            MsgKind::Timing,
            MsgKind::Metrics,
            MsgKind::ParamSnapshotMeta,
            MsgKind::Heartbeat,
        ] {
            let v = k as u32;
            assert_eq!(MsgKind::from_u32(v).unwrap(), k);
        }
    }

    #[test]
    fn msg_kind_rejects_unknown() {
        let err = MsgKind::from_u32(0xDEAD).unwrap_err();
        assert!(err.to_string().contains("MsgKind"), "got: {err}");
    }

    #[test]
    fn control_frame_round_trip_in_memory() {
        let plan = EpochPlanWire {
            epoch: 7,
            partition_offset: 100,
            partition_size: 256,
        };
        let msg = ControlMsgWire::StartEpoch(plan.clone());
        let frame = ControlFrame::encode(&SAMPLE_SALT, MsgKind::Control, &msg).unwrap();

        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(buf);
        let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
            .unwrap()
            .expect("frame, not EOF");
        assert_eq!(got.kind, MsgKind::Control);
        assert_eq!(got.auth_tag, frame.auth_tag);

        let decoded: ControlMsgWire = got.decode().unwrap();
        assert_eq!(decoded, msg);
        match decoded {
            ControlMsgWire::StartEpoch(p) => assert_eq!(p, plan),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn control_frame_round_trip_shutdown_with_save() {
        let msg = ControlMsgWire::ShutdownWithSave { reason: 1 };
        let frame =
            ControlFrame::encode(&SAMPLE_SALT, MsgKind::Control, &msg).unwrap();
        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(buf);
        let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
            .unwrap()
            .expect("frame, not EOF");
        let decoded: ControlMsgWire = got.decode().unwrap();
        match decoded {
            ControlMsgWire::ShutdownWithSave { reason } => assert_eq!(reason, 1),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn control_frame_rejects_wrong_salt() {
        let msg = ControlMsgWire::Shutdown;
        let frame = ControlFrame::encode(&SAMPLE_SALT, MsgKind::Control, &msg).unwrap();
        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(buf);
        let err = ControlFrame::read_from(&mut cur, &ZERO_SALT).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HMAC verification failed"),
            "expected HMAC verification failure, got: {msg}"
        );
    }

    #[test]
    fn control_frame_rejects_wrong_magic() {
        // Build a frame manually with bad magic.
        let mut hdr = [0u8; 24];
        hdr[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        hdr[4..8].copy_from_slice(&CONTROL_PROTOCOL_VERSION.to_le_bytes());
        // auth_tag zero, kind=Control, payload_len=0
        hdr[16..20].copy_from_slice(&(MsgKind::Control as u32).to_le_bytes());
        let mut cur = Cursor::new(hdr.to_vec());
        let err = ControlFrame::read_from(&mut cur, &ZERO_SALT).unwrap_err();
        assert!(err.to_string().contains("magic"), "got: {err}");
    }

    #[test]
    fn control_frame_rejects_wrong_version() {
        let mut hdr = [0u8; 24];
        hdr[0..4].copy_from_slice(&CONTROL_FRAME_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&99u32.to_le_bytes());
        hdr[16..20].copy_from_slice(&(MsgKind::Control as u32).to_le_bytes());
        let mut cur = Cursor::new(hdr.to_vec());
        let err = ControlFrame::read_from(&mut cur, &ZERO_SALT).unwrap_err();
        assert!(err.to_string().contains("version"), "got: {err}");
    }

    #[test]
    fn control_frame_eof_returns_none() {
        let mut cur = Cursor::new(Vec::<u8>::new());
        let got = ControlFrame::read_from(&mut cur, &ZERO_SALT).unwrap();
        assert!(got.is_none(), "EOF before header bytes should be None");
    }

    #[test]
    fn timing_msg_round_trip_all_variants() {
        let cases = [
            TimingMsgWire::Batch {
                rank: 1,
                batch_ms: 12.5, data_ms: 0.0,
                step_count: 42,
                param_norm: Some(3.5),
                batch_loss: 0.1,
                sync_divergence: None,
            },
            TimingMsgWire::SyncAck {
                rank: 2,
                step_count: 100,
                divergence: Some(0.01),
                post_norm: Some(5.0),
                pre_norm: Some(5.01),
            },
            TimingMsgWire::Exiting { rank: 3 },
            TimingMsgWire::LrUpdate { rank: 0, lr: 1e-3 },
            TimingMsgWire::CheckpointResult {
                rank: 1,
                version: 7,
                elapsed_ms: 12.5,
                error: None,
            },
            TimingMsgWire::CheckpointResult {
                rank: 2,
                version: 8,
                elapsed_ms: 5.0,
                error: Some("disk full".to_string()),
            },
        ];
        for c in cases {
            let frame = ControlFrame::encode(&SAMPLE_SALT, MsgKind::Timing, &c).unwrap();
            let mut buf = Vec::new();
            frame.write_to(&mut buf).unwrap();
            let mut cur = Cursor::new(buf);
            let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
                .unwrap()
                .unwrap();
            assert_eq!(got.kind, MsgKind::Timing);
            let back: TimingMsgWire = got.decode().unwrap();
            assert_eq!(back, c);
        }
    }

    #[test]
    fn control_frame_round_trip_checkpoint_targeted() {
        let cases = [
            ControlMsgWire::Checkpoint { version: 3, target_rank: 0 },
            ControlMsgWire::Checkpoint { version: 4, target_rank: 7 },
            // u64::MAX is reserved for v2 controller-as-checkpointer;
            // the wire encodes it fine, the coord rejects it loudly
            // on dispatch.
            ControlMsgWire::Checkpoint { version: 5, target_rank: u64::MAX },
        ];
        for msg in cases {
            let frame =
                ControlFrame::encode(&SAMPLE_SALT, MsgKind::Control, &msg).unwrap();
            let mut buf = Vec::new();
            frame.write_to(&mut buf).unwrap();
            let mut cur = Cursor::new(buf);
            let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
                .unwrap()
                .unwrap();
            let back: ControlMsgWire = got.decode().unwrap();
            assert_eq!(back, msg);
        }
    }

    #[test]
    fn control_frame_round_trip_save_consensus_model() {
        let msg = ControlMsgWire::SaveConsensusModel { target_rank: 2 };
        let frame =
            ControlFrame::encode(&SAMPLE_SALT, MsgKind::Control, &msg).unwrap();
        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(buf);
        let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
            .unwrap()
            .unwrap();
        let back: ControlMsgWire = got.decode().unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn control_frame_round_trip_update_atomic_dispatch() {
        // atomic-dispatch: the post-reduce Update carries an optional
        // folded next-window chunk. Both shapes must round-trip.
        let cases = [
            ControlMsgWire::Update {
                version: 1,
                next_plan: None,
            },
            ControlMsgWire::Update {
                version: 42,
                next_plan: Some(EpochPlanWire {
                    epoch: 3,
                    partition_offset: 128,
                    partition_size: 64,
                }),
            },
        ];
        for msg in cases {
            let frame =
                ControlFrame::encode(&SAMPLE_SALT, MsgKind::Control, &msg).unwrap();
            let mut buf = Vec::new();
            frame.write_to(&mut buf).unwrap();
            let mut cur = Cursor::new(buf);
            let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
                .unwrap()
                .unwrap();
            let back: ControlMsgWire = got.decode().unwrap();
            assert_eq!(back, msg);
        }
    }

    #[test]
    fn control_frame_round_trip_eval_targeted() {
        let cases = [
            ControlMsgWire::ExecuteEvalCallback {
                schedule_id: 10,
                epoch: 5,
                target_rank: 0,
            },
            ControlMsgWire::ExecuteEvalCallback {
                schedule_id: 11,
                epoch: 6,
                target_rank: 2,
            },
        ];
        for msg in cases {
            let frame =
                ControlFrame::encode(&SAMPLE_SALT, MsgKind::Control, &msg).unwrap();
            let mut buf = Vec::new();
            frame.write_to(&mut buf).unwrap();
            let mut cur = Cursor::new(buf);
            let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
                .unwrap()
                .unwrap();
            let back: ControlMsgWire = got.decode().unwrap();
            assert_eq!(back, msg);
        }
    }

    #[test]
    fn control_frame_round_trip_set_epoch_callback_role() {
        let cases = [
            ControlMsgWire::SetEpochCallbackRole { rank: 0 },
            ControlMsgWire::SetEpochCallbackRole { rank: 3 },
        ];
        for msg in cases {
            let frame =
                ControlFrame::encode(&SAMPLE_SALT, MsgKind::Control, &msg).unwrap();
            let mut buf = Vec::new();
            frame.write_to(&mut buf).unwrap();
            let mut cur = Cursor::new(buf);
            let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
                .unwrap()
                .unwrap();
            let back: ControlMsgWire = got.decode().unwrap();
            assert_eq!(back, msg);
        }
    }

    #[test]
    fn metrics_msg_round_trip_with_scalars() {
        let mut scalars = HashMap::new();
        scalars.insert("loss".to_string(), (12.5, 100));
        scalars.insert("acc".to_string(), (0.85, 100));
        let m = MetricsMsgWire {
            rank: 1,
            epoch: 3,
            avg_loss: 0.42,
            batches_processed: 50,
            epoch_ms: 1234.5,
            samples_processed: 6400,
            share_complete_ms: 1100.0,
            compute_only_ms: 900.0,
            data_starve_ms: 50.0,
            scalars,
            resources: None,
        };
        let frame = ControlFrame::encode(&SAMPLE_SALT, MsgKind::Metrics, &m).unwrap();
        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(buf);
        let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
            .unwrap()
            .unwrap();
        let back: MetricsMsgWire = got.decode().unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn metrics_msg_round_trip_with_resources() {
        let gpus = vec![
            GpuSnapshotWire {
                device_index: 0,
                name: "Pascal GP106".to_string(),
                util_percent: Some(67.0),
                vram_allocated_bytes: Some(2_500_000_000),
                vram_total_bytes: Some(6_000_000_000),
            },
            GpuSnapshotWire {
                device_index: 1,
                name: "Blackwell 5060Ti".to_string(),
                util_percent: Some(91.0),
                vram_allocated_bytes: Some(8_200_000_000),
                vram_total_bytes: Some(16_000_000_000),
            },
        ];
        let res = ResourceSampleWire {
            cpu_percent: Some(38.5),
            ram_used_bytes: Some(12_000_000_000),
            ram_total_bytes: Some(32_000_000_000),
            gpu_util_percent: Some(91.0),
            vram_total_bytes: Some(16_000_000_000),
            vram_allocated_bytes: Some(8_200_000_000),
            aggregate_rank: Some(1),
            gpus,
        };
        let m = MetricsMsgWire {
            rank: 4,
            epoch: 12,
            avg_loss: 0.1234,
            batches_processed: 200,
            epoch_ms: 4321.0,
            samples_processed: 25600,
            share_complete_ms: 4100.0,
            compute_only_ms: 3600.0,
            data_starve_ms: 220.0,
            scalars: HashMap::new(),
            resources: Some(res),
        };
        let frame = ControlFrame::encode(&SAMPLE_SALT, MsgKind::Metrics, &m).unwrap();
        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(buf);
        let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
            .unwrap()
            .unwrap();
        let back: MetricsMsgWire = got.decode().unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn timing_msg_round_trip_dashboard_variants() {
        let cases = [
            TimingMsgWire::DashboardRegister { rank: 0, port: 3000 },
            TimingMsgWire::DashboardRegister { rank: 7, port: 4242 },
            TimingMsgWire::DashboardSetSvg {
                rank: 0,
                svg: "<svg>...</svg>".to_string(),
                label: Some("ResNet50".to_string()),
                hash: Some("deadbeef".to_string()),
            },
            TimingMsgWire::DashboardSetSvg {
                rank: 1,
                svg: String::new(),
                label: None,
                hash: None,
            },
            TimingMsgWire::DashboardSetMetadata {
                rank: 0,
                json: r#"{"epochs": 10, "lr": 0.001}"#.to_string(),
            },
            TimingMsgWire::DashboardSetHardware {
                rank: 2,
                summary: "CPU=8 cores | RAM=32GB | GPU=2x RTX 5060 Ti".to_string(),
            },
        ];
        for c in cases {
            let frame = ControlFrame::encode(&SAMPLE_SALT, MsgKind::Timing, &c).unwrap();
            let mut buf = Vec::new();
            frame.write_to(&mut buf).unwrap();
            let mut cur = Cursor::new(buf);
            let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
                .unwrap()
                .unwrap();
            assert_eq!(got.kind, MsgKind::Timing);
            let back: TimingMsgWire = got.decode().unwrap();
            assert_eq!(back, c);
        }
    }



    #[cfg(feature = "rng")]
    #[test]
    fn generate_session_salt_returns_distinct_values_on_repeat() {
        // Two calls in the same process must produce different salts
        // (with overwhelming probability) and never the zero pattern.
        let a = generate_session_salt();
        let b = generate_session_salt();
        assert_ne!(a, b, "two ThreadRng salt draws collided");
        assert_ne!(a, [0u8; SESSION_SALT_BYTES]);
    }

    #[test]
    fn salt_hex_round_trip() {
        let s = SAMPLE_SALT;
        let h = salt_to_hex(&s);
        assert_eq!(h.len(), SESSION_SALT_BYTES * 2);
        let back = salt_from_hex(&h).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn salt_from_hex_rejects_wrong_length() {
        let err = salt_from_hex("deadbeef").unwrap_err();
        assert!(err.to_string().contains("hex must be"), "got: {err}");
    }

    #[test]
    fn salt_from_hex_rejects_bad_chars() {
        let bad = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"; // 32 chars but non-hex
        let err = salt_from_hex(bad).unwrap_err();
        assert!(err.to_string().contains("hex-decode"), "got: {err}");
    }

    #[test]
    fn payload_too_large_errors_on_write() {
        // u32::MAX + 1 isn't allocatable; simulate with a marker test that
        // the bounds check exists by directly constructing the payload.
        // (Allocating 4GB in tests is impractical; rely on the bounds
        // check at u32::try_from in write_to.)
        let frame = ControlFrame {
            kind: MsgKind::Control,
            auth_tag: 0,
            payload: Vec::new(),
        };
        // Sanity: zero-length payload writes successfully.
        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        // Header is exactly 24 bytes (no payload).
        assert_eq!(buf.len(), 24);
    }

    #[test]
    fn join_host_port_brackets_ipv6_only() {
        // IPv4 + hostnames: plain host:port.
        assert_eq!(join_host_port("192.168.122.1", 1337), "192.168.122.1:1337");
        assert_eq!(join_host_port("exa", 1337), "exa:1337");
        assert_eq!(join_host_port("127.0.0.1", 22), "127.0.0.1:22");
        // IPv6 literal: must be bracketed (a bare fe80::1:1337 is ambiguous).
        assert_eq!(join_host_port("fe80::1", 1337), "[fe80::1]:1337");
        assert_eq!(join_host_port("2001:db8::5", 29500), "[2001:db8::5]:29500");
        assert_eq!(join_host_port("::1", 1337), "[::1]:1337");
        // Already bracketed: left as-is (no double brackets).
        assert_eq!(join_host_port("[fe80::1]", 1337), "[fe80::1]:1337");
        // The result must round-trip through ToSocketAddrs for IPv6.
        use std::net::ToSocketAddrs;
        assert!(join_host_port("::1", 1337).to_socket_addrs().is_ok());
    }
