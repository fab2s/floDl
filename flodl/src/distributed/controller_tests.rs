    use super::*;
    use std::net::Ipv4Addr;

    /// Deterministic non-zero test salt: exercises the HMAC path (zero
    /// salt is degenerate enough that an accidental "skip the HMAC"
    /// regression could silently still produce all-zero footers and
    /// "pass" — a non-zero salt catches that).
    const TEST_SALT: SessionSalt = [
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    ];

    /// Fake per-host relay speaking the fold protocol: connects, sends
    /// `RelayHello` for `ranks`, then per round folds its local ranks'
    /// frames via [`sum_frames`] (exactly like the production relay) and
    /// ships ONE `HostFrame` up, collecting the round's single
    /// `Broadcast` consensus back. Returns, per rank (parallel to
    /// `ranks`), the consensus frames received — replicated per rank the
    /// way the production relay's fan-out delivers them.
    fn fake_relay(
        port: u16,
        ranks: Vec<u32>,
        salt: SessionSalt,
        per_rank_frames: Vec<Vec<RoundFrame>>,
    ) -> Result<Vec<Vec<RoundFrame>>> {
        assert_eq!(ranks.len(), per_rank_frames.len());
        let n_rounds = per_rank_frames.first().map(|v| v.len()).unwrap_or(0);
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
        let mut stream = TcpStream::connect(addr)
            .map_err(|e| TensorError::new(&format!("fake_relay: connect: {e}")))?;
        stream.set_nodelay(true).ok();

        // Channel-select magic, then the relay handshake.
        crate::distributed::wire::write_channel_magic(
            &mut stream,
            crate::distributed::wire::CHANNEL_MAGIC_DATA,
        )?;
        MuxRecord::control(RelayControlMsg::Hello {
            host: "test-host".into(),
            ranks: ranks.clone(),
            model_sigs: vec![],
        })
        .write_to(&mut stream, &salt)?;
        match MuxRecord::read_from(&mut stream, &salt)? {
            Some(MuxRecord::Control(RelayControlMsg::HelloAck)) => {}
            other => {
                return Err(TensorError::new(&format!(
                    "fake_relay: expected HelloAck, got {other:?}"
                )));
            }
        }

        let mut received: Vec<Vec<RoundFrame>> = ranks.iter().map(|_| Vec::new()).collect();
        for r in 0..n_rounds {
            // Fold the host's local contributions and ship ONE HostFrame.
            let round_frames: Vec<&RoundFrame> =
                per_rank_frames.iter().map(|frames| &frames[r]).collect();
            let folded = sum_frames(&round_frames)?;
            let mut buf = Vec::new();
            write_round_frame(&mut buf, &folded, &salt)?;
            MuxRecord::host_frame(buf).write_to(&mut stream, &salt)?;
            // Collect the round's single Broadcast consensus; fan it out
            // to every local rank slot like the production relay does.
            match MuxRecord::read_from(&mut stream, &salt)? {
                Some(MuxRecord::Broadcast { payload }) => {
                    let frame = read_round_frame(&mut payload.as_slice(), &salt)?
                        .ok_or_else(|| {
                            TensorError::new("fake_relay: truncated consensus frame")
                        })?;
                    for slot in received.iter_mut() {
                        slot.push(frame.clone());
                    }
                }
                other => {
                    return Err(TensorError::new(&format!(
                        "fake_relay: expected Broadcast reply, got {other:?}"
                    )));
                }
            }
        }
        // Drop stream → relay-conn EOF → controller clean shutdown.
        Ok(received)
    }

    fn one_tensor_frame(data: &[f32]) -> RoundFrame {
        RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_F32,
                shape: vec![data.len() as u32],
                bytes: f32_to_bytes(data),
            }],
            // Equal-mass contributions: the realized-work reduce then
            // returns the plain mean over the accepted cohort.
            weight: 1.0,
            ..Default::default()
        }
    }

    fn two_tensor_frame(a: &[f32], b: &[f32]) -> RoundFrame {
        RoundFrame {
            tensors: vec![
                TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![a.len() as u32],
                    bytes: f32_to_bytes(a),
                },
                TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![b.len() as u32],
                    bytes: f32_to_bytes(b),
                },
            ],
            // Equal-mass contributions (see `one_tensor_frame`).
            weight: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn two_rank_average_one_round() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        // One relay carries both ranks over a single connection.
        let recv = fake_relay(
            port,
            vec![0, 1],
            TEST_SALT,
            vec![
                vec![one_tensor_frame(&[1.0, 2.0, 3.0])],
                vec![one_tensor_frame(&[3.0, 4.0, 5.0])],
            ],
        )
        .unwrap();
        avg.shutdown().unwrap();

        // Average of (1,2,3) and (3,4,5) = (2,3,4)
        let expected = bytes_as_f32(&recv[0][0].tensors[0].bytes).unwrap();
        assert_eq!(expected, vec![2.0, 3.0, 4.0]);
        // Both ranks receive the same averaged frame.
        assert_eq!(recv[0][0], recv[1][0]);
    }

    #[test]
    fn three_rank_average_multi_round_multi_tensor() {
        // Three ranks (one relay), two rounds each, each round carries two
        // tensors. Exercises multi-rank star summation, the multi-round
        // reduce loop, multi-tensor frames, and clean shutdown on EOF.
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            3,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        let r0_frames = vec![
            two_tensor_frame(&[0.0, 10.0], &[1.0, 1.0]),
            two_tensor_frame(&[0.0, 5.0], &[0.5, 0.5]),
        ];
        let r1_frames = vec![
            two_tensor_frame(&[10.0, 20.0], &[2.0, 2.0]),
            two_tensor_frame(&[5.0, 10.0], &[1.0, 1.0]),
        ];
        let r2_frames = vec![
            two_tensor_frame(&[20.0, 30.0], &[3.0, 3.0]),
            two_tensor_frame(&[10.0, 15.0], &[1.5, 1.5]),
        ];

        let recv = fake_relay(
            port,
            vec![0, 1, 2],
            TEST_SALT,
            vec![r0_frames, r1_frames, r2_frames],
        )
        .unwrap();
        avg.shutdown().unwrap();

        // Each rank received exactly 2 averaged frames.
        assert_eq!(recv[0].len(), 2, "rank 0 should receive 2 averaged frames");
        assert_eq!(recv[1].len(), 2);
        assert_eq!(recv[2].len(), 2);

        // Round 1 averages: tensor 0 = (10, 20), tensor 1 = (2, 2)
        let r1_t0 = bytes_as_f32(&recv[0][0].tensors[0].bytes).unwrap();
        let r1_t1 = bytes_as_f32(&recv[0][0].tensors[1].bytes).unwrap();
        assert_eq!(r1_t0, vec![10.0, 20.0]);
        assert_eq!(r1_t1, vec![2.0, 2.0]);

        // Round 2 averages: tensor 0 = (5, 10), tensor 1 = (1, 1)
        let r2_t0 = bytes_as_f32(&recv[0][1].tensors[0].bytes).unwrap();
        let r2_t1 = bytes_as_f32(&recv[0][1].tensors[1].bytes).unwrap();
        assert_eq!(r2_t0, vec![5.0, 10.0]);
        assert_eq!(r2_t1, vec![1.0, 1.0]);

        // All three ranks see bit-identical averaged frames.
        assert_eq!(recv[0], recv[1]);
        assert_eq!(recv[1], recv[2]);
    }

    #[test]
    fn rejects_non_hello_first_record() {
        // The controller's phase-1 expects a RelayHello as the first
        // record; a Data record up front must be rejected (connection
        // dropped, no HelloAck).
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            1,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        let mut s =
            TcpStream::connect(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)).unwrap();
        crate::distributed::wire::write_channel_magic(
            &mut s,
            crate::distributed::wire::CHANNEL_MAGIC_DATA,
        )
        .unwrap();
        MuxRecord::data(0, vec![1, 2, 3])
            .write_to(&mut s, &TEST_SALT)
            .unwrap();
        // Controller rejects + drops us: the HelloAck never arrives.
        let ack = MuxRecord::read_from(&mut s, &TEST_SALT);
        assert!(
            matches!(ack, Ok(None)) || ack.is_err(),
            "controller should drop the connection, got {ack:?}"
        );
        drop(s);
        let _ = avg.shutdown(); // phase-1 error propagates; ignore here
    }

    #[test]
    fn rejects_rank_out_of_range() {
        // Relay announces a rank >= world_size: loud phase-1 rejection.
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            1,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        let mut s =
            TcpStream::connect(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)).unwrap();
        crate::distributed::wire::write_channel_magic(
            &mut s,
            crate::distributed::wire::CHANNEL_MAGIC_DATA,
        )
        .unwrap();
        MuxRecord::control(RelayControlMsg::Hello {
            host: "rogue".into(),
            ranks: vec![5], // >= world_size (1)
            model_sigs: vec![],
        })
        .write_to(&mut s, &TEST_SALT)
        .unwrap();
        let ack = MuxRecord::read_from(&mut s, &TEST_SALT);
        assert!(
            matches!(ack, Ok(None)) || ack.is_err(),
            "controller should reject out-of-range rank, got {ack:?}"
        );
        drop(s);
        let _ = avg.shutdown();
    }

    #[test]
    fn rejects_non_f32_dtype_in_reduce() {
        // Pure unit test of reduce_realized_work without TCP wiring.
        let frames = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: 7, // bogus dtype
                    shape: vec![2],
                    bytes: vec![0; 8],
                }],
                ..Default::default()
            }),
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: 7,
                    shape: vec![2],
                    bytes: vec![0; 8],
                }],
                ..Default::default()
            }),
        ];
        let err = reduce_realized_work(&frames).unwrap_err();
        assert!(
            err.to_string().contains("dtype tag 7"),
            "expected dtype-tag-7-not-supported, got: {err}"
        );
    }

    #[test]
    fn rejects_shape_mismatch_across_ranks() {
        let frames = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![2],
                    bytes: f32_to_bytes(&[1.0, 2.0]),
                }],
                ..Default::default()
            }),
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![3],
                    bytes: f32_to_bytes(&[1.0, 2.0, 3.0]),
                }],
                ..Default::default()
            }),
        ];
        let err = reduce_realized_work(&frames).unwrap_err();
        assert!(err.to_string().contains("shape"), "got: {err}");
    }

    #[test]
    fn reduce_realized_work_normalizes_by_accepted_mass_only() {
        // 3-rank world, rank 1 dead (None). Contributions are pre-scaled
        // by the sender's mass (3 and 1); the divisor is the accepted
        // mass sum (4) — the dead rank enters neither sum nor divisor.
        let frames = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![2],
                    bytes: f32_to_bytes(&[3.0, 6.0]), // 3 × [1, 2]
                }],
                weight: 3.0,
                ..Default::default()
            }),
            None, // rank 1 dead — its work was never realized
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![2],
                    bytes: f32_to_bytes(&[5.0, 10.0]), // 1 × [5, 10]
                }],
                weight: 1.0,
                ..Default::default()
            }),
        ];
        let out = reduce_realized_work(&frames).unwrap();
        let consensus = bytes_as_f32(&out.tensors[0].bytes).unwrap();
        // (3·1 + 1·5) / 4 = 2.0; (3·2 + 1·10) / 4 = 4.0
        assert!((consensus[0] - 2.0).abs() < 1e-6, "got {consensus:?}");
        assert!((consensus[1] - 4.0).abs() < 1e-6, "got {consensus:?}");
        assert!(
            (out.weight - 4.0).abs() < 1e-9,
            "accepted mass, got {}",
            out.weight
        );
    }

    #[test]
    fn reduce_realized_work_control_is_pure_sum() {
        // Control frames (gathers / broadcasts) sum without a divide,
        // whatever the weights say.
        let frames = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![2],
                    bytes: f32_to_bytes(&[3.0, 0.0]),
                }],
                kind: RoundKind::Control,
                ..Default::default()
            }),
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![2],
                    bytes: f32_to_bytes(&[0.0, 5.0]),
                }],
                kind: RoundKind::Control,
                ..Default::default()
            }),
        ];
        let out = reduce_realized_work(&frames).unwrap();
        let sum = bytes_as_f32(&out.tensors[0].bytes).unwrap();
        assert!((sum[0] - 3.0).abs() < 1e-6, "got {sum:?}");
        assert!((sum[1] - 5.0).abs() < 1e-6, "got {sum:?}");
    }

    #[test]
    fn reduce_realized_work_zero_mass_returns_untouched_sum() {
        // A Model round whose accepted mass is zero (all contributors
        // idle) must not divide — the output carries weight 0.0 so
        // receivers keep local state.
        let frames = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![1],
                    bytes: f32_to_bytes(&[0.0]),
                }],
                weight: 0.0,
                ..Default::default()
            }),
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![1],
                    bytes: f32_to_bytes(&[0.0]),
                }],
                weight: 0.0,
                ..Default::default()
            }),
        ];
        let out = reduce_realized_work(&frames).unwrap();
        assert_eq!(out.weight, 0.0);
        assert!(bytes_as_f32(&out.tensors[0].bytes).unwrap()[0].abs() < 1e-9);
    }

    #[test]
    fn reduce_realized_work_rejects_all_dead() {
        let frames: Vec<Option<RoundFrame>> = vec![None, None];
        let err = reduce_realized_work(&frames).unwrap_err();
        assert!(
            err.to_string().contains("no accepted frames"),
            "got: {err}"
        );
    }

    #[test]
    fn averager_zero_world_size_errors() {
        let err = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            0,
            TEST_SALT,
        )
        .unwrap_err();
        assert!(err.to_string().contains("world_size"), "got: {err}");
    }

    /// Cross-session safety: a relay forwarding a RoundFrame whose inner
    /// body is keyed with a salt the controller doesn't share must fail
    /// the inner HMAC check loudly. The mux envelope uses the correct
    /// salt (so the controller accepts + demuxes the record), but the
    /// wrapped RoundFrame is rogue-keyed — the controller's reduce reader
    /// surfaces a loud HMAC error.
    #[test]
    fn rejects_round_frame_with_wrong_inner_salt() {
        use crate::distributed::wire::SESSION_SALT_BYTES;
        let controller_salt = TEST_SALT;
        let rogue_salt: SessionSalt = [0xAAu8; SESSION_SALT_BYTES];
        assert_ne!(controller_salt, rogue_salt);

        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            1,
            controller_salt,
        )
        .unwrap();
        let port = avg.port();

        let mut stream =
            TcpStream::connect(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)).unwrap();
        // Valid channel magic + relay handshake (correct salt).
        crate::distributed::wire::write_channel_magic(
            &mut stream,
            crate::distributed::wire::CHANNEL_MAGIC_DATA,
        )
        .unwrap();
        MuxRecord::control(RelayControlMsg::Hello {
            host: "test".into(),
            ranks: vec![0],
            model_sigs: vec![],
        })
        .write_to(&mut stream, &controller_salt)
        .unwrap();
        match MuxRecord::read_from(&mut stream, &controller_salt).unwrap() {
            Some(MuxRecord::Control(RelayControlMsg::HelloAck)) => {}
            other => panic!("expected HelloAck, got {other:?}"),
        }
        // Forward a HostFrame whose mux envelope is correctly keyed but
        // whose inner RoundFrame is rogue-keyed.
        let mut buf = Vec::new();
        write_round_frame(&mut buf, &one_tensor_frame(&[1.0, 2.0, 3.0]), &rogue_salt).unwrap();
        MuxRecord::host_frame(buf)
            .write_to(&mut stream, &controller_salt)
            .unwrap();
        // The controller errors on the inner HMAC and tears down; our next
        // read sees the connection close.
        let _ = MuxRecord::read_from(&mut stream, &controller_salt);
        drop(stream);

        let err = avg.shutdown().expect_err(
            "controller's reduce loop must propagate an inner-RoundFrame HMAC error",
        );
        assert!(
            err.to_string().contains("HMAC verification failed"),
            "expected HMAC verification failure, got: {err}"
        );
    }

    // ---- elastic scatter ---------------------------------------------------

    /// A scatter write failure (wedged or vanished connection — the
    /// write-stall timeout surfaces the wedged case as an Err) must
    /// declare that CONNECTION's ranks dead and keep scattering to the
    /// survivors, not kill the reduce thread. One wedged host degrades
    /// membership; the realized-work reduce stays exact over the rest.
    #[test]
    fn elastic_scatter_declares_broken_connection_dead_and_continues() {
        use std::net::{Shutdown, TcpListener, TcpStream};
        let listener =
            TcpListener::bind(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let pair = || {
            let client = TcpStream::connect(addr).unwrap();
            let (server, _) = listener.accept().unwrap();
            (server, client)
        };
        // conn 0 carries rank 0 and is broken (locally shut down so the
        // very first write errors deterministically); conn 1 carries
        // ranks 1+2 and is live.
        let (ctrl0, _peer0) = pair();
        let (ctrl1, mut peer1) = pair();
        ctrl0.shutdown(Shutdown::Both).unwrap();
        let mut conn_writes = vec![ctrl0, ctrl1];
        let conn_ranks = vec![vec![0usize], vec![1usize, 2usize]];
        let dead = DeadRanks::new(3);

        // Folded host frames: conn 0 contributes value 0 at mass 1;
        // conn 1's fold contributes 1+2=3 at mass 2 → consensus 1.0.
        let frames: Vec<Option<RoundFrame>> = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![1],
                    bytes: f32_to_bytes(&[0.0]),
                }],
                weight: 1.0,
                ..Default::default()
            }),
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![1],
                    bytes: f32_to_bytes(&[3.0]),
                }],
                weight: 2.0,
                ..Default::default()
            }),
        ];

        average_and_scatter(
            &frames,
            &mut conn_writes,
            &conn_ranks,
            &dead,
            &TEST_SALT,
            None,
            None,
        )
        .expect("elastic scatter must not propagate a per-connection failure");

        assert!(dead.is_dead(0), "broken connection's rank must be declared dead");
        assert!(!dead.is_dead(1) && !dead.is_dead(2), "survivors stay alive");

        // The live connection received the round's single consensus
        // Broadcast (the relay fans it out locally).
        match MuxRecord::read_from(&mut peer1, &TEST_SALT).unwrap() {
            Some(MuxRecord::Broadcast { payload }) => {
                let frame = read_round_frame(&mut payload.as_slice(), &TEST_SALT)
                    .unwrap()
                    .expect("scattered frame");
                let vals = bytes_as_f32(&frame.tensors[0].bytes).unwrap();
                assert!((vals[0] - 1.0).abs() < 1e-6, "consensus, got {vals:?}");
                assert!((frame.weight - 3.0).abs() < 1e-9, "accepted mass rides down");
            }
            other => panic!("expected Broadcast record, got {other:?}"),
        }
    }

    // ---- fold monoid (sum_frames) -------------------------------------------

    /// The shared fold NEVER divides — masses and values are plain sums,
    /// whatever the kind. Dividing in a fold tier would reintroduce
    /// averaging-of-averages; only `reduce_realized_work` normalizes.
    #[test]
    fn sum_frames_is_a_pure_sum_with_summed_mass() {
        let a = RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_F32,
                shape: vec![2],
                bytes: f32_to_bytes(&[3.0, 6.0]),
            }],
            weight: 3.0,
            ..Default::default()
        };
        let b = RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_F32,
                shape: vec![2],
                bytes: f32_to_bytes(&[5.0, 10.0]),
            }],
            weight: 1.0,
            ..Default::default()
        };
        let folded = sum_frames(&[&a, &b]).unwrap();
        let vals = bytes_as_f32(&folded.tensors[0].bytes).unwrap();
        assert_eq!(vals, vec![8.0, 16.0], "values sum, never divide");
        assert!((folded.weight - 4.0).abs() < 1e-9, "masses sum");
        assert_eq!(folded.kind, RoundKind::Model, "kind preserved");
    }

    /// Mixed kinds in one fold = desynced rounds; must error loudly, not
    /// silently sum a Control gather into the model.
    #[test]
    fn sum_frames_rejects_kind_mismatch() {
        let model = one_tensor_frame(&[1.0]);
        let control = RoundFrame {
            kind: RoundKind::Control,
            ..one_tensor_frame(&[1.0])
        };
        let err = sum_frames(&[&model, &control]).unwrap_err();
        assert!(err.to_string().contains("kind"), "got: {err}");
    }

    // ---- bf16 wire codec ----------------------------------------------------

    /// One-tensor Model frame with bf16-encoded payload bytes.
    fn one_tensor_frame_bf16(data: &[f32], weight: f64) -> RoundFrame {
        RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_BF16,
                shape: vec![data.len() as u32],
                bytes: f32_slice_to_payload_bytes(data, DTYPE_BF16).unwrap(),
            }],
            kind: RoundKind::Model,
            weight,
        }
    }

    /// The bf16 primitives: round-to-nearest-even encoding, exact
    /// decoding, and special-value behavior.
    #[test]
    fn bf16_bits_round_to_nearest_even() {
        use super::round_frame::{bf16_bits_to_f32, f32_to_bf16_bits};
        // Exactly representable values survive the round trip bit-exact.
        for v in [0.0f32, 1.0, -2.0, 0.5, 256.0, -0.09375] {
            assert_eq!(bf16_bits_to_f32(f32_to_bf16_bits(v)), v, "{v}");
        }
        // 1.0 + 2⁻⁹ sits exactly halfway between 1.0 and the next bf16
        // (1.0 + 2⁻⁸); nearest-EVEN resolves down to 1.0 (even mantissa).
        let halfway = f32::from_bits(0x3F80_8000);
        assert_eq!(bf16_bits_to_f32(f32_to_bf16_bits(halfway)), 1.0);
        // Just above halfway rounds up.
        let above = f32::from_bits(0x3F80_8001);
        assert_eq!(
            bf16_bits_to_f32(f32_to_bf16_bits(above)),
            f32::from_bits(0x3F81_0000)
        );
        // Infinities pass through; NaN stays NaN (never rounds to inf).
        assert_eq!(bf16_bits_to_f32(f32_to_bf16_bits(f32::INFINITY)), f32::INFINITY);
        assert_eq!(
            bf16_bits_to_f32(f32_to_bf16_bits(f32::NEG_INFINITY)),
            f32::NEG_INFINITY
        );
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::NAN)).is_nan());
        // Decode is exact: bf16 → f32 → bf16 is the identity.
        for b in [0u16, 1, 0x3F80, 0x7F80, 0x8000, 0x4049] {
            assert_eq!(f32_to_bf16_bits(bf16_bits_to_f32(b)), b, "bits {b:#06x}");
        }
    }

    /// Payload encode/decode round trip in both dtypes, and the loud
    /// error on an unknown tag.
    #[test]
    fn payload_codec_round_trips() {
        let data = [1.5f32, -3.0, 0.0, 42.0];
        for dtype in [DTYPE_F32, DTYPE_BF16] {
            let p = TensorPayload {
                dtype,
                shape: vec![4],
                bytes: f32_slice_to_payload_bytes(&data, dtype).unwrap(),
            };
            // All values are bf16-representable, so both dtypes decode exact.
            assert_eq!(payload_to_f32(&p).unwrap(), data.to_vec(), "dtype {dtype}");
        }
        assert!(f32_slice_to_payload_bytes(&data, 9).is_err());
    }

    /// The streamed writer's length contract: `round_frame_wire_len`
    /// must equal the serialized frame byte-for-byte, in both dtypes —
    /// it is what the sender commits as the length prefix BEFORE any
    /// payload bytes exist, so a drift here desyncs every stream.
    #[test]
    fn round_frame_wire_len_matches_serialized_frame() {
        for dtype in [DTYPE_F32, DTYPE_BF16] {
            let frame = RoundFrame {
                tensors: vec![
                    TensorPayload {
                        dtype,
                        shape: vec![2, 3],
                        bytes: f32_slice_to_payload_bytes(&[1.5f32; 6], dtype).unwrap(),
                    },
                    TensorPayload {
                        dtype,
                        shape: vec![4],
                        bytes: f32_slice_to_payload_bytes(&[-2.0f32; 4], dtype).unwrap(),
                    },
                ],
                kind: RoundKind::Model,
                weight: 3.0,
            };
            let parts: Vec<PayloadPart<'_>> = frame
                .tensors
                .iter()
                .map(|t| PayloadPart {
                    dtype: t.dtype,
                    shape: &t.shape,
                    nbytes: t.bytes.len() as u64,
                })
                .collect();
            let mut buf = Vec::new();
            write_round_frame(&mut buf, &frame, &TEST_SALT).unwrap();
            assert_eq!(
                buf.len() as u64,
                round_frame_wire_len(&parts),
                "dtype {dtype}"
            );
            // And the (streamed-under-the-hood) writer still round-trips.
            let parsed = read_round_frame(&mut buf.as_slice(), &TEST_SALT)
                .unwrap()
                .unwrap();
            assert_eq!(parsed, frame, "dtype {dtype}");
        }
    }

    /// An elided frame (`nbytes = 0` per payload, schema intact) must
    /// FOLD identically to shipping actual `zeros_like` payloads — same
    /// consensus, byte for byte — in every mix: elided-with-full,
    /// all-elided, and for both wire dtypes. Pins the wire zero-elision
    /// semantics `broadcast_from_root`'s non-root path (and the idle
    /// rank's zero-mass contribution) rely on.
    #[test]
    fn an_elided_frame_folds_identically_to_a_zeros_model() {
        for dtype in [DTYPE_F32, DTYPE_BF16] {
            let values = RoundFrame {
                tensors: vec![
                    TensorPayload {
                        dtype,
                        shape: vec![2, 3],
                        bytes: f32_slice_to_payload_bytes(&[1.5f32; 6], dtype).unwrap(),
                    },
                    TensorPayload {
                        dtype,
                        shape: vec![4],
                        bytes: f32_slice_to_payload_bytes(&[-2.0f32; 4], dtype).unwrap(),
                    },
                ],
                kind: RoundKind::Control,
                weight: 0.0,
            };
            let zeros = RoundFrame {
                tensors: vec![
                    TensorPayload {
                        dtype,
                        shape: vec![2, 3],
                        bytes: f32_slice_to_payload_bytes(&[0.0f32; 6], dtype).unwrap(),
                    },
                    TensorPayload {
                        dtype,
                        shape: vec![4],
                        bytes: f32_slice_to_payload_bytes(&[0.0f32; 4], dtype).unwrap(),
                    },
                ],
                kind: RoundKind::Control,
                weight: 0.0,
            };
            let elided = RoundFrame {
                tensors: vec![
                    TensorPayload {
                        dtype,
                        shape: vec![2, 3],
                        bytes: Vec::new(),
                    },
                    TensorPayload {
                        dtype,
                        shape: vec![4],
                        bytes: Vec::new(),
                    },
                ],
                kind: RoundKind::Control,
                weight: 0.0,
            };
            assert!(elided.tensors.iter().all(|t| t.is_elided()));

            let via_zeros = sum_frames(&[&values, &zeros]).unwrap();
            let via_elided = sum_frames(&[&values, &elided]).unwrap();
            assert_eq!(
                via_zeros, via_elided,
                "dtype {dtype}: elided must fold exactly like zeros"
            );
            // Either operand order — the elided frame may be frame 0
            // (it pins the schema the others validate against).
            let elided_first = sum_frames(&[&elided, &values]).unwrap();
            assert_eq!(
                via_zeros.tensors, elided_first.tensors,
                "dtype {dtype}: elided-as-reference must fold the same"
            );
        }
    }

    /// Zeros + zeros = zeros: a round whose every frame elided a tensor
    /// keeps that tensor elided in the sum — the elision composes
    /// through fold tiers (relay → controller → scatter) instead of
    /// rematerializing a model of zero bytes at the first fold.
    #[test]
    fn an_all_elided_round_sums_to_an_elided_frame() {
        let elided = |weight: f64| RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_F32,
                shape: vec![3, 2],
                bytes: Vec::new(),
            }],
            kind: RoundKind::Model,
            weight,
        };
        let a = elided(0.0);
        let b = elided(0.0);
        let sum = sum_frames(&[&a, &b]).unwrap();
        assert!(
            sum.tensors[0].is_elided(),
            "all-elided inputs must produce an elided sum"
        );
        assert_eq!(sum.tensors[0].shape, vec![3, 2], "schema must survive");
        assert_eq!(sum.weight, 0.0);
        // And the elided sum still DECODES as zeros of its shape.
        assert_eq!(payload_to_f32(&sum.tensors[0]).unwrap(), vec![0.0f32; 6]);
    }

    /// An elided frame round-trips the wire: `nbytes = 0` per payload,
    /// schema + MAC intact, and the whole frame is schema-sized — the
    /// wire cost of a 190M-param zeros contribution drops from the
    /// model's bytes to its shape list.
    #[test]
    fn an_elided_frame_round_trips_the_wire_schema_sized() {
        let frame = RoundFrame {
            tensors: vec![
                TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![64, 128],
                    bytes: Vec::new(),
                },
                TensorPayload {
                    dtype: DTYPE_BF16,
                    shape: vec![512],
                    bytes: Vec::new(),
                },
            ],
            kind: RoundKind::Model,
            weight: 0.0,
        };
        let mut buf = Vec::new();
        write_round_frame(&mut buf, &frame, &TEST_SALT).unwrap();
        // hdr(8) + kind(1) + weight(8) + t0(2 + 2*4 + 8) + t1(2 + 1*4 + 8)
        // + footer(8) = 57 — no payload bytes at all.
        assert_eq!(buf.len(), 57, "elided frame must be schema-sized");
        let parsed = read_round_frame(&mut buf.as_slice(), &TEST_SALT)
            .unwrap()
            .unwrap();
        assert_eq!(parsed, frame, "elision must survive the round-trip");
        assert!(parsed.tensors.iter().all(|t| t.is_elided()));
    }

    /// An emitter that writes a different byte count than its declared
    /// `nbytes` must error loudly — the length prefix is already on the
    /// wire by then, so silence would mean a desynced stream instead of
    /// a named bug.
    #[test]
    fn streamed_writer_rejects_emitter_byte_count_mismatch() {
        let shape = [2u32];
        let parts = [PayloadPart {
            dtype: DTYPE_F32,
            shape: &shape,
            nbytes: 8,
        }];
        let mut buf = Vec::new();
        let err = write_round_frame_streamed(
            &mut buf,
            RoundKind::Model,
            1.0,
            &parts,
            &TEST_SALT,
            &mut |_, tee| {
                use std::io::Write;
                tee.write_all(&[0u8; 4])
                    .map_err(|e| TensorError::new(&e.to_string()))
            },
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("declared") && msg.contains("4"),
            "expected loud emitter byte-count mismatch, got: {msg}"
        );
    }

    /// The fold must accumulate in f32 whatever the wire dtype. 512
    /// frames of 1.0: a (wrong) bf16 running sum would stall at 256
    /// (256 + 1 rounds back to 256 in bf16); the f32 accumulator reaches
    /// 512, which IS bf16-representable, so the output encodes it exactly.
    #[test]
    fn sum_frames_bf16_accumulates_in_f32() {
        let frames: Vec<RoundFrame> =
            (0..512).map(|_| one_tensor_frame_bf16(&[1.0], 1.0)).collect();
        let refs: Vec<&RoundFrame> = frames.iter().collect();
        let folded = sum_frames(&refs).unwrap();
        assert_eq!(folded.tensors[0].dtype, DTYPE_BF16, "fold preserves the wire dtype");
        assert_eq!(payload_to_f32(&folded.tensors[0]).unwrap(), vec![512.0]);
        assert!((folded.weight - 512.0).abs() < 1e-9);
    }

    /// A cohort mixing bf16 and f32 frames is a config error (some ranks
    /// enabled `bf16_wire`, some did not) — loud schema error, never a
    /// silent byte-level mangle.
    #[test]
    fn sum_frames_rejects_dtype_mix() {
        let bf16 = one_tensor_frame_bf16(&[1.0], 1.0);
        let f32f = one_tensor_frame(&[1.0]);
        let err = sum_frames(&[&bf16, &f32f]).unwrap_err();
        assert!(err.to_string().contains("dtype"), "got: {err}");
    }

    /// Divide-once on a bf16 round: contributions pre-scaled by mass,
    /// sum divided exactly once, output still bf16.
    #[test]
    fn reduce_realized_work_bf16_normalizes() {
        // rank A: params [2, 4] × weight 3 → [6, 12]; rank B: [4, 8] × 1.
        let frames = vec![
            Some(one_tensor_frame_bf16(&[6.0, 12.0], 3.0)),
            Some(one_tensor_frame_bf16(&[4.0, 8.0], 1.0)),
        ];
        let reduced = reduce_realized_work(&frames).unwrap();
        assert_eq!(reduced.tensors[0].dtype, DTYPE_BF16);
        // (6+4)/4 = 2.5, (12+8)/4 = 5.0 — both bf16-exact.
        assert_eq!(
            payload_to_f32(&reduced.tensors[0]).unwrap(),
            vec![2.5, 5.0]
        );
        assert!((reduced.weight - 4.0).abs() < 1e-9);
    }

    /// bf16 and f32 pipelines agree within bf16 wire precision on
    /// realistic (non-representable) values: same contributions, same
    /// masses, relative error bounded by a few wire quantization steps.
    #[test]
    fn bf16_reduce_tracks_f32_reduce_within_wire_precision() {
        let a = [0.123_456_7f32, -1.234_567_8, 3.317_742_9];
        let b = [2.941_385_2f32, -0.577_215_7, 1.489_306_4];
        let f32_frames = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![3],
                    bytes: f32_slice_to_payload_bytes(&a, DTYPE_F32).unwrap(),
                }],
                kind: RoundKind::Model,
                weight: 2.0,
            }),
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![3],
                    bytes: f32_slice_to_payload_bytes(&b, DTYPE_F32).unwrap(),
                }],
                kind: RoundKind::Model,
                weight: 1.0,
            }),
        ];
        let bf16_frames = vec![
            Some(one_tensor_frame_bf16(&a, 2.0)),
            Some(one_tensor_frame_bf16(&b, 1.0)),
        ];
        let exact = payload_to_f32(&reduce_realized_work(&f32_frames).unwrap().tensors[0]).unwrap();
        let quant = payload_to_f32(&reduce_realized_work(&bf16_frames).unwrap().tensors[0]).unwrap();
        for (e, q) in exact.iter().zip(&quant) {
            // Two quantization stages (contribution encode + result
            // encode), each ≤ 2⁻⁸ relative.
            assert!(
                (e - q).abs() <= e.abs() * (2.0 / 256.0) + 1e-6,
                "exact {e} vs bf16 {q}"
            );
        }
    }

    /// Associativity: folding per host first then reducing equals the
    /// flat reduce over all rank frames (exact here — values chosen so
    /// f32 addition order cannot bite).
    #[test]
    fn host_fold_then_reduce_matches_flat_reduce() {
        let r0 = one_tensor_frame(&[1.0, 2.0]);
        let r1 = one_tensor_frame(&[3.0, 4.0]);
        let r2 = one_tensor_frame(&[5.0, 6.0]);
        let flat = reduce_realized_work(&[
            Some(r0.clone()),
            Some(r1.clone()),
            Some(r2.clone()),
        ])
        .unwrap();
        let host_a = sum_frames(&[&r0, &r1]).unwrap();
        let host_b = sum_frames(&[&r2]).unwrap();
        let folded = reduce_realized_work(&[Some(host_a), Some(host_b)]).unwrap();
        assert_eq!(
            bytes_as_f32(&flat.tensors[0].bytes).unwrap(),
            bytes_as_f32(&folded.tensors[0].bytes).unwrap(),
        );
        assert!((flat.weight - folded.weight).abs() < 1e-9);
    }

    /// A per-rank `Data` record on the data channel means a stale
    /// (pre-fold) relay build; the controller must error loudly instead
    /// of silently mis-accounting the round.
    #[test]
    fn rejects_per_rank_data_record_on_data_channel() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            1,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        let mut stream =
            TcpStream::connect(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)).unwrap();
        crate::distributed::wire::write_channel_magic(
            &mut stream,
            crate::distributed::wire::CHANNEL_MAGIC_DATA,
        )
        .unwrap();
        MuxRecord::control(RelayControlMsg::Hello {
            host: "stale".into(),
            ranks: vec![0],
            model_sigs: vec![],
        })
        .write_to(&mut stream, &TEST_SALT)
        .unwrap();
        match MuxRecord::read_from(&mut stream, &TEST_SALT).unwrap() {
            Some(MuxRecord::Control(RelayControlMsg::HelloAck)) => {}
            other => panic!("expected HelloAck, got {other:?}"),
        }
        let mut buf = Vec::new();
        write_round_frame(&mut buf, &one_tensor_frame(&[1.0]), &TEST_SALT).unwrap();
        MuxRecord::data(0, buf).write_to(&mut stream, &TEST_SALT).unwrap();
        // The controller errors on the stale record and tears down; wait
        // for the connection close so shutdown() can't win the race and
        // mask the error with a clean external-shutdown outcome.
        let _ = MuxRecord::read_from(&mut stream, &TEST_SALT);
        drop(stream);

        let err = avg.shutdown().expect_err(
            "a per-rank Data record on the data channel must surface as an error",
        );
        assert!(
            err.to_string().contains("HostFrame"),
            "expected the mixed-builds diagnostic, got: {err}"
        );
    }

    /// Two hosts, each folding locally, one reduce round: the controller
    /// accounts per connection and the consensus matches the flat
    /// average over all three ranks. Exercises the real per-host round
    /// barrier (both relays must deposit before either gets the
    /// Broadcast).
    #[test]
    fn two_host_fold_average_one_round() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            3,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        // Host A carries ranks 0+1, host B carries rank 2. Equal-mass
        // frames → consensus = plain mean (2, 3).
        let host_a = std::thread::spawn(move || {
            fake_relay(
                port,
                vec![0, 1],
                TEST_SALT,
                vec![
                    vec![one_tensor_frame(&[1.0, 2.0])],
                    vec![one_tensor_frame(&[2.0, 3.0])],
                ],
            )
        });
        let host_b = std::thread::spawn(move || {
            fake_relay(
                port,
                vec![2],
                TEST_SALT,
                vec![vec![one_tensor_frame(&[3.0, 4.0])]],
            )
        });
        let recv_a = host_a.join().unwrap().unwrap();
        let recv_b = host_b.join().unwrap().unwrap();
        avg.shutdown().unwrap();

        let consensus = bytes_as_f32(&recv_a[0][0].tensors[0].bytes).unwrap();
        assert_eq!(consensus, vec![2.0, 3.0]);
        // Every rank on every host sees the identical consensus, with
        // the full accepted mass riding down.
        assert_eq!(recv_a[0][0], recv_a[1][0]);
        assert_eq!(recv_a[0][0], recv_b[0][0]);
        assert!((recv_b[0][0].weight - 3.0).abs() < 1e-9);
    }
