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

    /// Fake per-host relay: the controller now speaks the relay/mux
    /// protocol (one connection per host carrying many ranks), so the test
    /// peer is a relay, not a rank. Connects, sends `RelayHello` for
    /// `ranks`, then per round forwards each rank's frame up (tagged
    /// `MuxRecord::Data`) and collects each rank's averaged reply (demuxed
    /// by tag). Returns, per rank (parallel to `ranks`), the averaged
    /// frames received.
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

        // Relay handshake.
        MuxRecord::control(RelayControlMsg::Hello {
            host: "test-host".into(),
            ranks: ranks.clone(),
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

        // Transpose rank-major frames into round-major so we drive the
        // reduce loop one synchronized round at a time.
        let rounds: Vec<Vec<(u32, &RoundFrame)>> = (0..n_rounds)
            .map(|r| {
                ranks
                    .iter()
                    .zip(&per_rank_frames)
                    .map(|(rank, frames)| (*rank, &frames[r]))
                    .collect()
            })
            .collect();

        let mut received: Vec<Vec<RoundFrame>> = ranks.iter().map(|_| Vec::new()).collect();
        for round in &rounds {
            // Forward each rank's frame up, tagged.
            for (rank, frame) in round {
                let mut buf = Vec::new();
                write_round_frame(&mut buf, frame, &salt)?;
                MuxRecord::data(*rank, buf).write_to(&mut stream, &salt)?;
            }
            // Collect one averaged reply per rank (tagged, any order).
            for _ in 0..ranks.len() {
                match MuxRecord::read_from(&mut stream, &salt)? {
                    Some(MuxRecord::Data { rank, payload }) => {
                        let frame = read_round_frame(&mut payload.as_slice(), &salt)?
                            .ok_or_else(|| {
                                TensorError::new("fake_relay: truncated averaged frame")
                            })?;
                        let idx = ranks.iter().position(|r| *r == rank).ok_or_else(|| {
                            TensorError::new(&format!("fake_relay: reply for unknown rank {rank}"))
                        })?;
                        received[idx].push(frame);
                    }
                    other => {
                        return Err(TensorError::new(&format!(
                            "fake_relay: expected Data reply, got {other:?}"
                        )));
                    }
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
        MuxRecord::control(RelayControlMsg::Hello {
            host: "rogue".into(),
            ranks: vec![5], // >= world_size (1)
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
            err.to_string().contains("dtype 7"),
            "expected dtype-7-not-supported, got: {err}"
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
        // Valid relay handshake (correct salt).
        MuxRecord::control(RelayControlMsg::Hello {
            host: "test".into(),
            ranks: vec![0],
        })
        .write_to(&mut stream, &controller_salt)
        .unwrap();
        match MuxRecord::read_from(&mut stream, &controller_salt).unwrap() {
            Some(MuxRecord::Control(RelayControlMsg::HelloAck)) => {}
            other => panic!("expected HelloAck, got {other:?}"),
        }
        // Forward a Data record whose mux envelope is correctly keyed but
        // whose inner RoundFrame is rogue-keyed.
        let mut buf = Vec::new();
        write_round_frame(&mut buf, &one_tensor_frame(&[1.0, 2.0, 3.0]), &rogue_salt).unwrap();
        MuxRecord::data(0, buf)
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
        let rank_conn = vec![Some(0usize), Some(1usize), Some(1usize)];
        let dead = DeadRanks::new(3);

        // Realized-work frames: values [0,1,2] at mass 1 each → consensus 1.0.
        let frames: Vec<Option<RoundFrame>> = (0..3)
            .map(|r| {
                Some(RoundFrame {
                    tensors: vec![TensorPayload {
                        dtype: DTYPE_F32,
                        shape: vec![1],
                        bytes: f32_to_bytes(&[r as f32]),
                    }],
                    weight: 1.0,
                    ..Default::default()
                })
            })
            .collect();

        average_and_scatter(
            &frames,
            &mut conn_writes,
            &rank_conn,
            &dead,
            &TEST_SALT,
            None,
            None,
        )
        .expect("elastic scatter must not propagate a per-connection failure");

        assert!(dead.is_dead(0), "broken connection's rank must be declared dead");
        assert!(!dead.is_dead(1) && !dead.is_dead(2), "survivors stay alive");

        // Both survivors received the consensus on the live connection.
        for _ in 0..2 {
            match MuxRecord::read_from(&mut peer1, &TEST_SALT).unwrap() {
                Some(MuxRecord::Data { rank, payload }) => {
                    assert!(rank == 1 || rank == 2, "unexpected rank {rank}");
                    let frame = read_round_frame(&mut payload.as_slice(), &TEST_SALT)
                        .unwrap()
                        .expect("scattered frame");
                    let vals = bytes_as_f32(&frame.tensors[0].bytes).unwrap();
                    assert!((vals[0] - 1.0).abs() < 1e-6, "consensus, got {vals:?}");
                    assert!((frame.weight - 3.0).abs() < 1e-9, "accepted mass rides down");
                }
                other => panic!("expected Data record, got {other:?}"),
            }
        }
    }
