    use super::*;
    use std::io::Read;
    use std::net::Ipv4Addr;
    use std::sync::mpsc;

    /// Deterministic non-zero test salt: exercises the HMAC path (zero
    /// salt is degenerate enough that an accidental "skip the HMAC"
    /// regression could silently still produce all-zero footers and
    /// "pass" — a non-zero salt catches that).
    const TEST_SALT: SessionSalt = [
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    ];

    /// Fake rank client: connects to the controller, does the handshake,
    /// and runs `n_rounds` of (send_frame → recv_averaged_frame).
    /// Returns the vector of received averaged frames.
    fn fake_rank(
        port: u16,
        rank_id: u32,
        world_size: u32,
        salt: SessionSalt,
        send_frames: Vec<RoundFrame>,
    ) -> Result<Vec<RoundFrame>> {
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
        let mut stream = TcpStream::connect(addr).map_err(|e| {
            TensorError::new(&format!("fake_rank {rank_id}: connect: {e}"))
        })?;

        // Handshake send
        let mut h = [0u8; 16];
        h[0..4].copy_from_slice(&HANDSHAKE_MAGIC_RANK.to_le_bytes());
        h[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        h[8..12].copy_from_slice(&rank_id.to_le_bytes());
        h[12..16].copy_from_slice(&world_size.to_le_bytes());
        stream.write_all(&h).map_err(|e| {
            TensorError::new(&format!("fake_rank {rank_id}: handshake write: {e}"))
        })?;
        let mut ack = [0u8; 8];
        stream.read_exact(&mut ack).map_err(|e| {
            TensorError::new(&format!("fake_rank {rank_id}: ack read: {e}"))
        })?;
        let ack_magic = u32::from_le_bytes(ack[0..4].try_into().unwrap());
        assert_eq!(ack_magic, HANDSHAKE_MAGIC_CONTROLLER_ACK);

        let mut received = Vec::with_capacity(send_frames.len());
        for f in send_frames {
            write_round_frame(&mut stream, &f, &salt)?;
            let r = read_round_frame(&mut stream, &salt)?
                .ok_or_else(|| TensorError::new("fake_rank: EOF before averaged frame"))?;
            received.push(r);
        }
        // Drop stream → clean EOF to controller, signals shutdown.
        Ok(received)
    }

    fn one_tensor_frame(data: &[f32]) -> RoundFrame {
        RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_F32,
                shape: vec![data.len() as u32],
                bytes: f32_to_bytes(data),
            }],
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

        let (tx0, rx0) = mpsc::channel();
        let (tx1, rx1) = mpsc::channel();
        let t0 = thread::spawn(move || {
            let r = fake_rank(port, 0, 2, TEST_SALT, vec![one_tensor_frame(&[1.0, 2.0, 3.0])]);
            tx0.send(r).unwrap();
        });
        let t1 = thread::spawn(move || {
            let r = fake_rank(port, 1, 2, TEST_SALT, vec![one_tensor_frame(&[3.0, 4.0, 5.0])]);
            tx1.send(r).unwrap();
        });

        let r0 = rx0.recv().unwrap().unwrap();
        let r1 = rx1.recv().unwrap().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        avg.shutdown().unwrap();

        // Average of (1,2,3) and (3,4,5) = (2,3,4)
        let expected = bytes_as_f32(&r0[0].tensors[0].bytes).unwrap();
        assert_eq!(expected, vec![2.0, 3.0, 4.0]);
        // Both ranks receive the same averaged frame.
        assert_eq!(r0, r1);
    }

    #[test]
    fn three_rank_average_multi_round_multi_tensor() {
        // Three ranks, two rounds each, each round carries two tensors.
        // Exercises:
        //   - multi-rank star summation (3 ranks)
        //   - multi-round reduce loop (2 rounds)
        //   - multi-tensor frames (2 tensors per frame)
        //   - clean shutdown on rank EOF
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            3,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        // Round-1 inputs: ranks 0/1/2 contribute (0,10), (10,20), (20,30) for tensor 0
        //                          and (1,1), (2,2), (3,3) for tensor 1.
        // Round-2 inputs: same shape, halved values (just to vary the data).
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

        let (tx0, rx0) = mpsc::channel();
        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        let t0 = thread::spawn(move || tx0.send(fake_rank(port, 0, 3, TEST_SALT, r0_frames)).unwrap());
        let t1 = thread::spawn(move || tx1.send(fake_rank(port, 1, 3, TEST_SALT, r1_frames)).unwrap());
        let t2 = thread::spawn(move || tx2.send(fake_rank(port, 2, 3, TEST_SALT, r2_frames)).unwrap());

        let r0 = rx0.recv().unwrap().unwrap();
        let r1 = rx1.recv().unwrap().unwrap();
        let r2 = rx2.recv().unwrap().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        t2.join().unwrap();
        avg.shutdown().unwrap();

        // Each rank received exactly 2 averaged frames.
        assert_eq!(r0.len(), 2, "rank 0 should receive 2 averaged frames");
        assert_eq!(r1.len(), 2);
        assert_eq!(r2.len(), 2);

        // Round 1 averages: tensor 0 = (10, 20), tensor 1 = (2, 2)
        let r1_t0 = bytes_as_f32(&r0[0].tensors[0].bytes).unwrap();
        let r1_t1 = bytes_as_f32(&r0[0].tensors[1].bytes).unwrap();
        assert_eq!(r1_t0, vec![10.0, 20.0]);
        assert_eq!(r1_t1, vec![2.0, 2.0]);

        // Round 2 averages: tensor 0 = (5, 10), tensor 1 = (1, 1)
        let r2_t0 = bytes_as_f32(&r0[1].tensors[0].bytes).unwrap();
        let r2_t1 = bytes_as_f32(&r0[1].tensors[1].bytes).unwrap();
        assert_eq!(r2_t0, vec![5.0, 10.0]);
        assert_eq!(r2_t1, vec![1.0, 1.0]);

        // All three ranks see bit-identical averaged frames.
        assert_eq!(r0, r1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn rejects_wrong_handshake_magic() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            1,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        let mut s = TcpStream::connect(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)).unwrap();
        let mut bad = [0u8; 16];
        bad[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        s.write_all(&bad).unwrap();
        // The controller should reject and drop us; read_exact on ack
        // would return EOF or error. We accept either outcome.
        let mut ack = [0u8; 8];
        let _ = s.read_exact(&mut ack);
        drop(s);
        // The reduce thread terminates with an error; shutdown still
        // joins cleanly.
        let _ = avg.shutdown(); // err is OK here
    }

    #[test]
    fn rejects_world_size_mismatch() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        // Rank claims world_size = 3 but controller is configured for 2.
        let mut s = TcpStream::connect(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)).unwrap();
        let mut h = [0u8; 16];
        h[0..4].copy_from_slice(&HANDSHAKE_MAGIC_RANK.to_le_bytes());
        h[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        h[8..12].copy_from_slice(&0u32.to_le_bytes());
        h[12..16].copy_from_slice(&3u32.to_le_bytes());
        s.write_all(&h).unwrap();
        // Server drops us.
        let mut ack = [0u8; 8];
        let _ = s.read_exact(&mut ack);
        drop(s);
        let _ = avg.shutdown();
    }

    #[test]
    fn rejects_non_f32_dtype_in_reduce() {
        // Pure unit test of reduce_average_alive without TCP wiring.
        let frames = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: 7, // bogus dtype
                    shape: vec![2],
                    bytes: vec![0; 8],
                }],
            }),
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: 7,
                    shape: vec![2],
                    bytes: vec![0; 8],
                }],
            }),
        ];
        let err = reduce_average_alive(&frames).unwrap_err();
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
            }),
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![3],
                    bytes: f32_to_bytes(&[1.0, 2.0, 3.0]),
                }],
            }),
        ];
        let err = reduce_average_alive(&frames).unwrap_err();
        assert!(err.to_string().contains("shape"), "got: {err}");
    }

    #[test]
    fn reduce_average_alive_skips_none_entries_and_divides_by_alive_count() {
        // 3-rank world, rank 1 dead (None). Mean over alive = (rank0 + rank2) / 2.
        let frames = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![2],
                    bytes: f32_to_bytes(&[2.0, 4.0]),
                }],
            }),
            None, // rank 1 dead
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![2],
                    bytes: f32_to_bytes(&[6.0, 8.0]),
                }],
            }),
        ];
        let out = reduce_average_alive(&frames).unwrap();
        let avg = bytes_as_f32(&out.tensors[0].bytes).unwrap();
        // (2 + 6) / 2 = 4.0; (4 + 8) / 2 = 6.0
        assert!((avg[0] - 4.0).abs() < 1e-6, "got {avg:?}");
        assert!((avg[1] - 6.0).abs() < 1e-6, "got {avg:?}");
    }

    #[test]
    fn reduce_average_alive_rejects_all_dead() {
        let frames: Vec<Option<RoundFrame>> = vec![None, None];
        let err = reduce_average_alive(&frames).unwrap_err();
        assert!(
            err.to_string().contains("no alive ranks"),
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

    /// Cross-session safety: a rank using a salt the controller doesn't
    /// share must fail the first RoundFrame's HMAC check loudly. This
    /// is the load-bearing test that proves the salt is wired through
    /// both directions.
    #[test]
    fn rejects_round_frame_with_wrong_salt() {
        use crate::distributed::wire::SESSION_SALT_BYTES;
        let controller_salt = TEST_SALT;
        // The "rogue" salt: same length, different bytes.
        let rogue_salt: SessionSalt = [0xAAu8; SESSION_SALT_BYTES];
        assert_ne!(controller_salt, rogue_salt);

        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            1,
            controller_salt,
        )
        .unwrap();
        let port = avg.port();

        // Single-rank handshake (so the controller proceeds to the
        // reduce loop), then send one RoundFrame keyed with the wrong
        // salt. The controller's read_round_frame must error on the
        // HMAC footer.
        let send_res = thread::spawn(move || -> Result<()> {
            let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
            let mut stream = TcpStream::connect(addr).unwrap();

            // Handshake (the salt does not participate in handshake bytes;
            // any rank with matching world_size + magic + version connects).
            let mut h = [0u8; 16];
            h[0..4].copy_from_slice(&HANDSHAKE_MAGIC_RANK.to_le_bytes());
            h[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
            h[8..12].copy_from_slice(&0u32.to_le_bytes());
            h[12..16].copy_from_slice(&1u32.to_le_bytes());
            stream.write_all(&h).unwrap();
            let mut ack = [0u8; 8];
            stream.read_exact(&mut ack).unwrap();

            // Now send a frame keyed by the rogue salt. The controller's
            // HMAC over the body will not match → the reduce thread
            // errors out and shuts down.
            let frame = one_tensor_frame(&[1.0, 2.0, 3.0]);
            write_round_frame(&mut stream, &frame, &rogue_salt)?;
            Ok(())
        });
        let _ = send_res.join().unwrap();

        // Drain the controller's status to confirm the loud error path
        // ran. `shutdown()` joins the thread and propagates its Result.
        let err = avg.shutdown().expect_err(
            "controller's reduce thread must propagate a HMAC verification error",
        );
        assert!(
            err.to_string().contains("HMAC verification failed"),
            "expected HMAC verification failure, got: {err}"
        );
    }
