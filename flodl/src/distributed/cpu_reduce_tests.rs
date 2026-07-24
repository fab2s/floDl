    use super::*;
    use crate::distributed::controller::{
        ClusterController, DTYPE_F32, RoundFrame, TensorPayload,
    };
    use crate::distributed::relay::agent::{ChannelKind, RelayChannel};
    use std::net::Ipv4Addr;
    use std::sync::mpsc;
    use std::thread;

    /// Deterministic non-zero test salt; matches the controller-side
    /// constant in `controller::tests`.
    const TEST_SALT: SessionSalt = [
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    ];

    fn local(port: u16) -> SocketAddr {
        SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)
    }

    fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len() * 4);
        for x in data {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn bytes_as_f32(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for i in 0..bytes.len() / 4 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[i * 4..(i + 1) * 4]);
            out.push(f32::from_le_bytes(b));
        }
        out
    }

    fn frame_with(data: &[f32]) -> RoundFrame {
        RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_F32,
                shape: vec![data.len() as u32],
                bytes: f32_to_bytes(data),
            }],
            // Equal-mass contributions: the realized-work reduce then
            // returns the plain mean over the accepted cohort, which is
            // what these wire-level tests assert.
            weight: 1.0,
            ..Default::default()
        }
    }

    /// Stand up a `ClusterController` fronted by a real per-host
    /// `RelayChannel` carrying `ranks`, run `client_body` (which connects
    /// `CpuReduceClient`s to the relay's loopback address), then tear both
    /// down. This is the true 3-tier path: client -> relay -> controller.
    ///
    /// `RelayChannel::start` blocks accepting ranks, so it runs on a
    /// background thread; once `client_body`'s clients connect, the
    /// handshake phase completes and the started handle is collected for
    /// shutdown after the clients finish.
    fn with_relayed_controller<F, T>(
        world_size: u32,
        ranks: Vec<u32>,
        salt: SessionSalt,
        client_body: F,
    ) -> T
    where
        F: FnOnce(SocketAddr) -> T,
    {
        let controller =
            ClusterController::start(local(0), world_size as usize, salt).unwrap();
        let controller_addr = local(controller.port());
        let (listener, relay_port) = RelayChannel::bind(local(0)).unwrap();

        let (htx, hrx) = mpsc::channel();
        thread::spawn(move || {
            let started = RelayChannel::start(
                listener,
                ChannelKind::Data,
                controller_addr,
                "test-host".into(),
                ranks,
                world_size as usize,
                salt,
            );
            let _ = htx.send(started);
        });

        let result = client_body(local(relay_port));

        // Clients have connected (and finished), so the relay's handshake
        // phase returned; collect the handle and shut it down.
        match hrx.recv().unwrap() {
            Ok(ch) => {
                let _ = ch.shutdown();
            }
            Err(e) => panic!("relay start failed: {e}"),
        }
        controller.shutdown().unwrap();
        result
    }

    /// End-to-end through a relay: two rank clients average; the mean
    /// comes back to each.
    #[test]
    fn two_rank_client_average() {
        let (r0, r1) = with_relayed_controller(2, vec![0, 1], TEST_SALT, |addr| {
            let (tx0, rx0) = mpsc::channel();
            let (tx1, rx1) = mpsc::channel();
            let t0 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 0, 2, TEST_SALT).unwrap();
                tx0.send(c.all_reduce(frame_with(&[2.0, 4.0, 6.0])).unwrap())
                    .unwrap();
                drop(c);
            });
            let t1 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 1, 2, TEST_SALT).unwrap();
                tx1.send(c.all_reduce(frame_with(&[4.0, 8.0, 12.0])).unwrap())
                    .unwrap();
                drop(c);
            });
            let r0 = rx0.recv().unwrap();
            let r1 = rx1.recv().unwrap();
            t0.join().unwrap();
            t1.join().unwrap();
            (r0, r1)
        });

        let avg0 = bytes_as_f32(&r0.tensors[0].bytes);
        assert_eq!(avg0, vec![3.0, 6.0, 9.0]);
        assert_eq!(r0, r1);
    }

    /// Each client survives multiple rounds through the relay.
    #[test]
    fn client_multi_round_persistence() {
        let (r0_results, r1_results) =
            with_relayed_controller(2, vec![0, 1], TEST_SALT, |addr| {
                let t0 = thread::spawn(move || -> Vec<RoundFrame> {
                    let mut c = CpuReduceClient::connect(addr, 0, 2, TEST_SALT).unwrap();
                    let r1 = c.all_reduce(frame_with(&[1.0])).unwrap();
                    let r2 = c.all_reduce(frame_with(&[5.0])).unwrap();
                    let r3 = c.all_reduce(frame_with(&[9.0])).unwrap();
                    vec![r1, r2, r3]
                });
                let t1 = thread::spawn(move || -> Vec<RoundFrame> {
                    let mut c = CpuReduceClient::connect(addr, 1, 2, TEST_SALT).unwrap();
                    let r1 = c.all_reduce(frame_with(&[3.0])).unwrap();
                    let r2 = c.all_reduce(frame_with(&[7.0])).unwrap();
                    let r3 = c.all_reduce(frame_with(&[11.0])).unwrap();
                    vec![r1, r2, r3]
                });
                let r0 = t0.join().unwrap();
                let r1 = t1.join().unwrap();
                (r0, r1)
            });

        // Round-by-round averages: (1,3)/2=2, (5,7)/2=6, (9,11)/2=10.
        let expected = [2.0_f32, 6.0, 10.0];
        for (i, want) in expected.iter().enumerate() {
            let got_0 = bytes_as_f32(&r0_results[i].tensors[0].bytes);
            let got_1 = bytes_as_f32(&r1_results[i].tensors[0].bytes);
            assert_eq!(got_0, vec![*want], "rank 0 round {i}");
            assert_eq!(got_1, vec![*want], "rank 1 round {i}");
        }
    }

    /// World-size disagreement is now validated by the relay's handshake
    /// termination (it reuses the controller's handshake reader): a client
    /// claiming a different world_size than the relay was started with
    /// gets a wire-level error on its handshake ack.
    #[test]
    fn rejects_world_size_disagreement() {
        let controller = ClusterController::start(local(0), 2, TEST_SALT).unwrap();
        let controller_addr = local(controller.port());
        let (listener, relay_port) = RelayChannel::bind(local(0)).unwrap();
        // Relay carries rank 0 with world_size 2; its handshake reader
        // rejects a mismatching client. start() errors (ignored).
        thread::spawn(move || {
            let _ = RelayChannel::start(
                listener,
                ChannelKind::Data,
                controller_addr,
                "h".into(),
                vec![0],
                2,
                TEST_SALT,
            );
        });

        // Client claims world_size = 3.
        let err = CpuReduceClient::connect(local(relay_port), 0, 3, TEST_SALT).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("ack") || msg.contains("handshake") || msg.contains("read"),
            "expected wire-level error, got: {msg}"
        );
        let _ = controller.shutdown();
    }

    /// Constructing a client with rank_id >= world_size is a local-only
    /// error (caught before any TCP traffic happens).
    #[test]
    fn rejects_rank_id_out_of_bounds_locally() {
        let err = CpuReduceClient::connect(local(12345), 5, 3, TEST_SALT).unwrap_err();
        assert!(
            err.to_string().contains("rank_id 5 must be < world_size 3"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_zero_world_size_locally() {
        let err = CpuReduceClient::connect(local(12345), 0, 0, TEST_SALT).unwrap_err();
        assert!(err.to_string().contains("world_size"), "got: {err}");
    }

    // --- Tensor ↔ RoundFrame conversion ---

    #[test]
    fn tensor_round_trip_through_round_frame() {
        let data: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(data, &[2, 3], Device::CPU).unwrap();
        let refs = vec![&t];
        let frame = tensors_to_round_frame(&refs, DTYPE_F32).unwrap();
        assert_eq!(frame.tensors.len(), 1);
        assert_eq!(frame.tensors[0].dtype, DTYPE_F32);
        assert_eq!(frame.tensors[0].shape, vec![2u32, 3]);
        assert_eq!(frame.tensors[0].bytes.len(), 6 * 4);

        let recovered = round_frame_to_tensors(&frame).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].shape(), vec![2i64, 3]);
        assert_eq!(recovered[0].to_f32_vec().unwrap(), data);
    }

    /// bf16 wire encoding: an f32 tensor serializes to half the bytes
    /// and decodes back exactly for bf16-representable values (the
    /// decode side always returns f32 tensors).
    #[test]
    fn tensor_round_trip_through_round_frame_bf16() {
        let data: &[f32] = &[1.0, -2.5, 0.0, 42.0, 0.125, -256.0];
        let t = Tensor::from_f32(data, &[2, 3], Device::CPU).unwrap();
        let refs = vec![&t];
        let frame =
            tensors_to_round_frame(&refs, crate::distributed::controller::DTYPE_BF16).unwrap();
        assert_eq!(frame.tensors[0].dtype, crate::distributed::controller::DTYPE_BF16);
        assert_eq!(frame.tensors[0].bytes.len(), 6 * 2, "half the f32 bytes");

        let recovered = round_frame_to_tensors(&frame).unwrap();
        assert_eq!(recovered[0].dtype(), crate::tensor::DType::Float32);
        assert_eq!(recovered[0].shape(), vec![2i64, 3]);
        assert_eq!(recovered[0].to_f32_vec().unwrap(), data);
    }

    /// A tensor already staged in bf16 (the pinned-snapshot path)
    /// serializes verbatim into a bf16 frame — no second cast.
    #[test]
    fn bf16_tensor_serializes_verbatim_into_bf16_frame() {
        let t = Tensor::from_f32(&[1.5, -3.0], &[2], Device::CPU)
            .unwrap()
            .to_dtype(crate::tensor::DType::BFloat16)
            .unwrap();
        let frame = tensors_to_round_frame(
            &[&t],
            crate::distributed::controller::DTYPE_BF16,
        )
        .unwrap();
        assert_eq!(frame.tensors[0].bytes, t.to_blob().unwrap());
        // And the safety cast the other way: a bf16-staged tensor on an
        // f32 wire upcasts rather than desyncing the frame schema.
        let f32_frame = tensors_to_round_frame(&[&t], DTYPE_F32).unwrap();
        assert_eq!(f32_frame.tensors[0].dtype, DTYPE_F32);
        assert_eq!(
            round_frame_to_tensors(&f32_frame).unwrap()[0]
                .to_f32_vec()
                .unwrap(),
            vec![1.5, -3.0]
        );
    }

    #[test]
    fn tensors_to_round_frame_rejects_non_f32() {
        let t = Tensor::from_f64(&[1.0, 2.0], &[2], Device::CPU).unwrap();
        let refs = vec![&t];
        let err = tensors_to_round_frame(&refs, DTYPE_F32).unwrap_err();
        assert!(
            err.to_string().contains("Float64") && err.to_string().contains("Float32"),
            "got: {err}"
        );
    }

    #[test]
    fn round_frame_to_tensors_rejects_shape_byte_mismatch() {
        // shape claims 4 elements but only 2 f32 worth of bytes (8 bytes)
        let bogus = RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_F32,
                shape: vec![4],
                bytes: vec![0u8; 8],
            }],
            ..Default::default()
        };
        let err = round_frame_to_tensors(&bogus).unwrap_err();
        let msg = err.to_string();
        // The length check now lives in `Tensor::from_blob` (the decode
        // goes blob-direct, no intermediate Vec); its loud message names
        // the byte count and the numel-derived expectation.
        assert!(
            msg.contains("numel") && msg.contains("expected"),
            "got: {msg}"
        );
    }

    /// End-to-end through a relay: three ranks call
    /// `broadcast_from_root(root=0)` with distinct local tensors. Every
    /// rank must receive rank 0's values.
    #[test]
    fn three_rank_broadcast_from_root_delivers_root_values() {
        let root_vals: [f32; 3] = [1.5, 2.5, 3.5];
        let r1_vals: [f32; 3] = [10.0, 20.0, 30.0];
        let r2_vals: [f32; 3] = [100.0, 200.0, 300.0];

        let (r0, r1, r2) = with_relayed_controller(3, vec![0, 1, 2], TEST_SALT, |addr| {
            let (tx0, rx0) = mpsc::channel();
            let (tx1, rx1) = mpsc::channel();
            let (tx2, rx2) = mpsc::channel();
            let t0 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 0, 3, TEST_SALT).unwrap();
                let t = Tensor::from_f32(&root_vals, &[3], Device::CPU).unwrap();
                tx0.send(c.broadcast_from_root(&[&t], 0).unwrap()).unwrap();
            });
            let t1 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 1, 3, TEST_SALT).unwrap();
                let t = Tensor::from_f32(&r1_vals, &[3], Device::CPU).unwrap();
                tx1.send(c.broadcast_from_root(&[&t], 0).unwrap()).unwrap();
            });
            let t2 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 2, 3, TEST_SALT).unwrap();
                let t = Tensor::from_f32(&r2_vals, &[3], Device::CPU).unwrap();
                tx2.send(c.broadcast_from_root(&[&t], 0).unwrap()).unwrap();
            });
            let r0 = rx0.recv().unwrap();
            let r1 = rx1.recv().unwrap();
            let r2 = rx2.recv().unwrap();
            t0.join().unwrap();
            t1.join().unwrap();
            t2.join().unwrap();
            (r0, r1, r2)
        });

        let expected = root_vals.to_vec();
        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].to_f32_vec().unwrap(), expected, "rank 0 (root)");
        assert_eq!(r1[0].to_f32_vec().unwrap(), expected, "rank 1");
        assert_eq!(r2[0].to_f32_vec().unwrap(), expected, "rank 2");
    }

    /// Multi-tensor broadcast through a relay: a small "parameter list" of
    /// two tensors with distinct shapes.
    #[test]
    fn broadcast_from_root_multi_tensor_distinct_shapes() {
        let root_w: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let root_b: [f32; 4] = [-1.0, -2.0, -3.0, -4.0];
        let r1_w: [f32; 6] = [9.0; 6];
        let r1_b: [f32; 4] = [9.0; 4];

        let (r0, r1) = with_relayed_controller(2, vec![0, 1], TEST_SALT, |addr| {
            let (tx0, rx0) = mpsc::channel();
            let (tx1, rx1) = mpsc::channel();
            let t0 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 0, 2, TEST_SALT).unwrap();
                let w = Tensor::from_f32(&root_w, &[2, 3], Device::CPU).unwrap();
                let b = Tensor::from_f32(&root_b, &[4], Device::CPU).unwrap();
                tx0.send(c.broadcast_from_root(&[&w, &b], 0).unwrap()).unwrap();
            });
            let t1 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 1, 2, TEST_SALT).unwrap();
                let w = Tensor::from_f32(&r1_w, &[2, 3], Device::CPU).unwrap();
                let b = Tensor::from_f32(&r1_b, &[4], Device::CPU).unwrap();
                tx1.send(c.broadcast_from_root(&[&w, &b], 0).unwrap()).unwrap();
            });
            let r0 = rx0.recv().unwrap();
            let r1 = rx1.recv().unwrap();
            t0.join().unwrap();
            t1.join().unwrap();
            (r0, r1)
        });

        assert_eq!(r0.len(), 2);
        assert_eq!(r1.len(), 2);
        assert_eq!(r0[0].shape(), vec![2_i64, 3]);
        assert_eq!(r0[1].shape(), vec![4_i64]);
        assert_eq!(r0[0].to_f32_vec().unwrap(), root_w.to_vec());
        assert_eq!(r0[1].to_f32_vec().unwrap(), root_b.to_vec());
        assert_eq!(r1[0].to_f32_vec().unwrap(), root_w.to_vec());
        assert_eq!(r1[1].to_f32_vec().unwrap(), root_b.to_vec());
    }

    /// End-to-end through a relay: two ranks ship Tensor lists; receive
    /// averaged Tensor lists back.
    #[test]
    fn two_rank_tensor_average() {
        let (r0, r1) = with_relayed_controller(2, vec![0, 1], TEST_SALT, |addr| {
            let (tx0, rx0) = mpsc::channel();
            let (tx1, rx1) = mpsc::channel();
            let t0 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 0, 2, TEST_SALT).unwrap();
                let t = Tensor::from_f32(&[2.0, 4.0, 6.0], &[3], Device::CPU).unwrap();
                tx0.send(c.all_reduce_tensors(&[&t]).unwrap()).unwrap();
                drop(c);
            });
            let t1 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 1, 2, TEST_SALT).unwrap();
                let t = Tensor::from_f32(&[4.0, 8.0, 12.0], &[3], Device::CPU).unwrap();
                tx1.send(c.all_reduce_tensors(&[&t]).unwrap()).unwrap();
                drop(c);
            });
            let r0 = rx0.recv().unwrap();
            let r1 = rx1.recv().unwrap();
            t0.join().unwrap();
            t1.join().unwrap();
            (r0, r1)
        });

        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].to_f32_vec().unwrap(), vec![3.0, 6.0, 9.0]);
        assert_eq!(r1[0].to_f32_vec().unwrap(), vec![3.0, 6.0, 9.0]);
    }

    /// Full bf16 wire path end-to-end: client encode → relay fold parse
    /// → controller f32 sum + divide → bf16 scatter → client decode.
    /// Values chosen bf16-exact so the average is exact too.
    #[test]
    fn two_rank_tensor_average_bf16_wire() {
        let (r0, r1) = with_relayed_controller(2, vec![0, 1], TEST_SALT, |addr| {
            let (tx0, rx0) = mpsc::channel();
            let (tx1, rx1) = mpsc::channel();
            let t0 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 0, 2, TEST_SALT).unwrap();
                c.set_bf16_wire(true);
                let t = Tensor::from_f32(&[2.0, 4.0, 6.0], &[3], Device::CPU).unwrap();
                tx0.send(c.all_reduce_tensors(&[&t]).unwrap()).unwrap();
                drop(c);
            });
            let t1 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 1, 2, TEST_SALT).unwrap();
                c.set_bf16_wire(true);
                let t = Tensor::from_f32(&[4.0, 8.0, 12.0], &[3], Device::CPU).unwrap();
                tx1.send(c.all_reduce_tensors(&[&t]).unwrap()).unwrap();
                drop(c);
            });
            let r0 = rx0.recv().unwrap();
            let r1 = rx1.recv().unwrap();
            t0.join().unwrap();
            t1.join().unwrap();
            (r0, r1)
        });

        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].to_f32_vec().unwrap(), vec![3.0, 6.0, 9.0]);
        assert_eq!(r1[0].to_f32_vec().unwrap(), vec![3.0, 6.0, 9.0]);
        assert_eq!(r0[0].dtype(), crate::tensor::DType::Float32, "decode returns f32");
    }

    /// The fused byte-level γ-scale must match the tensor-level
    /// `mul_scalar` it replaced, bit-for-bit, in both wire dtypes —
    /// libtorch's scalar mul computes in f32 and rounds to nearest-even,
    /// exactly like `scale_payload_bytes`' decode → f32 multiply →
    /// re-encode. A failure here means the fused path changed reduce
    /// numerics, not just its RAM profile.
    #[test]
    fn fused_byte_scale_matches_mul_scalar() {
        use crate::distributed::controller::scale_payload_bytes;
        let vals = [0.731_442_6f32, -1.902_337_5, 3.017_882e-3, -42.523_86];
        let w = 0.834_921_7f64;
        for (dtype, tag) in [
            (DType::Float32, DTYPE_F32),
            (DType::BFloat16, DTYPE_BF16),
        ] {
            let t = Tensor::from_f32(&vals, &[4], Device::CPU)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let expected = t.mul_scalar(w).unwrap().to_blob().unwrap();
            let mut fused = t.to_blob().unwrap();
            scale_payload_bytes(&mut fused, tag, w as f32).unwrap();
            assert_eq!(fused, expected, "dtype {dtype:?}");
        }
    }

    /// `all_reduce_scaled` end-to-end on the bf16 wire with UNEQUAL
    /// masses: the fused sender-side scale must produce the same
    /// realized-work consensus the prescaled path would. Values chosen
    /// so every intermediate (scaled contributions, sum, consensus) is
    /// bf16-exact: (3·[2,8] + 1·[4,16]) / 4 = [2.5, 10.0].
    #[test]
    fn two_rank_scaled_weighted_average_bf16_wire() {
        use crate::distributed::controller::RoundKind;
        let (r0, r1) = with_relayed_controller(2, vec![0, 1], TEST_SALT, |addr| {
            let spawn = |rank: u32, vals: [f32; 2], w: f64| {
                thread::spawn(move || {
                    let mut c =
                        CpuReduceClient::connect(addr, rank, 2, TEST_SALT).unwrap();
                    c.set_bf16_wire(true);
                    let t = Tensor::from_f32(&vals, &[2], Device::CPU).unwrap();
                    c.all_reduce_scaled(&[&t], w, RoundKind::Model, w).unwrap()
                })
            };
            let t0 = spawn(0, [2.0, 8.0], 3.0);
            let t1 = spawn(1, [4.0, 16.0], 1.0);
            (t0.join().unwrap(), t1.join().unwrap())
        });

        for (rank, (tensors, mass)) in [(0, &r0), (1, &r1)] {
            assert_eq!(*mass, 4.0, "rank {rank} accepted mass");
            assert_eq!(
                tensors[0].to_f32_vec().unwrap(),
                vec![2.5, 10.0],
                "rank {rank} consensus"
            );
        }
    }

    /// Connect failure (no controller listening) surfaces a clear error.
    #[test]
    fn surfaces_connect_failure_clearly() {
        // 1 is a reserved port — connect should fail with refused.
        let err = CpuReduceClient::connect(local(1), 0, 1, TEST_SALT).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("connect") && msg.contains(":1 failed"),
            "expected connect-failed error mentioning port 1, got: {msg}"
        );
    }

    // -- realized-work invariant -------------------------------------------
    //
    // The consensus equals Σ(wᵣ·Tᵣ) / Σwᵣ over exactly the contributions
    // the controller accepted into the round ("realized work"), and the
    // frame the controller scatters — which the checkpoint forge and the
    // controller-hosted outer optimizer consume — IS that consensus.
    //
    // These drive the REAL rank-side math (`realized_work::gamma_mass` +
    // `sumcount_reduce`) through the true 3-tier path
    // (client → relay → controller), all-alive and after a death.

    use crate::distributed::cluster_worker::sumcount_reduce;
    use crate::distributed::realized_work::gamma_mass;
    use crate::distributed::controller::DeadRanks;
    use crate::tensor::{Device, Tensor};
    use std::sync::Arc;

    /// Like `with_relayed_controller` but shares a caller-provided
    /// dead-rank ledger so tests can declare deaths.
    fn with_relayed_controller_and_ledger<F, T>(
        world_size: u32,
        ranks: Vec<u32>,
        salt: SessionSalt,
        dead: Arc<DeadRanks>,
        client_body: F,
    ) -> T
    where
        F: FnOnce(SocketAddr) -> T,
    {
        let controller = ClusterController::start_with_dead_ranks(
            local(0),
            world_size as usize,
            salt,
            dead,
            None,
            None,
        )
        .unwrap();
        let controller_addr = local(controller.port());
        let (listener, relay_port) = RelayChannel::bind(local(0)).unwrap();
        let (htx, hrx) = mpsc::channel();
        thread::spawn(move || {
            let started = RelayChannel::start(
                listener,
                ChannelKind::Data,
                controller_addr,
                "test-host".into(),
                ranks,
                world_size as usize,
                salt,
            );
            let _ = htx.send(started);
        });
        let result = client_body(local(relay_port));
        match hrx.recv().unwrap() {
            Ok(ch) => {
                let _ = ch.shutdown();
            }
            Err(e) => panic!("relay start failed: {e}"),
        }
        controller.shutdown().unwrap();
        result
    }

    /// One rank's full param-bridge window math against live wire state:
    /// count gather → gamma weights → weighted reduce. Returns
    /// (gathered counts, rank-adopted consensus value, raw scattered
    /// value = what the forge/outer stepper see).
    fn window_round(
        c: &mut CpuReduceClient,
        rank: usize,
        world: usize,
        n_i: usize,
        t_val: f32,
        gamma: f64,
    ) -> (Vec<f64>, f32, f32) {
        let mut counts = vec![0.0f64; world];
        counts[rank] = n_i as f64;
        c.all_reduce_per_rank_f64(&mut counts).unwrap();
        let my_w = gamma_mass(n_i as f64, gamma);
        let t = Tensor::from_f32(&[t_val], &[1], Device::CPU).unwrap();
        let adopted = sumcount_reduce(c, std::slice::from_ref(&t), my_w)
            .unwrap()[0]
            .to_f32_vec()
            .unwrap()[0];
        // Second round replicating `sumcount_reduce`'s send side, keeping
        // the RAW reduce output: byte-identical to the frame the
        // controller scatters and hands to the forge / outer stepper.
        let scaled = t.mul_scalar(my_w).unwrap();
        let (raw_t, _mass) = c
            .all_reduce_weighted(
                &[&scaled],
                crate::distributed::controller::RoundKind::Model,
                my_w,
            )
            .unwrap();
        let raw = raw_t[0].to_f32_vec().unwrap()[0];
        let _ = world;
        (counts, adopted, raw)
    }

    /// All ranks alive, γ=1, counts [3,5,2], values [1,2,3]:
    /// consensus = (3·1 + 5·2 + 2·3) / 10 = 1.9. Both the rank-adopted
    /// value AND the scattered frame must equal it.
    #[test]
    fn realized_work_all_alive_scattered_frame_is_consensus() {
        let dead = DeadRanks::new(3);
        let (r0, r1, r2) = with_relayed_controller_and_ledger(
            3,
            vec![0, 1, 2],
            TEST_SALT,
            dead,
            |addr| {
                let spawn = |rank: usize, n: usize, v: f32| {
                    thread::spawn(move || {
                        let mut c =
                            CpuReduceClient::connect(addr, rank as u32, 3, TEST_SALT)
                                .unwrap();
                        window_round(&mut c, rank, 3, n, v, 1.0)
                    })
                };
                let t0 = spawn(0, 3, 1.0);
                let t1 = spawn(1, 5, 2.0);
                let t2 = spawn(2, 2, 3.0);
                (t0.join().unwrap(), t1.join().unwrap(), t2.join().unwrap())
            },
        );
        for (rank, (counts, adopted, raw)) in
            [(0, &r0), (1, &r1), (2, &r2)]
        {
            assert_eq!(counts, &vec![3.0, 5.0, 2.0], "rank {rank} gathered counts");
            assert!(
                (adopted - 1.9).abs() < 1e-6,
                "rank {rank} adopted {adopted}, want consensus 1.9"
            );
            assert!(
                (raw - 1.9).abs() < 1e-6,
                "rank {rank} scattered frame carries {raw}, want the \
                 consensus 1.9 — this is the value the checkpoint forge \
                 writes and the outer stepper transforms"
            );
        }
    }

    /// Rank 2 declared dead before the window; ranks 0,1 do counts [3,5],
    /// values [1,2], buffers [1,2]. Realized work says: params
    /// consensus = 13/8 = 1.625 (γ=1), buffers = mean of movers = 1.5,
    /// γ=0.5 params = (√3·1 + √5·2)/(√3+√5) ≈ 1.563798, and the gathered
    /// count vector reports the realized counts [3,5,0].
    #[test]
    fn realized_work_after_death() {
        let dead = DeadRanks::new(3);
        let dead_for_body = Arc::clone(&dead);
        let (r0, r1) = with_relayed_controller_and_ledger(
            3,
            vec![0, 1, 2],
            TEST_SALT,
            dead,
            move |addr| {
                // Rank 2 connects (relay + controller cohort formation
                // need all announced ranks) then goes permanently silent.
                let (park_tx, park_rx) = mpsc::channel::<()>();
                let t2 = thread::spawn(move || {
                    let _c =
                        CpuReduceClient::connect(addr, 2, 3, TEST_SALT).unwrap();
                    let _ = park_rx.recv(); // hold the connection open, idle
                });
                dead_for_body.declare_dead(2);
                let spawn = |rank: usize, n: usize, v: f32, b: f32| {
                    thread::spawn(move || {
                        let mut c =
                            CpuReduceClient::connect(addr, rank as u32, 3, TEST_SALT)
                                .unwrap();
                        let g1 = window_round(&mut c, rank, 3, n, v, 1.0);
                        // Buffer window: mover-indicator weighting.
                        let bt =
                            Tensor::from_f32(&[b], &[1], Device::CPU).unwrap();
                        let buf = sumcount_reduce(
                            &mut c,
                            std::slice::from_ref(&bt),
                            1.0,
                        )
                        .unwrap()[0]
                            .to_f32_vec()
                            .unwrap()[0];
                        let g_half = window_round(&mut c, rank, 3, n, v, 0.5);
                        (g1, buf, g_half)
                    })
                };
                let t0 = spawn(0, 3, 1.0, 1.0);
                let t1 = spawn(1, 5, 2.0, 2.0);
                let out = (t0.join().unwrap(), t1.join().unwrap());
                let _ = park_tx.send(());
                t2.join().unwrap();
                out
            },
        );
        let want_gamma_half = (3f64.sqrt() + 2.0 * 5f64.sqrt())
            / (3f64.sqrt() + 5f64.sqrt());
        for (rank, ((counts, adopted, raw), buf, (_, adopted_h, _))) in
            [(0, &r0), (1, &r1)]
        {
            assert_eq!(
                counts,
                &vec![3.0, 5.0, 0.0],
                "rank {rank}: gathered counts must be the realized counts"
            );
            assert!(
                (adopted - 1.625).abs() < 1e-6,
                "rank {rank} adopted {adopted}, want realized consensus 1.625"
            );
            assert!(
                (raw - 1.625).abs() < 1e-6,
                "rank {rank} scattered frame carries {raw}, want 1.625"
            );
            assert!(
                (buf - 1.5).abs() < 1e-6,
                "rank {rank} buffer consensus {buf}, want mean-of-movers 1.5 \
                 — inflation here compounds every window on running stats"
            );
            assert!(
                (*adopted_h as f64 - want_gamma_half).abs() < 1e-5,
                "rank {rank} γ=0.5 adopted {adopted_h}, want {want_gamma_half}"
            );
        }
    }
