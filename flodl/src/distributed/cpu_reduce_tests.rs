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
                tx0.send(c.all_reduce(&frame_with(&[2.0, 4.0, 6.0])).unwrap())
                    .unwrap();
                drop(c);
            });
            let t1 = thread::spawn(move || {
                let mut c = CpuReduceClient::connect(addr, 1, 2, TEST_SALT).unwrap();
                tx1.send(c.all_reduce(&frame_with(&[4.0, 8.0, 12.0])).unwrap())
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
                    let r1 = c.all_reduce(&frame_with(&[1.0])).unwrap();
                    let r2 = c.all_reduce(&frame_with(&[5.0])).unwrap();
                    let r3 = c.all_reduce(&frame_with(&[9.0])).unwrap();
                    vec![r1, r2, r3]
                });
                let t1 = thread::spawn(move || -> Vec<RoundFrame> {
                    let mut c = CpuReduceClient::connect(addr, 1, 2, TEST_SALT).unwrap();
                    let r1 = c.all_reduce(&frame_with(&[3.0])).unwrap();
                    let r2 = c.all_reduce(&frame_with(&[7.0])).unwrap();
                    let r3 = c.all_reduce(&frame_with(&[11.0])).unwrap();
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
        let frame = tensors_to_round_frame(&refs).unwrap();
        assert_eq!(frame.tensors.len(), 1);
        assert_eq!(frame.tensors[0].dtype, DTYPE_F32);
        assert_eq!(frame.tensors[0].shape, vec![2u32, 3]);
        assert_eq!(frame.tensors[0].bytes.len(), 6 * 4);

        let recovered = round_frame_to_tensors(&frame).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].shape(), vec![2i64, 3]);
        assert_eq!(recovered[0].to_f32_vec().unwrap(), data);
    }

    #[test]
    fn tensors_to_round_frame_rejects_non_f32() {
        let t = Tensor::from_f64(&[1.0, 2.0], &[2], Device::CPU).unwrap();
        let refs = vec![&t];
        let err = tensors_to_round_frame(&refs).unwrap_err();
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
        assert!(
            msg.contains("numel") && (msg.contains("!=") || msg.contains("mismatch")),
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
