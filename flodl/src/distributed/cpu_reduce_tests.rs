    use super::*;
    use crate::distributed::controller::{
        ClusterController, DTYPE_F32, RoundFrame, TensorPayload,
    };
    use std::net::Ipv4Addr;
    use std::sync::mpsc;
    use std::thread;

    /// Deterministic non-zero test salt; matches the controller-side
    /// constant in `controller::tests` so the two test modules could
    /// theoretically interop (they don't today, but might be wired
    /// together in future cross-component tests).
    const TEST_SALT: SessionSalt = [
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    ];

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
        }
    }

    /// End-to-end: spawn the controller and two rank clients via this
    /// crate's `CpuReduceClient`; verify the average comes back to each.
    #[test]
    fn two_rank_client_average() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);

        let (tx0, rx0) = mpsc::channel();
        let (tx1, rx1) = mpsc::channel();
        let t0 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 0, 2, TEST_SALT).unwrap();
            let avg_frame = c.all_reduce(&frame_with(&[2.0, 4.0, 6.0])).unwrap();
            tx0.send(avg_frame).unwrap();
            drop(c);
        });
        let t1 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 1, 2, TEST_SALT).unwrap();
            let avg_frame = c.all_reduce(&frame_with(&[4.0, 8.0, 12.0])).unwrap();
            tx1.send(avg_frame).unwrap();
            drop(c);
        });

        let r0 = rx0.recv().unwrap();
        let r1 = rx1.recv().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        avg.shutdown().unwrap();

        let avg0 = bytes_as_f32(&r0.tensors[0].bytes);
        assert_eq!(avg0, vec![3.0, 6.0, 9.0]);
        assert_eq!(r0, r1);
    }

    /// Each client survives multiple rounds, gets the per-round average
    /// back. Exercises the persistent-connection path.
    #[test]
    fn client_multi_round_persistence() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);

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

        let r0_results = t0.join().unwrap();
        let r1_results = t1.join().unwrap();
        avg.shutdown().unwrap();

        // Round-by-round averages: (1,3)/2=2, (5,7)/2=6, (9,11)/2=10.
        let expected = [2.0_f32, 6.0, 10.0];
        for (i, want) in expected.iter().enumerate() {
            let got_0 = bytes_as_f32(&r0_results[i].tensors[0].bytes);
            let got_1 = bytes_as_f32(&r1_results[i].tensors[0].bytes);
            assert_eq!(got_0, vec![*want], "rank 0 round {i}");
            assert_eq!(got_1, vec![*want], "rank 1 round {i}");
        }
    }

    /// Rank disagreeing on world_size with the controller must surface
    /// loudly (controller drops the bad rank; our handshake_ack read
    /// then fails).
    #[test]
    fn rejects_world_size_disagreement() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);

        // Rank claims world_size = 3 but controller is configured for 2.
        let err = CpuReduceClient::connect(addr, 0, 3, TEST_SALT).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("ack") || msg.contains("handshake") || msg.contains("read"),
            "expected wire-level error, got: {msg}"
        );
        let _ = avg.shutdown();
    }

    /// Constructing a client with rank_id >= world_size is a local-only
    /// error (caught before any TCP traffic happens).
    #[test]
    fn rejects_rank_id_out_of_bounds_locally() {
        let err = CpuReduceClient::connect(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 12345),
            5,
            3,
            TEST_SALT,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("rank_id 5 must be < world_size 3"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_zero_world_size_locally() {
        let err = CpuReduceClient::connect(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 12345),
            0,
            0,
            TEST_SALT,
        )
        .unwrap_err();
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
        };
        let err = round_frame_to_tensors(&bogus).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("numel") && (msg.contains("!=") || msg.contains("mismatch")),
            "got: {msg}"
        );
    }

    /// End-to-end: three ranks call `broadcast_from_root(root=0)` with
    /// distinct local tensors. Every rank must receive rank 0's values.
    ///
    /// This is the bootstrap-time path used by cluster-rank entry points
    /// to align initial parameter state before training. Regressing this
    /// would let ranks start with divergent weights even when the user's
    /// factory is non-deterministic — the exact "initial broadcast not
    /// transferring root's params" failure mode flagged for B1.
    #[test]
    fn three_rank_broadcast_from_root_delivers_root_values() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            3,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);

        let root_vals: [f32; 3] = [1.5, 2.5, 3.5];
        let r1_vals: [f32; 3] = [10.0, 20.0, 30.0];
        let r2_vals: [f32; 3] = [100.0, 200.0, 300.0];

        let (tx0, rx0) = mpsc::channel();
        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        let t0 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 0, 3, TEST_SALT).unwrap();
            let t = Tensor::from_f32(&root_vals, &[3], Device::CPU).unwrap();
            let out = c.broadcast_from_root(&[&t], 0).unwrap();
            tx0.send(out).unwrap();
        });
        let t1 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 1, 3, TEST_SALT).unwrap();
            let t = Tensor::from_f32(&r1_vals, &[3], Device::CPU).unwrap();
            let out = c.broadcast_from_root(&[&t], 0).unwrap();
            tx1.send(out).unwrap();
        });
        let t2 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 2, 3, TEST_SALT).unwrap();
            let t = Tensor::from_f32(&r2_vals, &[3], Device::CPU).unwrap();
            let out = c.broadcast_from_root(&[&t], 0).unwrap();
            tx2.send(out).unwrap();
        });

        let r0 = rx0.recv().unwrap();
        let r1 = rx1.recv().unwrap();
        let r2 = rx2.recv().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        t2.join().unwrap();
        avg.shutdown().unwrap();

        let expected = root_vals.to_vec();
        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].to_f32_vec().unwrap(), expected, "rank 0 (root)");
        assert_eq!(r1[0].to_f32_vec().unwrap(), expected, "rank 1");
        assert_eq!(r2[0].to_f32_vec().unwrap(), expected, "rank 2");
    }

    /// Multi-tensor broadcast: a small "parameter list" of two tensors
    /// with distinct shapes. Mirrors the orchestrator call site, which
    /// passes the full `parameters()` Vec at once.
    #[test]
    fn broadcast_from_root_multi_tensor_distinct_shapes() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);

        // Root tensors: a [2,3] "weight" and a [4] "bias".
        let root_w: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let root_b: [f32; 4] = [-1.0, -2.0, -3.0, -4.0];
        // Non-root sends garbage; broadcast must overwrite it.
        let r1_w: [f32; 6] = [9.0; 6];
        let r1_b: [f32; 4] = [9.0; 4];

        let (tx0, rx0) = mpsc::channel();
        let (tx1, rx1) = mpsc::channel();
        let t0 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 0, 2, TEST_SALT).unwrap();
            let w = Tensor::from_f32(&root_w, &[2, 3], Device::CPU).unwrap();
            let b = Tensor::from_f32(&root_b, &[4], Device::CPU).unwrap();
            let out = c.broadcast_from_root(&[&w, &b], 0).unwrap();
            tx0.send(out).unwrap();
        });
        let t1 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 1, 2, TEST_SALT).unwrap();
            let w = Tensor::from_f32(&r1_w, &[2, 3], Device::CPU).unwrap();
            let b = Tensor::from_f32(&r1_b, &[4], Device::CPU).unwrap();
            let out = c.broadcast_from_root(&[&w, &b], 0).unwrap();
            tx1.send(out).unwrap();
        });

        let r0 = rx0.recv().unwrap();
        let r1 = rx1.recv().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        avg.shutdown().unwrap();

        assert_eq!(r0.len(), 2);
        assert_eq!(r1.len(), 2);
        assert_eq!(r0[0].shape(), vec![2_i64, 3]);
        assert_eq!(r0[1].shape(), vec![4_i64]);
        assert_eq!(r0[0].to_f32_vec().unwrap(), root_w.to_vec());
        assert_eq!(r0[1].to_f32_vec().unwrap(), root_b.to_vec());
        assert_eq!(r1[0].to_f32_vec().unwrap(), root_w.to_vec());
        assert_eq!(r1[1].to_f32_vec().unwrap(), root_b.to_vec());
    }

    /// End-to-end: two ranks ship Tensor lists through ClusterController;
    /// receive averaged Tensor lists back.
    #[test]
    fn two_rank_tensor_average() {
        let avg = crate::distributed::controller::ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);

        let (tx0, rx0) = mpsc::channel();
        let (tx1, rx1) = mpsc::channel();
        let t0 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 0, 2, TEST_SALT).unwrap();
            let t = Tensor::from_f32(&[2.0, 4.0, 6.0], &[3], Device::CPU).unwrap();
            let out = c.all_reduce_tensors(&[&t]).unwrap();
            tx0.send(out).unwrap();
            drop(c);
        });
        let t1 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 1, 2, TEST_SALT).unwrap();
            let t = Tensor::from_f32(&[4.0, 8.0, 12.0], &[3], Device::CPU).unwrap();
            let out = c.all_reduce_tensors(&[&t]).unwrap();
            tx1.send(out).unwrap();
            drop(c);
        });

        let r0 = rx0.recv().unwrap();
        let r1 = rx1.recv().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        avg.shutdown().unwrap();

        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].to_f32_vec().unwrap(), vec![3.0, 6.0, 9.0]);
        assert_eq!(r1[0].to_f32_vec().unwrap(), vec![3.0, 6.0, 9.0]);
    }

    /// Connect failure (no controller listening) surfaces a clear error.
    #[test]
    fn surfaces_connect_failure_clearly() {
        // 1 is a reserved port — connect should fail with refused.
        let err = CpuReduceClient::connect(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 1),
            0,
            1,
            TEST_SALT,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("connect") && msg.contains(":1 failed"),
            "expected connect-failed error mentioning port 1, got: {msg}"
        );
    }
