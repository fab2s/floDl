    use super::*;
    use crate::tensor::TensorOptions;

    fn make_named_params(sizes: &[(i64, i64)]) -> Vec<(String, Parameter)> {
        sizes.iter().enumerate().map(|(i, &(rows, cols))| {
            let t = Tensor::randn(&[rows, cols], TensorOptions {
                dtype: DType::Float32,
                device: crate::tensor::test_device(),
            }).unwrap();
            let name = format!("layer_{}/weight", i);
            (name.clone(), Parameter::new(t, "weight"))
        }).collect()
    }

    fn make_named_buffers(sizes: &[i64]) -> Vec<(String, Buffer)> {
        sizes.iter().enumerate().map(|(i, &features)| {
            let t = Tensor::randn(&[features], TensorOptions {
                dtype: DType::Float32,
                device: crate::tensor::test_device(),
            }).unwrap();
            let name = format!("bn_{}/running_mean", i);
            (name.clone(), Buffer::new(t, "running_mean"))
        }).collect()
    }

    #[test]
    fn test_named_roundtrip() {
        let params = make_named_params(&[(4, 8), (8, 2)]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], None).unwrap();

        let load_params = make_named_params(&[(4, 8), (8, 2)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &load_params, &[], None).unwrap();

        assert_eq!(report.loaded.len(), 2);
        assert!(report.skipped.is_empty());
        assert!(report.missing.is_empty());

        for ((_, src), (_, dst)) in params.iter().zip(load_params.iter()) {
            let src_data = src.variable.data().to_f32_vec().unwrap();
            let dst_data = dst.variable.data().to_f32_vec().unwrap();
            assert_eq!(src_data, dst_data);
        }
    }

    #[test]
    fn test_buffer_roundtrip() {
        let params = make_named_params(&[(4, 8)]);
        let buffers = make_named_buffers(&[8]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &buffers, None).unwrap();

        // Fresh model with same structure
        let load_params = make_named_params(&[(4, 8)]);
        let load_buffers = make_named_buffers(&[8]);
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &load_params, &load_buffers, None).unwrap();

        assert_eq!(report.loaded.len(), 2); // 1 param + 1 buffer
        assert!(report.skipped.is_empty());
        assert!(report.missing.is_empty());

        // Verify buffer data matches
        let src_data = buffers[0].1.get().to_f32_vec().unwrap();
        let dst_data = load_buffers[0].1.get().to_f32_vec().unwrap();
        assert_eq!(src_data, dst_data);
    }

    #[test]
    fn test_named_partial_load() {
        let params_3 = make_named_params(&[(4, 8), (8, 4), (4, 2)]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params_3, &[], None).unwrap();

        let mut params_4 = make_named_params(&[(4, 8), (8, 4), (4, 2), (2, 1)]);
        params_4[3].0 = "extra/weight".to_string();

        let before_extra = params_4[3].1.variable.data().to_f32_vec().unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &params_4, &[], None).unwrap();

        assert_eq!(report.loaded.len(), 3);
        assert_eq!(report.missing.len(), 1);
        assert_eq!(report.missing[0], "extra/weight");
        assert!(report.skipped.is_empty());

        let after_extra = params_4[3].1.variable.data().to_f32_vec().unwrap();
        assert_eq!(before_extra, after_extra);
    }

    #[test]
    fn test_named_skipped_checkpoint_params() {
        let params = make_named_params(&[(4, 8), (8, 2)]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], None).unwrap();

        let model = vec![params[0].clone()];
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &model, &[], None).unwrap();

        assert_eq!(report.loaded.len(), 1);
        assert_eq!(report.skipped.len(), 1);
        assert!(report.missing.is_empty());
    }

    #[test]
    fn test_named_shape_mismatch_error() {
        let params = make_named_params(&[(4, 8)]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], None).unwrap();

        let wrong_shape = vec![(
            "layer_0/weight".to_string(),
            Parameter::new(
                Tensor::randn(&[4, 4], TensorOptions {
                    dtype: DType::Float32,
                    device: crate::tensor::test_device(),
                }).unwrap(),
                "weight",
            ),
        )];
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &wrong_shape, &[], None);
        assert!(result.is_err(), "shape mismatch should be an error");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("shape mismatch"), "error should mention shape: {}", err_msg);
    }

    #[test]
    fn test_buffer_shape_mismatch_error() {
        let buffers = make_named_buffers(&[8]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &[], &buffers, None).unwrap();

        let wrong_buffers = vec![(
            "bn_0/running_mean".to_string(),
            Buffer::new(
                Tensor::zeros(&[4], crate::tensor::test_opts()).unwrap(),
                "running_mean",
            ),
        )];
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &[], &wrong_buffers, None);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("shape mismatch"));
    }

    #[test]
    fn test_compressed_roundtrip() {
        let params = make_named_params(&[(16, 32), (32, 8)]);
        let buffers = make_named_buffers(&[32]);

        let dir = std::env::temp_dir();
        let gz_path = dir.join("test_ckpt_v2.fdl.gz");
        let plain_path = dir.join("test_ckpt_v2.fdl");
        let gz = gz_path.to_str().unwrap();
        let plain = plain_path.to_str().unwrap();

        save_checkpoint_file(gz, &params, &buffers, None).unwrap();
        save_checkpoint_file(plain, &params, &buffers, None).unwrap();

        // Compressed should be smaller
        let gz_size = std::fs::metadata(gz).unwrap().len();
        let plain_size = std::fs::metadata(plain).unwrap().len();
        assert!(gz_size < plain_size, "gz={} should be < plain={}", gz_size, plain_size);

        // Load from compressed and verify
        let load_params = make_named_params(&[(16, 32), (32, 8)]);
        let load_buffers = make_named_buffers(&[32]);
        let report = load_checkpoint_file(gz, &load_params, &load_buffers, None).unwrap();
        assert_eq!(report.loaded.len(), 3); // 2 params + 1 buffer

        for ((_, src), (_, dst)) in params.iter().zip(load_params.iter()) {
            assert_eq!(src.variable.data().to_f32_vec().unwrap(),
                       dst.variable.data().to_f32_vec().unwrap());
        }

        let src_buf = buffers[0].1.get().to_f32_vec().unwrap();
        let dst_buf = load_buffers[0].1.get().to_f32_vec().unwrap();
        assert_eq!(src_buf, dst_buf);

        std::fs::remove_file(gz).ok();
        std::fs::remove_file(plain).ok();
    }

    #[test]
    fn test_hash_roundtrip() {
        let params = make_named_params(&[(4, 8)]);
        // Use a known 64-char hex hash
        let hash = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6a7b8c9d0e1f2a3b4c5d6a7b8c9d0e1f2";

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], Some(hash)).unwrap();

        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        // Same hash: should succeed
        let report = load_checkpoint(&mut cursor, &load_params, &[], Some(hash)).unwrap();
        assert_eq!(report.loaded.len(), 1);
    }

    #[test]
    fn test_hash_mismatch_error() {
        let params = make_named_params(&[(4, 8)]);
        let hash_a = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6a7b8c9d0e1f2a3b4c5d6a7b8c9d0e1f2";
        let hash_b = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], Some(hash_a)).unwrap();

        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &load_params, &[], Some(hash_b));
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("architecture mismatch"), "error: {}", msg);
    }

    #[test]
    fn test_zero_hash_skips_validation() {
        let params = make_named_params(&[(4, 8)]);

        // Save with no hash (zero bytes)
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], None).unwrap();

        // Load with a hash expectation — should still succeed (file has zeros)
        let hash = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &load_params, &[], Some(hash)).unwrap();
        assert_eq!(report.loaded.len(), 1);

        // Save with hash, load with None — should succeed (no expected hash)
        let mut buf2 = Vec::new();
        save_checkpoint(&mut buf2, &params, &[], Some(hash)).unwrap();
        let load_params2 = make_named_params(&[(4, 8)]);
        let mut cursor2 = std::io::Cursor::new(&buf2);
        let report2 = load_checkpoint(&mut cursor2, &load_params2, &[], None).unwrap();
        assert_eq!(report2.loaded.len(), 1);
    }

    /// Write a checkpoint with an explicit version byte (for testing v1 migration).
    fn save_checkpoint_versioned<W: std::io::Write>(
        w: &mut W,
        version: u32,
        params: &[(String, Parameter)],
        buffers: &[(String, Buffer)],
    ) {
        w.write_all(&MAGIC).unwrap();
        w.write_all(&version.to_le_bytes()).unwrap();
        w.write_all(&[0u8; HASH_LEN]).unwrap();
        let total = (params.len() + buffers.len()) as u32;
        w.write_all(&total.to_le_bytes()).unwrap();
        for (name, p) in params {
            let name_bytes = name.as_bytes();
            w.write_all(&(name_bytes.len() as u32).to_le_bytes()).unwrap();
            w.write_all(name_bytes).unwrap();
            write_tensor_data(w, &p.variable.data()).unwrap();
        }
        for (name, b) in buffers {
            let name_bytes = name.as_bytes();
            w.write_all(&(name_bytes.len() as u32).to_le_bytes()).unwrap();
            w.write_all(name_bytes).unwrap();
            write_tensor_data(w, &b.get()).unwrap();
        }
    }

    #[test]
    fn test_migrate_all_renamed() {
        // Simulate v1 checkpoint with old-style names
        let old_params = vec![
            ("linear_0/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("linear_1/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 2], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &[]);

        // New model with renamed tags
        let new_params = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("decoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 2], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let mut out = Vec::new();
        let report = migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &[],
        ).unwrap();

        assert!(report.unchanged.is_empty());
        assert_eq!(report.remapped.len(), 2);
        assert!(report.dropped.is_empty());
        assert!(report.missing.is_empty());
        assert!(report.is_complete());

        // Verify the migrated checkpoint loads correctly
        let verify_params = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("decoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 2], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&out);
        let load_report = load_checkpoint(&mut cursor, &verify_params, &[], None).unwrap();
        assert_eq!(load_report.loaded.len(), 2);
        assert!(load_report.missing.is_empty());

        // Verify data preserved: old param data matches loaded data
        for (i, (_, vp)) in verify_params.iter().enumerate() {
            let expected = old_params[i].1.variable.data().to_f32_vec().unwrap();
            let got = vp.variable.data().to_f32_vec().unwrap();
            assert_eq!(expected, got, "data mismatch for param {}", i);
        }
    }

    #[test]
    fn test_migrate_partial_rename() {
        // Some names match, some don't
        let old_params = vec![
            ("shared/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("linear_0/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 2], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &[]);

        let new_params = vec![
            ("shared/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 2], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let mut out = Vec::new();
        let report = migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &[],
        ).unwrap();

        assert_eq!(report.unchanged, vec!["shared/weight"]);
        assert_eq!(report.remapped.len(), 1);
        assert_eq!(report.remapped[0], ("linear_0/weight".to_string(), "encoder/weight".to_string()));
        assert!(report.is_complete());
    }

    #[test]
    fn test_migrate_with_buffers() {
        let old_params = vec![
            ("linear_0/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let old_buffers = vec![
            ("bn_0/running_mean".to_string(), Buffer::new(
                Tensor::zeros(&[8], crate::tensor::test_opts()).unwrap(), "running_mean")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &old_buffers);

        let new_params = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let new_buffers = vec![
            ("norm/running_mean".to_string(), Buffer::new(
                Tensor::zeros(&[8], crate::tensor::test_opts()).unwrap(), "running_mean")),
        ];

        let mut out = Vec::new();
        let report = migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &new_buffers,
        ).unwrap();

        assert_eq!(report.remapped.len(), 2);
        assert!(report.is_complete());

        // Verify migrated checkpoint loads with new names
        let vp = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let vb = vec![
            ("norm/running_mean".to_string(), Buffer::new(
                Tensor::zeros(&[8], crate::tensor::test_opts()).unwrap(), "running_mean")),
        ];
        let mut cursor = std::io::Cursor::new(&out);
        let load_report = load_checkpoint(&mut cursor, &vp, &vb, None).unwrap();
        assert_eq!(load_report.loaded.len(), 2);
    }

    #[test]
    fn test_migrate_dropped_and_missing() {
        let old_params = vec![
            ("old/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("removed/weight".to_string(), Parameter::new(
                Tensor::randn(&[16, 16], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &[]);

        // New model: one matching shape, one entirely new
        let new_params = vec![
            ("new/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("added/weight".to_string(), Parameter::new(
                Tensor::randn(&[32, 32], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let mut out = Vec::new();
        let report = migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &[],
        ).unwrap();

        assert_eq!(report.remapped.len(), 1);
        assert_eq!(report.dropped, vec!["removed/weight"]);
        assert_eq!(report.missing, vec!["added/weight"]);
        assert!(!report.is_complete());
    }

    #[test]
    fn test_migrate_positional_disambiguation() {
        // Two params with identical shape — must match by position
        let old_params = vec![
            ("linear_0/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
            ("linear_1/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &[]);

        let new_params = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
            ("decoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let mut out = Vec::new();
        let report = migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &[],
        ).unwrap();

        assert_eq!(report.remapped.len(), 2);
        // Positional: first old → first new, second old → second new
        assert_eq!(report.remapped[0].0, "linear_0/weight");
        assert_eq!(report.remapped[0].1, "encoder/weight");
        assert_eq!(report.remapped[1].0, "linear_1/weight");
        assert_eq!(report.remapped[1].1, "decoder/weight");

        // Verify correct data assignment
        let vp = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
            ("decoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&out);
        load_checkpoint(&mut cursor, &vp, &[], None).unwrap();

        // encoder/weight should have linear_0's data, decoder/weight should have linear_1's data
        let enc_data = vp[0].1.variable.data().to_f32_vec().unwrap();
        let dec_data = vp[1].1.variable.data().to_f32_vec().unwrap();
        let old_0 = old_params[0].1.variable.data().to_f32_vec().unwrap();
        let old_1 = old_params[1].1.variable.data().to_f32_vec().unwrap();
        assert_eq!(enc_data, old_0);
        assert_eq!(dec_data, old_1);
    }

    #[test]
    fn test_migrate_v1_writes_v2() {
        let old_params = vec![
            ("x/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &[]);

        // Confirm source is v1
        let mut peek = std::io::Cursor::new(&ckpt);
        let mut magic = [0u8; 4];
        std::io::Read::read_exact(&mut peek, &mut magic).unwrap();
        let mut vbuf = [0u8; 4];
        std::io::Read::read_exact(&mut peek, &mut vbuf).unwrap();
        assert_eq!(u32::from_le_bytes(vbuf), 1);

        let new_params = vec![
            ("y/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let mut out = Vec::new();
        migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &[],
        ).unwrap();

        // Confirm output is v2
        let mut peek2 = std::io::Cursor::new(&out);
        std::io::Read::read_exact(&mut peek2, &mut magic).unwrap();
        assert_eq!(&magic, b"FDLC");
        std::io::Read::read_exact(&mut peek2, &mut vbuf).unwrap();
        assert_eq!(u32::from_le_bytes(vbuf), VERSION); // should be 2
    }

    #[test]
    fn test_migrate_file_roundtrip() {
        let old_params = vec![
            ("old/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let dir = std::env::temp_dir();
        let src = dir.join("test_migrate_src.fdl");
        let dst = dir.join("test_migrate_dst.fdl");

        // Write v1 checkpoint to file
        {
            let f = std::fs::File::create(&src).unwrap();
            let mut w = std::io::BufWriter::new(f);
            save_checkpoint_versioned(&mut w, 1, &old_params, &[]);
        }

        let new_params = vec![
            ("new/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let report = migrate_checkpoint_file(
            src.to_str().unwrap(),
            dst.to_str().unwrap(),
            &new_params, &[],
        ).unwrap();
        assert_eq!(report.remapped.len(), 1);
        assert!(report.is_complete());

        // Load migrated file
        let vp = vec![
            ("new/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let load_report = load_checkpoint_file(
            dst.to_str().unwrap(), &vp, &[], None,
        ).unwrap();
        assert_eq!(load_report.loaded.len(), 1);

        // Verify data preserved
        let expected = old_params[0].1.variable.data().to_f32_vec().unwrap();
        let got = vp[0].1.variable.data().to_f32_vec().unwrap();
        assert_eq!(expected, got);

        std::fs::remove_file(src).ok();
        std::fs::remove_file(dst).ok();
    }

    #[test]
    fn test_migrate_display() {
        let report = MigrateReport {
            unchanged: vec!["shared/weight".to_string()],
            remapped: vec![("old/bias".to_string(), "new/bias".to_string())],
            dropped: vec!["removed/weight".to_string()],
            missing: vec!["added/weight".to_string()],
        };
        let text = format!("{}", report);
        assert!(text.contains("unchanged (1)"));
        assert!(text.contains("remapped (1)"));
        assert!(text.contains("old/bias -> new/bias"));
        assert!(text.contains("dropped (1)"));
        assert!(text.contains("missing (1)"));
    }

    #[test]
    fn test_checkpoint_version_peek() {
        let params = make_named_params(&[(4, 8)]);
        let dir = std::env::temp_dir();
        let path = dir.join("test_version_peek.fdl");
        save_checkpoint_file(path.to_str().unwrap(), &params, &[], None).unwrap();

        let v = checkpoint_version(path.to_str().unwrap()).unwrap();
        assert_eq!(v, VERSION);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_accepts_v1() {
        // v1 checkpoints must still load in v2 builds
        let params = vec![
            ("x/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &params, &[]);

        let load_params = vec![
            ("x/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&ckpt);
        let report = load_checkpoint(&mut cursor, &load_params, &[], None).unwrap();
        assert_eq!(report.loaded.len(), 1);

        let expected = params[0].1.variable.data().to_f32_vec().unwrap();
        let got = load_params[0].1.variable.data().to_f32_vec().unwrap();
        assert_eq!(expected, got);
    }

    // --- Edge case / corruption tests ---

    #[test]
    fn test_truncated_checkpoint_header_only() {
        // Write valid header but truncate before any entry data
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&[0u8; HASH_LEN]);
        // Claim 5 entries, but provide none
        buf.extend_from_slice(&5u32.to_le_bytes());

        let params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err(), "truncated checkpoint should return Err, not panic");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("io:"), "should be an IO error: {}", msg);
    }

    #[test]
    fn test_truncated_checkpoint_mid_entry() {
        // Save a valid checkpoint, then truncate in the middle of the first entry
        let params = make_named_params(&[(4, 8)]);
        let mut full = Vec::new();
        save_checkpoint(&mut full, &params, &[], None).unwrap();

        // Header = 4 (magic) + 4 (version) + 32 (hash) + 4 (count) = 44
        // Truncate partway through the first entry (e.g., keep only 50 bytes)
        let truncated = full[..50.min(full.len())].to_vec();

        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&truncated);
        let result = load_checkpoint(&mut cursor, &load_params, &[], None);
        assert!(result.is_err(), "truncated mid-entry should return Err");
    }

    #[test]
    fn test_empty_file() {
        // Zero bytes: read_exact for magic should fail
        let buf: Vec<u8> = Vec::new();
        let params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err(), "empty file should return Err");
    }

    #[test]
    fn test_invalid_magic_bytes() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JUNK"); // wrong magic
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&[0u8; HASH_LEN]);
        buf.extend_from_slice(&0u32.to_le_bytes());

        let params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("bad magic"), "error should mention bad magic: {}", msg);
    }

    #[test]
    fn test_invalid_magic_checkpoint_version() {
        // checkpoint_version() should also reject bad magic
        let dir = std::env::temp_dir();
        let path = dir.join("test_bad_magic_version.fdl");
        std::fs::write(&path, b"NOT_FDLC_data").unwrap();

        let result = checkpoint_version(path.to_str().unwrap());
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("bad magic"), "error: {}", msg);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_unsupported_version_high() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&99u32.to_le_bytes()); // version 99
        buf.extend_from_slice(&[0u8; HASH_LEN]);
        buf.extend_from_slice(&0u32.to_le_bytes());

        let params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unsupported checkpoint version"), "error: {}", msg);
        assert!(msg.contains("99"), "should mention version 99: {}", msg);
    }

    #[test]
    fn test_unsupported_version_zero() {
        // Version 0 is also rejected (valid range is 1..=MAX_VERSION)
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&0u32.to_le_bytes()); // version 0
        buf.extend_from_slice(&[0u8; HASH_LEN]);
        buf.extend_from_slice(&0u32.to_le_bytes());

        let params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unsupported checkpoint version"), "error: {}", msg);
    }

    #[test]
    fn test_hash_mismatch_both_nonzero() {
        // Both file and expected have nonzero hashes that differ
        let params = make_named_params(&[(4, 8)]);
        let hash_a = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let hash_b = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], Some(hash_a)).unwrap();

        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &load_params, &[], Some(hash_b));
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("architecture mismatch"), "error: {}", msg);
        // Error message should include both hashes for diagnostics
        assert!(msg.contains(hash_b), "should show expected hash: {}", msg);
    }

    #[test]
    fn test_zero_entries_empty_model() {
        // Save a checkpoint with no parameters and no buffers
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &[], &[], None).unwrap();

        // Load into an empty model
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &[], &[], None).unwrap();
        assert!(report.loaded.is_empty());
        assert!(report.skipped.is_empty());
        assert!(report.missing.is_empty());
    }

    #[test]
    fn test_zero_entries_nonempty_model() {
        // Save empty checkpoint, load into model that expects params
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &[], &[], None).unwrap();

        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &load_params, &[], None).unwrap();
        assert!(report.loaded.is_empty());
        assert!(report.skipped.is_empty());
        assert_eq!(report.missing.len(), 1, "model param should be reported as missing");
    }

    #[test]
    fn test_shape_mismatch_transposed() {
        // Save [4, 8], try to load into [8, 4] (transposed, same numel)
        let params = vec![
            ("layer/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], None).unwrap();

        let wrong_params = vec![
            ("layer/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 4], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &wrong_params, &[], None);
        assert!(result.is_err(), "transposed shape should be a mismatch error");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("shape mismatch"), "error: {}", msg);
        assert!(msg.contains("[4, 8]"), "should show checkpoint shape: {}", msg);
        assert!(msg.contains("[8, 4]"), "should show model shape: {}", msg);
    }

    #[test]
    fn test_dtype_mismatch_auto_cast() {
        // Save as f32, load into f64 parameter. The code does to_dtype() automatically.
        let f32_param = vec![
            ("layer/weight".to_string(), Parameter::new(
                Tensor::ones(&[2, 3], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &f32_param, &[], None).unwrap();

        // Create f64 parameter with same shape
        let f64_param = vec![
            ("layer/weight".to_string(), Parameter::new(
                Tensor::zeros(&[2, 3], TensorOptions {
                    dtype: DType::Float64,
                    device: crate::tensor::test_device(),
                }).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &f64_param, &[], None).unwrap();
        assert_eq!(report.loaded.len(), 1, "dtype auto-cast should succeed");

        // Verify the loaded data is correct and in f64
        let loaded = f64_param[0].1.variable.data();
        assert_eq!(loaded.dtype(), DType::Float64);
        let vals = loaded.to_f64_vec().unwrap();
        for v in vals {
            assert!((v - 1.0).abs() < 1e-6, "expected ~1.0, got {}", v);
        }
    }

    #[test]
    fn test_dtype_mismatch_buffer_auto_cast() {
        // Same auto-cast test for buffers
        let f32_buffers = vec![
            ("norm/running_mean".to_string(), Buffer::new(
                Tensor::ones(&[8], crate::tensor::test_opts()).unwrap(), "running_mean")),
        ];
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &[], &f32_buffers, None).unwrap();

        let f64_buffers = vec![
            ("norm/running_mean".to_string(), Buffer::new(
                Tensor::zeros(&[8], TensorOptions {
                    dtype: DType::Float64,
                    device: crate::tensor::test_device(),
                }).unwrap(), "running_mean")),
        ];
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &[], &f64_buffers, None).unwrap();
        assert_eq!(report.loaded.len(), 1);
        assert_eq!(f64_buffers[0].1.get().dtype(), DType::Float64);
        let vals = f64_buffers[0].1.get().to_f64_vec().unwrap();
        for v in vals {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compressed_roundtrip_with_hash() {
        // Test gz compression with structural hash validation
        let params = make_named_params(&[(8, 16)]);
        let hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

        let dir = std::env::temp_dir();
        let gz_path = dir.join("test_ckpt_hash_gz.fdl.gz");
        let path_str = gz_path.to_str().unwrap();

        save_checkpoint_file(path_str, &params, &[], Some(hash)).unwrap();

        // Load with matching hash
        let load_params = make_named_params(&[(8, 16)]);
        let report = load_checkpoint_file(path_str, &load_params, &[], Some(hash)).unwrap();
        assert_eq!(report.loaded.len(), 1);

        // Load with wrong hash should fail
        let bad_hash = "1111111111111111111111111111111111111111111111111111111111111111";
        let load_params2 = make_named_params(&[(8, 16)]);
        let result = load_checkpoint_file(path_str, &load_params2, &[], Some(bad_hash));
        assert!(result.is_err());

        std::fs::remove_file(gz_path).ok();
    }

    #[test]
    fn test_corrupted_gz_file() {
        // Write valid gz header then garbage: should produce an error
        let dir = std::env::temp_dir();
        let path = dir.join("test_corrupt.fdl.gz");
        // Write some garbage that is not valid gzip
        std::fs::write(&path, b"\x1f\x8b\x08\x00GARBAGE_NOT_VALID_GZ").unwrap();

        let params = make_named_params(&[(4, 8)]);
        let result = load_checkpoint_file(path.to_str().unwrap(), &params, &[], None);
        assert!(result.is_err(), "corrupted gz should return Err");

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_unknown_dtype_tag() {
        // Manually craft a checkpoint with an invalid dtype tag byte
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&[0u8; HASH_LEN]);
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 entry

        // Entry name
        let name = b"layer/weight";
        buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
        buf.extend_from_slice(name);

        // ndim = 1, shape = [4]
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4i64.to_le_bytes());

        // Invalid dtype tag (255)
        buf.push(255);

        // byte_count = 16 (4 * f32), then dummy data
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 16]);

        let params = vec![
            ("layer/weight".to_string(), Parameter::new(
                Tensor::zeros(&[4], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unknown dtype tag"), "error: {}", msg);
    }

    #[test]
    fn test_checkpoint_keys_peeks_names_without_loading_data() {
        let params = vec![
            (
                "encoder/layer/weight".to_string(),
                Parameter::new(
                    Tensor::ones(&[4, 8], crate::tensor::test_opts()).unwrap(),
                    "weight",
                ),
            ),
            (
                "pooler/dense/weight".to_string(),
                Parameter::new(
                    Tensor::ones(&[8, 8], crate::tensor::test_opts()).unwrap(),
                    "weight",
                ),
            ),
        ];
        let buffers = vec![(
            "encoder/layer/running_mean".to_string(),
            Buffer::new(
                Tensor::zeros(&[8], crate::tensor::test_opts()).unwrap(),
                "running_mean",
            ),
        )];

        let dir = std::env::temp_dir();
        let path = dir.join("test_checkpoint_keys_peek.fdl");
        let path_str = path.to_str().unwrap();

        save_checkpoint_file(path_str, &params, &buffers, None).unwrap();
        let keys = checkpoint_keys(path_str).unwrap();
        assert_eq!(
            keys,
            vec![
                "encoder/layer/weight".to_string(),
                "pooler/dense/weight".to_string(),
                "encoder/layer/running_mean".to_string(),
            ],
            "params first then buffers, in declaration order",
        );

        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_checkpoint_keys_handles_gzip() {
        let params = vec![(
            "x/w".to_string(),
            Parameter::new(
                Tensor::ones(&[2, 2], crate::tensor::test_opts()).unwrap(),
                "w",
            ),
        )];
        let dir = std::env::temp_dir();
        let path = dir.join("test_checkpoint_keys_gz.fdl.gz");
        let path_str = path.to_str().unwrap();

        save_checkpoint_file(path_str, &params, &[], None).unwrap();
        let keys = checkpoint_keys(path_str).unwrap();
        assert_eq!(keys, vec!["x/w".to_string()]);

        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_checkpoint_keys_rejects_bad_magic() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_checkpoint_keys_bad.fdl");
        std::fs::write(&path, b"NOPEnotacheckpoint").unwrap();
        let err = checkpoint_keys(path.to_str().unwrap()).unwrap_err();
        assert!(format!("{err}").contains("bad magic"), "got: {err}");
        std::fs::remove_file(path).ok();
    }

    // A successful save commits via `<path>.tmp` + atomic rename, so the final
    // file loads AND no stale `.tmp` sibling is left behind. This is the M1
    // guarantee the NCCL elected-rank consensus checkpoint depends on: a crash
    // between `File::create` and the last byte can only ever leave a `.tmp`,
    // never a torn `<stem>.fdl` at the canonical path.
    #[test]
    fn test_atomic_save_commits_and_leaves_no_tmp() {
        let params = make_named_params(&[(8, 4)]);
        let buffers = make_named_buffers(&[4]);

        let dir = std::env::temp_dir();
        // Unique name so the parallel test harness cannot collide on the path.
        let path = dir.join("test_atomic_save_m1.fdl");
        let path_str = path.to_str().unwrap();
        let tmp = format!("{path_str}.tmp");
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&tmp).ok();

        save_checkpoint_file(path_str, &params, &buffers, None).unwrap();

        // Final file exists and round-trips.
        assert!(path.exists(), "committed checkpoint missing at {path_str}");
        let load_params = make_named_params(&[(8, 4)]);
        let load_buffers = make_named_buffers(&[4]);
        let report =
            load_checkpoint_file(path_str, &load_params, &load_buffers, None).unwrap();
        assert_eq!(report.loaded.len(), 2); // 1 param + 1 buffer

        // The tmp scratch file was renamed away, not left as litter.
        assert!(
            !std::path::Path::new(&tmp).exists(),
            "stale .tmp left behind at {tmp}",
        );

        std::fs::remove_file(&path).ok();
    }

    // gzip is chosen from the FINAL `.gz` extension, not the transient `.tmp`
    // name — the atomic-write refactor must not silently drop compression.
    #[test]
    fn test_atomic_save_gz_still_compresses() {
        let params = make_named_params(&[(16, 32), (32, 8)]);

        let dir = std::env::temp_dir();
        let gz_path = dir.join("test_atomic_save_m1.fdl.gz");
        let gz = gz_path.to_str().unwrap();
        std::fs::remove_file(&gz_path).ok();

        save_checkpoint_file(gz, &params, &[], None).unwrap();

        // A gzip stream begins with the 0x1f 0x8b magic; an uncompressed
        // `.fdl` would begin with `FDLC`. This asserts the `.gz` path fired
        // even though the bytes were first written to `<path>.tmp`.
        let bytes = std::fs::read(gz).unwrap();
        assert!(
            bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b,
            "expected gzip magic, got {:?}",
            &bytes[..bytes.len().min(4)],
        );
        assert!(
            !std::path::Path::new(&format!("{gz}.tmp")).exists(),
            "stale .tmp left behind",
        );

        std::fs::remove_file(&gz_path).ok();
    }

    // NN2 regression: the f16/bf16/i32 branch of `tensor_from_raw_bytes` used to
    // hand `raw.as_ptr()` straight to `flodl_from_blob`, which reads
    // numel × element_size bytes trusting the caller — a short buffer from a
    // truncated/corrupt checkpoint drove an out-of-bounds read. Routing through
    // `Tensor::from_blob` homes the length check, so it errors instead.
    #[test]
    fn test_tensor_from_raw_bytes_rejects_truncated_low_precision() {
        // shape [4] at f16 needs 4 × 2 = 8 bytes; hand it 4.
        assert!(
            tensor_from_raw_bytes(&[0u8; 4], &[4], DType::Float16).is_err(),
            "truncated f16 buffer must be rejected, not read OOB",
        );
        // i32 mirrors: shape [4] needs 16 bytes; hand it 12.
        assert!(
            tensor_from_raw_bytes(&[0u8; 12], &[4], DType::Int32).is_err(),
            "truncated i32 buffer must be rejected",
        );
        // Exact-length payload still round-trips (routing preserved).
        let t = tensor_from_raw_bytes(&[0u8; 8], &[4], DType::Float16).unwrap();
        assert_eq!(t.shape(), vec![4]);
        assert_eq!(t.dtype(), DType::Float16);
    }
