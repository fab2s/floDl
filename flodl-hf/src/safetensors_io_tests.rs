use super::*;

fn actual_map(entries: &[(&str, &[i64])]) -> HashMap<String, Vec<i64>> {
    entries
        .iter()
        .map(|(k, s)| ((*k).to_string(), s.to_vec()))
        .collect()
}

#[test]
fn keys_have_pooler_detects_slash_separator() {
    let keys = vec![
        "encoder/layer/0/attention/query/weight".to_string(),
        "pooler/dense/weight".to_string(),
        "pooler/dense/bias".to_string(),
    ];
    assert!(keys_have_pooler(&keys));
}

#[test]
fn keys_have_pooler_detects_dot_separator() {
    let keys = vec![
        "encoder.layer.0.attention.query.weight".to_string(),
        "pooler.dense.weight".to_string(),
    ];
    assert!(keys_have_pooler(&keys));
}

#[test]
fn keys_have_pooler_returns_false_for_encoder_only() {
    let keys = vec![
        "encoder/layer/0/attention/query/weight".to_string(),
        "encoder/layer/0/attention/query/bias".to_string(),
    ];
    assert!(!keys_have_pooler(&keys));
}

#[test]
fn keys_have_pooler_does_not_match_substrings() {
    // A key containing "pooler" mid-string must not false-positive.
    let keys = vec!["encoder/some_pooler_thing/weight".to_string()];
    assert!(!keys_have_pooler(&keys));
}

#[test]
fn weights_have_pooler_detects_bert_style() {
    let bytes = serialize_entries(&[
        (
            "bert.embeddings.word_embeddings.weight",
            Dtype::F32,
            vec![2, 4],
            f32_le_bytes(&[0.0; 8]),
        ),
        (
            "bert.pooler.dense.weight",
            Dtype::F32,
            vec![4, 4],
            f32_le_bytes(&[0.0; 16]),
        ),
        (
            "bert.pooler.dense.bias",
            Dtype::F32,
            vec![4],
            f32_le_bytes(&[0.0; 4]),
        ),
    ]);
    assert!(weights_have_pooler(&bytes).unwrap());
}

#[test]
fn weights_have_pooler_detects_albert_style_flat() {
    // ALBERT ships a flat `pooler.{weight,bias}` (bare nn.Linear,
    // no `.dense` wrapper). The detector matches both shapes.
    let bytes = serialize_entries(&[
        (
            "albert.embeddings.word_embeddings.weight",
            Dtype::F32,
            vec![2, 4],
            f32_le_bytes(&[0.0; 8]),
        ),
        (
            "albert.pooler.weight",
            Dtype::F32,
            vec![4, 4],
            f32_le_bytes(&[0.0; 16]),
        ),
        (
            "albert.pooler.bias",
            Dtype::F32,
            vec![4],
            f32_le_bytes(&[0.0; 4]),
        ),
    ]);
    assert!(weights_have_pooler(&bytes).unwrap());
}

#[test]
fn weights_have_pooler_returns_false_for_encoder_only() {
    // RoBERTa-style: encoder weights only, no pooler. Mirrors
    // `FacebookAI/roberta-base` Hub repo which drops the pooler with the NSP
    // objective.
    let bytes = serialize_entries(&[
        (
            "roberta.embeddings.word_embeddings.weight",
            Dtype::F32,
            vec![2, 4],
            f32_le_bytes(&[0.0; 8]),
        ),
        (
            "roberta.encoder.layer.0.attention.self.query.weight",
            Dtype::F32,
            vec![4, 4],
            f32_le_bytes(&[0.0; 16]),
        ),
    ]);
    assert!(!weights_have_pooler(&bytes).unwrap());
}

#[test]
fn weights_have_pooler_errors_on_invalid_safetensors() {
    // Garbage bytes must surface as a parse error rather than panic
    // or return a meaningless boolean.
    let err = weights_have_pooler(b"not a safetensors blob")
        .unwrap_err()
        .to_string();
    assert!(err.contains("safetensors parse error"), "got: {err}");
}

#[test]
fn bert_legacy_layernorm_rename_rewrites_gamma_and_beta() {
    assert_eq!(
        bert_legacy_layernorm_rename("bert.embeddings.LayerNorm.gamma"),
        "bert.embeddings.LayerNorm.weight",
    );
    assert_eq!(
        bert_legacy_layernorm_rename("bert.embeddings.LayerNorm.beta"),
        "bert.embeddings.LayerNorm.bias",
    );
}

#[test]
fn bert_legacy_layernorm_rename_passthrough_for_modern_keys() {
    // Modern HF saves already use weight/bias — must not be mangled.
    let key = "bert.embeddings.LayerNorm.weight";
    assert_eq!(bert_legacy_layernorm_rename(key), key);
}

#[test]
fn bert_legacy_layernorm_rename_does_not_touch_mlm_alias() {
    // Distinct from `bert_legacy_key_rename`: the LayerNorm-only
    // helper leaves the MLM decoder-bias alias alone.
    let k = "cls.predictions.bias";
    assert_eq!(bert_legacy_layernorm_rename(k), k);
}

#[test]
fn hf_canonical_save_key_inverts_mlm_decoder_bias_alias() {
    assert_eq!(
        hf_canonical_save_key("cls.predictions.decoder.bias"),
        "cls.predictions.bias",
    );
    assert_eq!(
        hf_canonical_save_key("lm_head.decoder.bias"),
        "lm_head.bias",
    );
}

#[test]
fn hf_canonical_save_key_passthrough_for_unrelated_keys() {
    // Only the two MLM tied-bias keys are rewritten — every other
    // key passes through unchanged. LayerNorm modern names stay
    // modern (legacy gamma/beta are not the inverse here; flodl
    // saves modern names directly).
    for k in [
        "bert.embeddings.word_embeddings.weight",
        "bert.embeddings.LayerNorm.weight",
        "bert.encoder.layer.0.attention.self.query.bias",
    ] {
        assert_eq!(hf_canonical_save_key(k), k);
    }
}

#[test]
fn all_keys_match_returns_ok() {
    let expected = vec![
        ExpectedParam {
            key: "bert.embeddings.word_embeddings.weight".into(),
            shape: vec![30522, 768],
        },
        ExpectedParam {
            key: "bert.pooler.dense.bias".into(),
            shape: vec![768],
        },
    ];
    let actual = actual_map(&[
        ("bert.embeddings.word_embeddings.weight", &[30522, 768]),
        ("bert.pooler.dense.bias", &[768]),
    ]);
    let v = validate_keys(&expected, &actual);
    assert!(v.is_ok());
    assert!(v.into_result().is_ok());
}

#[test]
fn missing_key_is_reported() {
    let expected = vec![ExpectedParam {
        key: "bert.pooler.dense.weight".into(),
        shape: vec![768, 768],
    }];
    let actual = actual_map(&[]);
    let v = validate_keys(&expected, &actual);
    assert_eq!(v.missing, vec!["bert.pooler.dense.weight"]);
    assert!(v.unused.is_empty());
    assert!(v.shape_mismatches.is_empty());
}

#[test]
fn unused_checkpoint_key_is_reported() {
    let expected: Vec<ExpectedParam> = Vec::new();
    let actual = actual_map(&[("bert.something.extra", &[4])]);
    let v = validate_keys(&expected, &actual);
    assert_eq!(v.unused, vec!["bert.something.extra"]);
    assert!(v.missing.is_empty());
    assert!(v.shape_mismatches.is_empty());
}

#[test]
fn shape_mismatch_is_reported() {
    // Vocab size drift: checkpoint has 30522 tokens, model expects 50257.
    let expected = vec![ExpectedParam {
        key: "bert.embeddings.word_embeddings.weight".into(),
        shape: vec![50257, 768],
    }];
    let actual = actual_map(&[("bert.embeddings.word_embeddings.weight", &[30522, 768])]);
    let v = validate_keys(&expected, &actual);
    assert!(v.missing.is_empty());
    assert!(v.unused.is_empty());
    assert_eq!(v.shape_mismatches.len(), 1);
    assert_eq!(
        v.shape_mismatches[0].key,
        "bert.embeddings.word_embeddings.weight"
    );
    assert_eq!(v.shape_mismatches[0].expected, vec![50257, 768]);
    assert_eq!(v.shape_mismatches[0].found, vec![30522, 768]);
}

#[test]
fn typo_queri_vs_query_reports_both_missing_and_unused() {
    // The motivating bug: author typed "queri" in a tag, checkpoint has "query".
    // Validator must surface both sides so the typo is unambiguous.
    let expected = vec![ExpectedParam {
        key: "bert.encoder.layer.0.attention.self.queri.weight".into(),
        shape: vec![768, 768],
    }];
    let actual = actual_map(&[(
        "bert.encoder.layer.0.attention.self.query.weight",
        &[768, 768],
    )]);
    let v = validate_keys(&expected, &actual);
    assert_eq!(
        v.missing,
        vec!["bert.encoder.layer.0.attention.self.queri.weight"]
    );
    assert_eq!(
        v.unused,
        vec!["bert.encoder.layer.0.attention.self.query.weight"]
    );
}

#[test]
fn mixed_failures_accumulate() {
    let expected = vec![
        ExpectedParam {
            key: "ok.weight".into(),
            shape: vec![4],
        },
        ExpectedParam {
            key: "missing.weight".into(),
            shape: vec![8],
        },
        ExpectedParam {
            key: "wrong_shape.weight".into(),
            shape: vec![16],
        },
    ];
    let actual = actual_map(&[
        ("ok.weight", &[4]),
        ("wrong_shape.weight", &[32]),
        ("extra.weight", &[1]),
    ]);
    let v = validate_keys(&expected, &actual);
    assert_eq!(v.missing, vec!["missing.weight"]);
    assert_eq!(v.unused, vec!["extra.weight"]);
    assert_eq!(v.shape_mismatches.len(), 1);
    assert_eq!(v.shape_mismatches[0].key, "wrong_shape.weight");
}

#[test]
fn into_result_error_message_lists_every_bucket() {
    let expected = vec![
        ExpectedParam {
            key: "m.w".into(),
            shape: vec![2],
        },
        ExpectedParam {
            key: "sm.w".into(),
            shape: vec![3],
        },
    ];
    let actual = actual_map(&[("sm.w", &[4]), ("extra.w", &[1])]);
    let v = validate_keys(&expected, &actual);
    let err = v.into_result().unwrap_err().to_string();
    assert!(
        err.contains("1 missing key"),
        "missing bucket not in msg: {err}"
    );
    assert!(
        err.contains("1 unused key"),
        "unused bucket not in msg: {err}"
    );
    assert!(
        err.contains("1 shape mismatch"),
        "shape bucket not in msg: {err}"
    );
    assert!(err.contains("m.w"));
    assert!(err.contains("extra.w"));
    assert!(err.contains("sm.w"));
    assert!(err.contains("[3]"));
    assert!(err.contains("[4]"));
}

#[test]
fn output_is_sorted_for_stable_messages() {
    let expected = vec![
        ExpectedParam {
            key: "z.w".into(),
            shape: vec![1],
        },
        ExpectedParam {
            key: "a.w".into(),
            shape: vec![1],
        },
    ];
    let actual = actual_map(&[("m.w", &[1]), ("c.w", &[1])]);
    let v = validate_keys(&expected, &actual);
    assert_eq!(v.missing, vec!["a.w", "z.w"]);
    assert_eq!(v.unused, vec!["c.w", "m.w"]);
}

#[test]
fn empty_everywhere_is_ok() {
    let v = validate_keys(&[], &HashMap::new());
    assert!(v.is_ok());
    assert!(v.missing.is_empty());
    assert!(v.unused.is_empty());
    assert!(v.shape_mismatches.is_empty());
    assert!(v.into_result().is_ok());
}

#[test]
fn into_result_truncates_long_missing_list() {
    // 25 missing keys — "... and N more" tail should appear for any
    // bucket longer than the 20-entry cap.
    let expected: Vec<ExpectedParam> = (0..25)
        .map(|i| ExpectedParam {
            key: format!("key.{i:02}"),
            shape: vec![1],
        })
        .collect();
    let v = validate_keys(&expected, &HashMap::new());
    assert_eq!(v.missing.len(), 25);
    let err = v.into_result().unwrap_err().to_string();
    assert!(
        err.contains("25 missing key"),
        "header must show full count: {err}"
    );
    assert!(
        err.contains("... and 5 more"),
        "truncation tail must show remaining count: {err}"
    );
    // First 20 keys listed; 21st onwards must not appear verbatim.
    assert!(err.contains("key.00"));
    assert!(err.contains("key.19"));
    assert!(!err.contains("key.20"));
}

/// Helper: raw f32 bytes (little-endian) from a flat slice.
fn f32_le_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Helper: serialise a list of (name, dtype, shape, byte-data) into a
/// safetensors byte buffer. Lifetimes stay simple because the byte
/// buffers are owned by the caller and outlive the `TensorView`s.
fn serialize_entries(entries: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    let views: HashMap<String, TensorView<'_>> = entries
        .iter()
        .map(|(n, d, s, b)| (n.to_string(), TensorView::new(*d, s.clone(), b).unwrap()))
        .collect();
    safetensors::serialize(&views, None).unwrap()
}

/// End-to-end: build a tagged Linear graph, pin its parameters to
/// deterministic values, serialise them as f32 safetensors, then load
/// the bytes into a second fresh graph. The second graph must end up
/// bit-exact on both `weight` and `bias`. Guards the main load path
/// plus the HF key conversion (slash → dot) inside the loader.
#[test]
fn load_safetensors_f32_roundtrip() {
    use flodl::{FlowBuilder, Linear, Module, Variable};

    let in_dim = 3_i64;
    let out_dim = 2_i64;
    let dev = Device::CPU;

    // Source graph: tag it and overwrite the random init with known values.
    let src_graph = FlowBuilder::new()
        .through(Linear::on_device(in_dim, out_dim, dev).unwrap())
        .tag("my.linear")
        .build()
        .unwrap();
    let src_weight: Vec<f32> = (0..(in_dim * out_dim) as usize)
        .map(|i| 1.0 + i as f32 * 0.25)
        .collect();
    let src_bias: Vec<f32> = (0..out_dim as usize).map(|i| -0.5 + i as f32).collect();
    for (k, p) in src_graph.named_parameters() {
        let hf = hf_key_from_flodl_key(&k);
        let t = match hf.as_str() {
            "my.linear.weight" => Tensor::from_f32(&src_weight, &[out_dim, in_dim], dev).unwrap(),
            "my.linear.bias" => Tensor::from_f32(&src_bias, &[out_dim], dev).unwrap(),
            other => panic!("unexpected key {other}"),
        };
        p.variable.set_data(t);
    }

    // Serialise source params as f32 safetensors.
    let w_bytes = f32_le_bytes(&src_weight);
    let b_bytes = f32_le_bytes(&src_bias);
    let bytes = serialize_entries(&[
        (
            "my.linear.weight",
            Dtype::F32,
            vec![out_dim as usize, in_dim as usize],
            w_bytes,
        ),
        (
            "my.linear.bias",
            Dtype::F32,
            vec![out_dim as usize],
            b_bytes,
        ),
    ]);

    // Destination graph: fresh, then load.
    let dst_graph = FlowBuilder::new()
        .through(Linear::on_device(in_dim, out_dim, dev).unwrap())
        .tag("my.linear")
        .build()
        .unwrap();
    load_safetensors_into_graph(&dst_graph, &bytes).unwrap();

    // Assert bit-exactness on both params (f32 → f32 is lossless).
    let mut dst_weight: Option<Vec<f32>> = None;
    let mut dst_bias: Option<Vec<f32>> = None;
    for (k, p) in dst_graph.named_parameters() {
        let hf = hf_key_from_flodl_key(&k);
        let data = p.variable.data().to_f32_vec().unwrap();
        match hf.as_str() {
            "my.linear.weight" => dst_weight = Some(data),
            "my.linear.bias" => dst_bias = Some(data),
            other => panic!("unexpected key {other}"),
        }
    }
    assert_eq!(dst_weight.unwrap(), src_weight);
    assert_eq!(dst_bias.unwrap(), src_bias);

    // Sanity: dst params are still alive as Variables (load shouldn't
    // replace them — just refill storage).
    let _keep_alive: Vec<Variable> = dst_graph
        .parameters()
        .into_iter()
        .map(|p| p.variable)
        .collect();
}

/// File-path variant: same roundtrip but through disk, to exercise the
/// file reader and the path-in-error-message behaviour indirectly.
#[test]
fn load_safetensors_file_roundtrip() {
    use flodl::{FlowBuilder, Linear};
    use std::io::Write;

    let dev = Device::CPU;
    let graph = FlowBuilder::new()
        .through(Linear::on_device(2, 1, dev).unwrap())
        .tag("m")
        .build()
        .unwrap();
    let w = vec![0.25_f32, 0.5];
    let b = vec![1.5_f32];
    for (k, p) in graph.named_parameters() {
        let hf = hf_key_from_flodl_key(&k);
        let t = match hf.as_str() {
            "m.weight" => Tensor::from_f32(&w, &[1, 2], dev).unwrap(),
            "m.bias" => Tensor::from_f32(&b, &[1], dev).unwrap(),
            other => panic!("unexpected {other}"),
        };
        p.variable.set_data(t);
    }
    let bytes = serialize_entries(&[
        ("m.weight", Dtype::F32, vec![1, 2], f32_le_bytes(&w)),
        ("m.bias", Dtype::F32, vec![1], f32_le_bytes(&b)),
    ]);

    let path =
        std::env::temp_dir().join(format!("flodl_hf_test_{}.safetensors", std::process::id()));
    std::fs::File::create(&path)
        .unwrap()
        .write_all(&bytes)
        .unwrap();

    let fresh = FlowBuilder::new()
        .through(Linear::on_device(2, 1, dev).unwrap())
        .tag("m")
        .build()
        .unwrap();
    load_safetensors_file_into_graph(&fresh, &path).unwrap();

    // Cleanup first so a failed assert doesn't leak the tmp file.
    let _ = std::fs::remove_file(&path);

    for (k, p) in fresh.named_parameters() {
        let hf = hf_key_from_flodl_key(&k);
        let data = p.variable.data().to_f32_vec().unwrap();
        match hf.as_str() {
            "m.weight" => assert_eq!(data, w),
            "m.bias" => assert_eq!(data, b),
            other => panic!("unexpected {other}"),
        }
    }
}

/// BF16 checkpoint preserves dtype on load. bf16 is exactly the top
/// 16 bits of a finite f32, so values representable as bf16 round-
/// trip exactly through libtorch's f16/bf16 storage. Verifies both
/// dtype preservation and value correctness via to_f32_vec()'s
/// libtorch cast.
#[test]
fn load_safetensors_bf16_preserves_dtype() {
    use flodl::{DType, FlowBuilder, Linear};

    let dev = Device::CPU;
    let graph = FlowBuilder::new()
        .through(Linear::on_device(2, 2, dev).unwrap())
        .tag("m")
        .build()
        .unwrap();

    // bf16 representable values: pick f32s whose bottom 16 bits are 0.
    let exact_w = [1.0_f32, 2.0, -0.5, 0.25];
    let exact_b = [0.0_f32, -1.0];
    let to_bf16_bytes = |data: &[f32]| -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len() * 2);
        for &f in data {
            let top = (f.to_bits() >> 16) as u16;
            out.extend_from_slice(&top.to_le_bytes());
        }
        out
    };
    let bytes = serialize_entries(&[
        ("m.weight", Dtype::BF16, vec![2, 2], to_bf16_bytes(&exact_w)),
        ("m.bias", Dtype::BF16, vec![2], to_bf16_bytes(&exact_b)),
    ]);

    load_safetensors_into_graph(&graph, &bytes).unwrap();

    for (k, p) in graph.named_parameters() {
        let hf = hf_key_from_flodl_key(&k);
        assert_eq!(
            p.variable.data().dtype(),
            DType::BFloat16,
            "{hf}: dtype must be preserved as BF16 after load"
        );
        let data = p.variable.data().to_f32_vec().unwrap();
        match hf.as_str() {
            "m.weight" => assert_eq!(data, exact_w),
            "m.bias" => assert_eq!(data, exact_b),
            other => panic!("unexpected {other}"),
        }
    }
}

/// F16 checkpoint preserves dtype on load. Tests +1, -1, +0.5, +0
/// — values representable bit-exactly in f16, so to_f32_vec()'s
/// libtorch f16→f32 cast is lossless and gives back the original
/// mathematical values.
#[test]
fn load_safetensors_f16_preserves_dtype() {
    use flodl::{DType, FlowBuilder, Linear};

    let dev = Device::CPU;
    let graph = FlowBuilder::new()
        .through(Linear::on_device(1, 4, dev).unwrap())
        .tag("m")
        .build()
        .unwrap();

    // IEEE 754 binary16 bit patterns:
    // 0x3C00 = 1.0
    // 0xBC00 = -1.0
    // 0x3800 = 0.5
    // 0x0000 = +0.0
    let f16_bits: [u16; 4] = [0x3C00, 0xBC00, 0x3800, 0x0000];
    let mut bytes_w = Vec::with_capacity(8);
    for b in f16_bits {
        bytes_w.extend_from_slice(&b.to_le_bytes());
    }
    let bias_bits: [u16; 1] = [0x3C00];
    let bytes_b: Vec<u8> = bias_bits[0].to_le_bytes().to_vec();

    let st_bytes = serialize_entries(&[
        ("m.weight", Dtype::F16, vec![4, 1], bytes_w),
        ("m.bias", Dtype::F16, vec![4], bytes_b.repeat(4)),
    ]);
    load_safetensors_into_graph(&graph, &st_bytes).unwrap();

    for (k, p) in graph.named_parameters() {
        let hf = hf_key_from_flodl_key(&k);
        assert_eq!(
            p.variable.data().dtype(),
            DType::Float16,
            "{hf}: dtype must be preserved as F16 after load"
        );
        let data = p.variable.data().to_f32_vec().unwrap();
        match hf.as_str() {
            "m.weight" => assert_eq!(data, vec![1.0, -1.0, 0.5, 0.0]),
            "m.bias" => assert_eq!(data, vec![1.0, 1.0, 1.0, 1.0]),
            other => panic!("unexpected {other}"),
        }
    }
}

/// Full round-trip: F16 safetensors → load → save → bit-exact F16
/// safetensors. This is the contract the verify-export matrix runner
/// relies on for DeBERTa-v3 (pure F16 upstream).
#[test]
fn save_safetensors_f16_roundtrip_byte_exact() {
    use flodl::{FlowBuilder, Linear};

    let dev = Device::CPU;
    let graph = FlowBuilder::new()
        .through(Linear::on_device(1, 4, dev).unwrap())
        .tag("m")
        .build()
        .unwrap();

    let f16_bits: [u16; 4] = [0x3C00, 0xBC00, 0x3800, 0x0000];
    let bytes_w: Vec<u8> = f16_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let bytes_b: Vec<u8> = (0..4).flat_map(|_| 0x3C00u16.to_le_bytes()).collect();

    let src = serialize_entries(&[
        ("m.weight", Dtype::F16, vec![4, 1], bytes_w.clone()),
        ("m.bias", Dtype::F16, vec![4], bytes_b.clone()),
    ]);
    load_safetensors_into_graph(&graph, &src).unwrap();

    let saved = save_safetensors_from_graph(&graph).unwrap();
    let saved_st = SafeTensors::deserialize(&saved).unwrap();
    for (k, expected_bytes) in [("m.weight", &bytes_w), ("m.bias", &bytes_b)] {
        let v = saved_st.tensor(k).unwrap();
        assert_eq!(v.dtype(), Dtype::F16, "{k}: must save back as F16");
        assert_eq!(
            v.data(),
            expected_bytes.as_slice(),
            "{k}: F16 bytes must be bit-exact through load+save"
        );
    }
}

/// Same contract as the F16 round-trip, but for BF16. BF16 is the
/// dtype of choice for many recent LLMs — we want to be sure
/// flodl preserves it without surprise downcasts.
#[test]
fn save_safetensors_bf16_roundtrip_byte_exact() {
    use flodl::{FlowBuilder, Linear};

    let dev = Device::CPU;
    let graph = FlowBuilder::new()
        .through(Linear::on_device(2, 2, dev).unwrap())
        .tag("m")
        .build()
        .unwrap();

    // BF16 = top 16 bits of f32. These f32s have zero low bits → exact bf16.
    let exact_w = [1.0_f32, 2.0, -0.5, 0.25];
    let exact_b = [0.0_f32, -1.0];
    let to_bf16_bytes = |data: &[f32]| -> Vec<u8> {
        data.iter()
            .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let bytes_w = to_bf16_bytes(&exact_w);
    let bytes_b = to_bf16_bytes(&exact_b);

    let src = serialize_entries(&[
        ("m.weight", Dtype::BF16, vec![2, 2], bytes_w.clone()),
        ("m.bias", Dtype::BF16, vec![2], bytes_b.clone()),
    ]);
    load_safetensors_into_graph(&graph, &src).unwrap();

    let saved = save_safetensors_from_graph(&graph).unwrap();
    let saved_st = SafeTensors::deserialize(&saved).unwrap();
    for (k, expected_bytes) in [("m.weight", &bytes_w), ("m.bias", &bytes_b)] {
        let v = saved_st.tensor(k).unwrap();
        assert_eq!(v.dtype(), Dtype::BF16, "{k}: must save back as BF16");
        assert_eq!(
            v.data(),
            expected_bytes.as_slice(),
            "{k}: BF16 bytes must be bit-exact through load+save"
        );
    }
}

/// Validation failures must leave the graph untouched — callers rely
/// on "either all params loaded, or none" so they can fall back or
/// report safely. Missing key → error; on the error path the caller's
/// graph is free for them to mutate without inconsistent state.
#[test]
fn load_safetensors_missing_key_errors_loudly() {
    use flodl::{FlowBuilder, Linear};

    let dev = Device::CPU;
    let graph = FlowBuilder::new()
        .through(Linear::on_device(2, 2, dev).unwrap())
        .tag("m")
        .build()
        .unwrap();
    // Only ship the weight, not the bias.
    let w = vec![0.0_f32, 1.0, 2.0, 3.0];
    let bytes = serialize_entries(&[("m.weight", Dtype::F32, vec![2, 2], f32_le_bytes(&w))]);
    let err = load_safetensors_into_graph(&graph, &bytes)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("missing key"),
        "error must mention missing keys: {err}"
    );
    assert!(
        err.contains("m.bias"),
        "error must name the missing key: {err}"
    );
}

/// Integer dtypes are rejected explicitly rather than silently cast —
/// a user shipping an I32 checkpoint almost certainly means something
/// went wrong upstream.
#[test]
fn load_safetensors_rejects_integer_dtype() {
    use flodl::{FlowBuilder, Linear};

    let dev = Device::CPU;
    let graph = FlowBuilder::new()
        .through(Linear::on_device(1, 1, dev).unwrap())
        .tag("m")
        .build()
        .unwrap();
    // I32 bias: 4 bytes per element × 1 element.
    let bias_i32: Vec<u8> = 1_i32.to_le_bytes().to_vec();
    let w_bytes = f32_le_bytes(&[0.5_f32]);
    let bytes = serialize_entries(&[
        ("m.weight", Dtype::F32, vec![1, 1], w_bytes),
        ("m.bias", Dtype::I32, vec![1], bias_i32),
    ]);
    let err = load_safetensors_into_graph(&graph, &bytes)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("unsupported safetensors dtype"),
        "error must call out dtype: {err}"
    );
    assert!(
        err.contains("I32"),
        "error must name the offending dtype: {err}"
    );
}

/// Legacy BERT checkpoint key rewriting: only `LayerNorm.gamma`
/// and `LayerNorm.beta` suffixes are remapped; every other key
/// passes through untouched. `google-bert/bert-base-uncased` from the Hub ships
/// with the legacy suffixes on every LayerNorm parameter, so this
/// is the one knob that separates "loads" from "doesn't".
#[test]
fn bert_legacy_key_rename_rewrites_layernorm_suffixes() {
    assert_eq!(
        bert_legacy_key_rename("bert.embeddings.LayerNorm.gamma"),
        "bert.embeddings.LayerNorm.weight",
    );
    assert_eq!(
        bert_legacy_key_rename("bert.embeddings.LayerNorm.beta"),
        "bert.embeddings.LayerNorm.bias",
    );
    assert_eq!(
        bert_legacy_key_rename("bert.encoder.layer.3.attention.output.LayerNorm.gamma"),
        "bert.encoder.layer.3.attention.output.LayerNorm.weight",
    );
    // Non-LayerNorm keys pass through.
    assert_eq!(
        bert_legacy_key_rename("bert.encoder.layer.0.attention.self.query.weight"),
        "bert.encoder.layer.0.attention.self.query.weight",
    );
    // Partial matches (wrong suffix) are NOT remapped.
    assert_eq!(bert_legacy_key_rename("something.gamma"), "something.gamma",);
}

/// MLM decoder-bias tying: HF's `BertForMaskedLM` and
/// `RobertaForMaskedLM` save a single top-level `bias` Parameter
/// that is tied to `decoder.bias`; our graph stores the bias on the
/// decoder `Linear` directly. The rename maps the checkpoint key
/// onto our graph key so MLM checkpoints load cleanly.
#[test]
fn bert_legacy_key_rename_retags_mlm_tied_bias() {
    assert_eq!(
        bert_legacy_key_rename("cls.predictions.bias"),
        "cls.predictions.decoder.bias",
    );
    assert_eq!(
        bert_legacy_key_rename("lm_head.bias"),
        "lm_head.decoder.bias",
    );
    // Exact-match, not suffix — a hypothetical nested key
    // `something.cls.predictions.bias` is untouched.
    assert_eq!(
        bert_legacy_key_rename("something.cls.predictions.bias"),
        "something.cls.predictions.bias",
    );
}

/// If a checkpoint has BOTH `foo.LayerNorm.gamma` and
/// `foo.LayerNorm.weight`, renaming collapses them onto the same
/// canonical slot. The loader must surface this as a loud error
/// rather than silently picking one — otherwise the user gets a
/// non-deterministic load depending on HashMap iteration order.
#[test]
fn load_safetensors_rename_collision_errors_loudly() {
    use flodl::{FlowBuilder, Linear};

    let dev = Device::CPU;
    let graph = FlowBuilder::new()
        .through(Linear::on_device(2, 2, dev).unwrap())
        .tag("m")
        .build()
        .unwrap();

    // Same numeric data, different legacy-vs-canonical key suffix.
    let w = f32_le_bytes(&[0.0, 1.0, 2.0, 3.0]);
    let b = f32_le_bytes(&[0.1, 0.2]);
    let bytes = serialize_entries(&[
        ("m.weight", Dtype::F32, vec![2, 2], w.clone()),
        ("m.LayerNorm.gamma", Dtype::F32, vec![2], b.clone()),
        ("m.LayerNorm.weight", Dtype::F32, vec![2], b),
    ]);

    let err = load_safetensors_into_graph_with_rename(&graph, &bytes, bert_legacy_key_rename)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("rename collision"),
        "error must identify the collision: {err}"
    );
    assert!(
        err.contains("LayerNorm.weight"),
        "error must name the canonical key involved: {err}"
    );
}

/// f16 helper unit check: +Inf, -Inf, NaN, smallest subnormal all
/// survive the widening with the right classification. The loader
/// catches these via the same function, so a regression here would
/// surface as Weird Numerical Behaviour™ in a loaded model.
#[test]
fn f16_bits_to_f32_special_values() {
    // +Inf: exp=0x1f, mantissa=0
    assert!(f16_bits_to_f32(0x7C00).is_infinite() && f16_bits_to_f32(0x7C00).is_sign_positive());
    // -Inf
    assert!(f16_bits_to_f32(0xFC00).is_infinite() && f16_bits_to_f32(0xFC00).is_sign_negative());
    // NaN: exp=0x1f, mantissa != 0
    assert!(f16_bits_to_f32(0x7E00).is_nan());
    // Smallest positive subnormal half: 0x0001 = 2^-24
    let tiny = f16_bits_to_f32(0x0001);
    assert!(
        (tiny - 2.0_f32.powi(-24)).abs() < 1e-10,
        "tiny subnormal wrong: {tiny}"
    );
    // -0.0 preserves sign
    assert!(f16_bits_to_f32(0x8000).is_sign_negative() && f16_bits_to_f32(0x8000) == 0.0);
}

#[test]
fn expected_from_graph_converts_slash_to_dot() {
    use flodl::{FlowBuilder, Linear, Module};
    let fb = FlowBuilder::new()
        .through(Linear::new(4, 2).unwrap())
        .tag("bert.pooler.dense");
    let graph = fb.build().unwrap();
    let expected = expected_from_graph(&graph);
    // Graph::named_parameters gives "bert.pooler.dense/weight" and
    // "bert.pooler.dense/bias". expected_from_graph must swap the last
    // slash for a dot.
    let keys: Vec<&str> = expected.iter().map(|e| e.key.as_str()).collect();
    assert!(
        keys.contains(&"bert.pooler.dense.weight"),
        "expected HF-dotted key missing, got {keys:?}"
    );
    assert!(
        keys.contains(&"bert.pooler.dense.bias"),
        "expected HF-dotted key missing, got {keys:?}"
    );
    // Sanity check: the parameter count matches Graph's own view.
    assert_eq!(expected.len(), graph.parameters().len());
}

/// Save → load roundtrip via the public API: build a tagged graph
/// with deterministic parameter values, save it as safetensors bytes,
/// load those bytes into a fresh graph with the same structure, and
/// assert every parameter is bit-exact f32 (lossless via the f32
/// storage dtype).
///
/// This is the strongest invariant the save layer must hold —
/// "anything flodl loads, flodl can save and reload identically" —
/// and it's what the per-family `_roundtrip_*_live` tests rely on.
#[test]
fn save_safetensors_load_roundtrip() {
    use flodl::{FlowBuilder, Linear, Module, Variable};

    let dev = Device::CPU;
    let in_dim = 3_i64;
    let out_dim = 2_i64;

    // Source graph with pinned weights.
    let src = FlowBuilder::new()
        .through(Linear::on_device(in_dim, out_dim, dev).unwrap())
        .tag("my.linear")
        .build()
        .unwrap();
    let src_weight: Vec<f32> = (0..(in_dim * out_dim) as usize)
        .map(|i| 0.5 + i as f32 * 0.1)
        .collect();
    let src_bias: Vec<f32> = (0..out_dim as usize)
        .map(|i| -1.0 + i as f32 * 0.25)
        .collect();
    for (k, p) in src.named_parameters() {
        let hf = hf_key_from_flodl_key(&k);
        let t = match hf.as_str() {
            "my.linear.weight" => Tensor::from_f32(&src_weight, &[out_dim, in_dim], dev).unwrap(),
            "my.linear.bias" => Tensor::from_f32(&src_bias, &[out_dim], dev).unwrap(),
            other => panic!("unexpected key {other}"),
        };
        p.variable.set_data(t);
    }

    let bytes = save_safetensors_from_graph(&src).unwrap();

    // Destination graph: fresh, same structure, load the saved bytes.
    let dst = FlowBuilder::new()
        .through(Linear::on_device(in_dim, out_dim, dev).unwrap())
        .tag("my.linear")
        .build()
        .unwrap();
    load_safetensors_into_graph(&dst, &bytes).unwrap();

    let mut dw: Option<Vec<f32>> = None;
    let mut db: Option<Vec<f32>> = None;
    for (k, p) in dst.named_parameters() {
        let hf = hf_key_from_flodl_key(&k);
        let data = p.variable.data().to_f32_vec().unwrap();
        match hf.as_str() {
            "my.linear.weight" => dw = Some(data),
            "my.linear.bias" => db = Some(data),
            other => panic!("unexpected key {other}"),
        }
    }
    assert_eq!(dw.unwrap(), src_weight);
    assert_eq!(db.unwrap(), src_bias);

    let _keep_alive: Vec<Variable> = dst.parameters().into_iter().map(|p| p.variable).collect();
}

/// Saved keys land in HF-dotted form (slash → dot on the last
/// segment) and the byte payload of each tensor matches the source's
/// little-endian f32 representation. Guards against the save path
/// drifting from `hf_key_from_flodl_key` and against endianness bugs
/// in the byte assembly.
#[test]
fn save_safetensors_uses_hf_dotted_keys_and_le_f32() {
    use flodl::{FlowBuilder, Linear};

    let dev = Device::CPU;
    let graph = FlowBuilder::new()
        .through(Linear::on_device(2, 1, dev).unwrap())
        .tag("encoder.layer.0.attention.output.dense")
        .build()
        .unwrap();
    let w = vec![0.25_f32, -0.5];
    let b = vec![1.0_f32];
    for (k, p) in graph.named_parameters() {
        let hf = hf_key_from_flodl_key(&k);
        let t = match hf.as_str() {
            "encoder.layer.0.attention.output.dense.weight" => {
                Tensor::from_f32(&w, &[1, 2], dev).unwrap()
            }
            "encoder.layer.0.attention.output.dense.bias" => {
                Tensor::from_f32(&b, &[1], dev).unwrap()
            }
            other => panic!("unexpected key {other}"),
        };
        p.variable.set_data(t);
    }

    let bytes = save_safetensors_from_graph(&graph).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();

    let names: HashSet<&str> = st.names().into_iter().collect();
    assert!(
        names.contains("encoder.layer.0.attention.output.dense.weight"),
        "expected HF-dotted key in output, got {names:?}"
    );
    assert!(
        names.contains("encoder.layer.0.attention.output.dense.bias"),
        "expected HF-dotted key in output, got {names:?}"
    );

    let w_view = st
        .tensor("encoder.layer.0.attention.output.dense.weight")
        .unwrap();
    assert_eq!(w_view.dtype(), Dtype::F32);
    assert_eq!(w_view.shape(), &[1_usize, 2]);
    let w_back: Vec<f32> = w_view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(w_back, w);

    let b_view = st
        .tensor("encoder.layer.0.attention.output.dense.bias")
        .unwrap();
    let b_back: Vec<f32> = b_view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(b_back, b);
}

/// Tied parameters — same `Variable` reachable under multiple tags —
/// are deduped by `named_parameters` upstream, so the saved file
/// contains the shared weight exactly once. Verifies the upstream
/// guarantee actually flows through to the byte output: a
/// hand-rolled tying via [`Linear::from_shared_weight`] yields one
/// weight key, not two.
#[test]
fn save_safetensors_dedups_shared_weights() {
    use flodl::{FlowBuilder, Linear, Parameter};

    let dev = Device::CPU;
    let primary = Linear::on_device(2, 2, dev).unwrap();
    let shared_weight = primary.weight.clone(); // Rc clone, same storage
    let tied_bias = Parameter::new(
        Tensor::from_f32(&[0.0_f32, 0.0], &[2], dev).unwrap(),
        "bias",
    );
    let tied = Linear::from_shared_weight(shared_weight, Some(tied_bias));

    let graph = FlowBuilder::new()
        .through(primary)
        .tag("primary")
        .through(tied)
        .tag("tied")
        .build()
        .unwrap();

    let bytes = save_safetensors_from_graph(&graph).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let names: HashSet<&str> = st.names().into_iter().collect();

    // Shared weight ships once under whichever tag named_parameters
    // visited first. Each Linear's own bias is a distinct Parameter,
    // so both bias keys appear.
    let weight_count = ["primary.weight", "tied.weight"]
        .iter()
        .filter(|k| names.contains(*k))
        .count();
    assert_eq!(
        weight_count, 1,
        "shared weight must ship exactly once, got {names:?}",
    );
    assert!(
        names.contains("primary.bias"),
        "primary bias missing in {names:?}"
    );
    assert!(
        names.contains("tied.bias"),
        "tied bias missing in {names:?}"
    );
}

/// File-path variant: same save → load roundtrip but through disk.
/// Exercises the file writer and the path-in-error-message behaviour
/// indirectly.
#[test]
fn save_safetensors_file_roundtrip() {
    use flodl::{FlowBuilder, Linear};

    let dev = Device::CPU;
    let graph = FlowBuilder::new()
        .through(Linear::on_device(2, 1, dev).unwrap())
        .tag("m")
        .build()
        .unwrap();
    let w = vec![0.1_f32, 0.2];
    let b = vec![0.3_f32];
    for (k, p) in graph.named_parameters() {
        let hf = hf_key_from_flodl_key(&k);
        let t = match hf.as_str() {
            "m.weight" => Tensor::from_f32(&w, &[1, 2], dev).unwrap(),
            "m.bias" => Tensor::from_f32(&b, &[1], dev).unwrap(),
            other => panic!("unexpected {other}"),
        };
        p.variable.set_data(t);
    }

    let path = std::env::temp_dir().join(format!(
        "flodl_hf_save_test_{}.safetensors",
        std::process::id()
    ));
    save_safetensors_file_from_graph(&graph, &path).unwrap();

    // Load back into a fresh graph through the file API; assert match.
    let fresh = FlowBuilder::new()
        .through(Linear::on_device(2, 1, dev).unwrap())
        .tag("m")
        .build()
        .unwrap();
    load_safetensors_file_into_graph(&fresh, &path).unwrap();

    let _ = std::fs::remove_file(&path);

    for (k, p) in fresh.named_parameters() {
        let hf = hf_key_from_flodl_key(&k);
        let data = p.variable.data().to_f32_vec().unwrap();
        match hf.as_str() {
            "m.weight" => assert_eq!(data, w),
            "m.bias" => assert_eq!(data, b),
            other => panic!("unexpected {other}"),
        }
    }
}
