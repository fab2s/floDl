    use super::*;
    use crate::safetensors_io::expected_from_graph;
    use flodl::{HasGraph, TensorOptions};

    /// The 16 parameter keys every encoder layer exposes, template-formatted
    /// for a given layer index.
    fn expected_layer_keys(i: i64) -> Vec<String> {
        let suffixes = [
            "attention.output.LayerNorm.bias",
            "attention.output.LayerNorm.weight",
            "attention.output.dense.bias",
            "attention.output.dense.weight",
            "attention.self.key.bias",
            "attention.self.key.weight",
            "attention.self.query.bias",
            "attention.self.query.weight",
            "attention.self.value.bias",
            "attention.self.value.weight",
            "intermediate.dense.bias",
            "intermediate.dense.weight",
            "output.LayerNorm.bias",
            "output.LayerNorm.weight",
            "output.dense.bias",
            "output.dense.weight",
        ];
        suffixes.iter().map(|s| format!("bert.encoder.layer.{i}.{s}")).collect()
    }

    /// The full BERT-base key set (199 keys: 5 embeddings + 16×12 layers +
    /// 2 pooler) matches HuggingFace safetensors dotted form exactly. The
    /// expected list is built from the layer template so adding or removing
    /// layers only changes the config, not the test.
    #[test]
    fn bert_parameter_keys_match_hf_dotted_form() {
        let config = BertConfig::bert_base_uncased();
        let graph = BertModel::build(&config).unwrap();
        let expected = expected_from_graph(&graph);

        let mut keys: Vec<String> = expected.iter().map(|p| p.key.clone()).collect();
        keys.sort();

        let mut want: Vec<String> = vec![
            "bert.embeddings.LayerNorm.bias".into(),
            "bert.embeddings.LayerNorm.weight".into(),
            "bert.embeddings.position_embeddings.weight".into(),
            "bert.embeddings.token_type_embeddings.weight".into(),
            "bert.embeddings.word_embeddings.weight".into(),
        ];
        for i in 0..config.num_hidden_layers {
            want.extend(expected_layer_keys(i));
        }
        want.extend([
            "bert.pooler.dense.bias".into(),
            "bert.pooler.dense.weight".into(),
        ]);
        want.sort();

        // Sanity: BERT-base has exactly 199 parameter tensors.
        assert_eq!(want.len(), 199, "expected-key list size drift");
        assert_eq!(keys, want, "BERT parameter keys must match HF exactly");
    }

    /// Parameter shapes must match the BERT-base-uncased reference. If any
    /// constant in `BertConfig::bert_base_uncased` regresses, this test
    /// pins it.
    #[test]
    fn bert_parameter_shapes_match_bert_base_uncased() {
        let config = BertConfig::bert_base_uncased();
        let graph = BertModel::build(&config).unwrap();
        let expected = expected_from_graph(&graph);
        let by_key: std::collections::HashMap<&str, &[i64]> = expected
            .iter()
            .map(|p| (p.key.as_str(), p.shape.as_slice()))
            .collect();

        assert_eq!(by_key["bert.embeddings.word_embeddings.weight"],       &[30522, 768]);
        assert_eq!(by_key["bert.embeddings.position_embeddings.weight"],   &[512, 768]);
        assert_eq!(by_key["bert.embeddings.token_type_embeddings.weight"], &[2, 768]);
        assert_eq!(by_key["bert.embeddings.LayerNorm.weight"],             &[768]);
        assert_eq!(by_key["bert.embeddings.LayerNorm.bias"],               &[768]);

        // Every encoder layer has the same shape profile. Sweep all
        // layers so any mis-wiring on a specific index surfaces here
        // rather than being masked by only checking layer 0.
        for i in 0..config.num_hidden_layers {
            let p = format!("bert.encoder.layer.{i}");
            assert_eq!(by_key[&*format!("{p}.attention.self.query.weight")],        &[768, 768]);
            assert_eq!(by_key[&*format!("{p}.attention.self.query.bias")],          &[768]);
            assert_eq!(by_key[&*format!("{p}.attention.self.key.weight")],          &[768, 768]);
            assert_eq!(by_key[&*format!("{p}.attention.self.value.weight")],        &[768, 768]);
            assert_eq!(by_key[&*format!("{p}.attention.output.dense.weight")],      &[768, 768]);
            assert_eq!(by_key[&*format!("{p}.attention.output.LayerNorm.weight")],  &[768]);
            assert_eq!(by_key[&*format!("{p}.intermediate.dense.weight")],          &[3072, 768]);
            assert_eq!(by_key[&*format!("{p}.intermediate.dense.bias")],            &[3072]);
            assert_eq!(by_key[&*format!("{p}.output.dense.weight")],                &[768, 3072]);
            assert_eq!(by_key[&*format!("{p}.output.dense.bias")],                  &[768]);
            assert_eq!(by_key[&*format!("{p}.output.LayerNorm.weight")],            &[768]);
        }

        assert_eq!(by_key["bert.pooler.dense.weight"], &[768, 768]);
        assert_eq!(by_key["bert.pooler.dense.bias"],   &[768]);
    }

    /// Encoder stack honours `config.num_hidden_layers`. Pins the loop so
    /// a future regression that (e.g.) wires in one hardcoded layer is
    /// caught immediately.
    #[test]
    fn bert_layer_count_scales_with_config() {
        for n in [1_i64, 3, 6] {
            let config = BertConfig {
                num_hidden_layers: n,
                ..BertConfig::bert_base_uncased()
            };
            let graph = BertModel::build(&config).unwrap();
            let expected = expected_from_graph(&graph);
            let total = expected.len();
            // 5 embedding keys + 16 per layer + 2 pooler keys.
            let want_total = 5 + 16 * n as usize + 2;
            assert_eq!(
                total, want_total,
                "num_hidden_layers={n}: got {total} keys, expected {want_total}",
            );

            // The highest-indexed layer actually exists — catches "stopped
            // one short" / "started at 1" bugs in the loop.
            let last_layer_key = format!(
                "bert.encoder.layer.{}.attention.self.query.weight", n - 1,
            );
            assert!(
                expected.iter().any(|p| p.key == last_layer_key),
                "last layer key {last_layer_key:?} missing from graph keys",
            );
        }
    }

    /// Small BERT preset the mask tests reuse. One layer, tiny hidden, no
    /// dropout — enough wiring to exercise embeddings + encoder + pooler
    /// without the cost of `bert-base`.
    fn tiny_bert_config() -> BertConfig {
        BertConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: 8,
            type_vocab_size: 2,
            pad_token_id: Some(0),
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            hidden_act: GeluApprox::Exact,
            num_labels: None,
            id2label: None,
            architectures: None,
        }
    }

    /// Standard `bert-base-uncased` config.json — round-trip through
    /// `from_json_str` must produce a config that matches the hardcoded
    /// `bert_base_uncased()` preset. Exercises every required field plus
    /// a few optional ones and unknown-field tolerance.
    #[test]
    fn bert_config_from_json_str_matches_base_preset() {
        let json = r#"{
            "architectures": ["BertForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "gradient_checkpointing": false,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.6.0.dev0",
            "type_vocab_size": 2,
            "use_cache": true,
            "vocab_size": 30522
        }"#;
        let got = BertConfig::from_json_str(json).unwrap();
        let want = BertConfig::bert_base_uncased();
        assert_eq!(got.vocab_size,              want.vocab_size);
        assert_eq!(got.hidden_size,             want.hidden_size);
        assert_eq!(got.num_hidden_layers,       want.num_hidden_layers);
        assert_eq!(got.num_attention_heads,     want.num_attention_heads);
        assert_eq!(got.intermediate_size,       want.intermediate_size);
        assert_eq!(got.max_position_embeddings, want.max_position_embeddings);
        assert_eq!(got.type_vocab_size,         want.type_vocab_size);
        assert_eq!(got.pad_token_id,            want.pad_token_id);
        assert!((got.layer_norm_eps               - want.layer_norm_eps).abs() < 1e-18);
        assert!((got.hidden_dropout_prob          - want.hidden_dropout_prob).abs() < 1e-9);
        assert!((got.attention_probs_dropout_prob - want.attention_probs_dropout_prob).abs() < 1e-9);
    }

    /// Missing a required integer field must surface a clear error that
    /// names the offending key — the whole point of the validator is to
    /// be loud about drift.
    #[test]
    fn bert_config_from_json_str_rejects_missing_field() {
        // No `hidden_size`.
        let json = r#"{
            "vocab_size": 30522,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2
        }"#;
        let err = BertConfig::from_json_str(json).unwrap_err().to_string();
        assert!(err.contains("hidden_size"),
            "error must name the missing field: {err}");
        assert!(err.contains("missing required integer field"),
            "error must explain the failure mode: {err}");
    }

    /// Explicit `"pad_token_id": null` and absent `pad_token_id` must both
    /// produce `None` — these are the two ways HF configs spell "no
    /// dedicated pad token" (e.g. GPT-2-style).
    #[test]
    fn bert_config_from_json_str_pad_token_id_nullable() {
        let required_fields = r#"
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2
        "#;
        let explicit_null = format!(r#"{{ {required_fields}, "pad_token_id": null }}"#);
        let absent        = format!(r#"{{ {required_fields} }}"#);
        let a = BertConfig::from_json_str(&explicit_null).unwrap();
        let b = BertConfig::from_json_str(&absent).unwrap();
        assert_eq!(a.pad_token_id, None);
        assert_eq!(b.pad_token_id, None);
    }

    /// `num_labels` + `id2label` parse correctly from a fine-tuned
    /// checkpoint's config. Labels are ordered by integer id so `Vec[k]`
    /// reads as `id2label[k]`, and `num_labels` tracks the entry count.
    #[test]
    fn bert_config_from_json_str_parses_task_head_metadata() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "num_labels": 3,
            "id2label": { "2": "JOY", "0": "ANGER", "1": "SADNESS" },
            "label2label": { "IGNORED": 1 }
        }"#;
        let c = BertConfig::from_json_str(json).unwrap();
        assert_eq!(c.num_labels, Some(3));
        assert_eq!(
            c.id2label,
            Some(vec!["ANGER".to_string(), "SADNESS".to_string(), "JOY".to_string()]),
        );
    }

    /// If only `id2label` is present (some older fine-tunes omit
    /// `num_labels`), `num_labels` is derived from the label count.
    #[test]
    fn bert_config_num_labels_derived_from_id2label() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "id2label": { "0": "NEGATIVE", "1": "POSITIVE" }
        }"#;
        let c = BertConfig::from_json_str(json).unwrap();
        assert_eq!(c.num_labels, Some(2));
        assert_eq!(c.id2label.unwrap(), vec!["NEGATIVE", "POSITIVE"]);
    }

    /// Base configs without any task-head metadata leave both fields as
    /// `None`. Task-head constructors pick sensible fallbacks from the
    /// runtime `num_labels` argument in that case.
    #[test]
    fn bert_config_without_task_metadata_is_none() {
        let c = BertConfig::bert_base_uncased();
        assert_eq!(c.num_labels, None);
        assert_eq!(c.id2label, None);
    }

    /// Non-contiguous label ids (gap, duplicate, or negative) must surface
    /// as a clear error. Silently reindexing would misalign class names
    /// with logits row indices on load.
    #[test]
    fn bert_config_rejects_non_contiguous_id2label() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "id2label": { "0": "A", "2": "C" }
        }"#;
        let err = BertConfig::from_json_str(json).unwrap_err().to_string();
        assert!(err.contains("contiguous"), "error must call out contiguity: {err}");
    }

    /// Optional dropout + layer-norm-eps fields fall back to BERT defaults
    /// when absent. Keeps configs for bare-metal / test-only checkpoints
    /// parseable without boilerplate.
    #[test]
    fn bert_config_from_json_str_uses_defaults_for_missing_optional_fields() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2
        }"#;
        let c = BertConfig::from_json_str(json).unwrap();
        assert!((c.layer_norm_eps               - 1e-12).abs() < 1e-18);
        assert!((c.hidden_dropout_prob          - 0.1).abs() < 1e-9);
        assert!((c.attention_probs_dropout_prob - 0.1).abs() < 1e-9);
    }

    /// Round-trip: preset -> to_json_str -> from_json_str recovers the
    /// same config. Guards against fields the writer forgets to emit
    /// (any required_i64 that's missing in the emitted JSON errors
    /// during parse) and against silent default drift.
    #[test]
    fn bert_config_to_json_str_round_trip() {
        let preset = BertConfig::bert_base_uncased();
        let s = preset.to_json_str();
        let recovered = BertConfig::from_json_str(&s).unwrap();
        // Round-trip is idempotent: emitting the recovered config
        // produces identical JSON.
        assert_eq!(preset.to_json_str(), recovered.to_json_str());
        // HF dispatch keys present so AutoConfig loads it.
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v.get("model_type").and_then(|x| x.as_str()), Some("bert"));
        assert_eq!(
            v.get("architectures")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>()),
            Some(vec!["BertModel"]),
        );
    }

    /// Task-head config survives round-trip: id2label + num_labels +
    /// non-None pad_token_id all land in the emitted JSON and re-parse.
    #[test]
    fn bert_config_to_json_str_preserves_task_head_metadata() {
        let mut preset = BertConfig::bert_base_uncased();
        preset.num_labels = Some(3);
        preset.id2label = Some(vec![
            "POS".to_string(),
            "NEG".to_string(),
            "NEU".to_string(),
        ]);
        let s = preset.to_json_str();
        let r = BertConfig::from_json_str(&s).unwrap();
        assert_eq!(r.num_labels, Some(3));
        assert_eq!(
            r.id2label.as_deref(),
            Some(&[
                "POS".to_string(),
                "NEG".to_string(),
                "NEU".to_string(),
            ][..])
        );
        // label2id was emitted alongside id2label (HF convention).
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        let lab2 = v.get("label2id").and_then(|x| x.as_object()).unwrap();
        assert_eq!(lab2.get("POS").and_then(|x| x.as_i64()), Some(0));
        assert_eq!(lab2.get("NEU").and_then(|x| x.as_i64()), Some(2));
    }

    /// Smoke test: construct a tiny BERT on CPU, run forward_multi with
    /// made-up ids + an all-attend mask, and verify the output shape.
    /// Catches obvious wiring breakage (residual mismatch, missing named
    /// input, transpose axis bug) without requiring real tokenized inputs.
    #[test]
    fn bert_forward_shape_smoke() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let graph = BertModel::on_device(&config, dev).unwrap();
        graph.eval();

        let batch = 2;
        let seq = 4;
        let word_ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4, 5, 6, 7, 0], &[batch, seq], dev).unwrap(),
            false,
        );
        let position_ids = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3, 0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let token_type_ids = Variable::new(
            Tensor::from_i64(&[0, 0, 0, 0, 1, 1, 1, 1], &[batch, seq], dev).unwrap(),
            false,
        );
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions { dtype: DType::Float32, device: dev }).unwrap();
        let attention_mask = Variable::new(
            build_extended_attention_mask(&mask_flat).unwrap(),
            false,
        );

        let out = graph
            .forward_multi(&[word_ids, position_ids, token_type_ids, attention_mask])
            .unwrap();
        // Pooler reduces [batch, seq, hidden] → [batch, hidden]
        assert_eq!(out.shape(), vec![batch, config.hidden_size]);
    }

    /// `build_extended_attention_mask` turns a `[B, S]` 0/1 mask into a
    /// `[B, 1, 1, S]` additive f32 mask: attend positions → `0.0`, mask
    /// positions → `-1e4`. Pins the HF convention so Phase-2 checkpoint
    /// parity doesn't drift when we compare against PyTorch.
    #[test]
    fn extended_attention_mask_shape_and_values() {
        let dev = Device::CPU;
        // [2, 3] with batch-0 fully attending and batch-1 masking the
        // trailing position (e.g. padding after [CLS] tok tok).
        let raw = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0, 1.0, 0.0], &[2, 3], dev).unwrap();
        let additive = build_extended_attention_mask(&raw).unwrap();
        assert_eq!(additive.shape(), vec![2, 1, 1, 3]);

        let values: Vec<f32> = additive.reshape(&[6]).unwrap().to_f32_vec().unwrap();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 0.0);
        assert_eq!(values[3], 0.0);
        assert_eq!(values[4], 0.0);
        assert!((values[5] - -1e4).abs() < 1e-3, "masked position should be ~-1e4, got {}", values[5]);
    }
    // ── Task head tests ──────────────────────────────────────────────────

    /// `BertModel::on_device_without_pooler` drops the last two parameter
    /// tensors (`bert.pooler.dense.{weight,bias}`). The remaining 197
    /// keys stay in lockstep with the pooled backbone — critical for
    /// checkpoint loading when task heads sit on top of
    /// `add_pooling_layer=False`.
    #[test]
    fn bert_without_pooler_drops_two_keys() {
        let config = BertConfig::bert_base_uncased();
        let graph = BertModel::on_device_without_pooler(&config, Device::CPU).unwrap();
        let expected = expected_from_graph(&graph);
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();

        assert_eq!(expected.len(), 197, "197 backbone keys expected");
        assert!(!keys.iter().any(|k| k.starts_with("bert.pooler.")));
    }

    /// `BertForSequenceClassification` adds exactly two classifier keys
    /// on top of a pooled backbone.
    #[test]
    fn sequence_classification_parameter_keys_match_hf() {
        let config = BertConfig::bert_base_uncased();
        let head = BertForSequenceClassification::on_device(&config, 3, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let mut head_keys: Vec<&str> = expected
            .iter()
            .map(|p| p.key.as_str())
            .filter(|k| !k.starts_with("bert."))
            .collect();
        head_keys.sort();
        assert_eq!(head_keys, vec!["classifier.bias", "classifier.weight"]);

        let by_key: std::collections::HashMap<&str, &[i64]> = expected
            .iter().map(|p| (p.key.as_str(), p.shape.as_slice())).collect();
        assert_eq!(by_key["classifier.weight"], &[3, 768]);
        assert_eq!(by_key["classifier.bias"],   &[3]);
    }

    /// The config's `id2label` (if present) flows through to `labels()`.
    /// Otherwise the `LABEL_k` fallback kicks in.
    #[test]
    fn sequence_classification_labels_from_config_or_fallback() {
        let mut cfg = BertConfig::bert_base_uncased();
        cfg.num_labels = Some(3);
        cfg.id2label = Some(vec!["A".into(), "B".into(), "C".into()]);
        let head = BertForSequenceClassification::on_device(&cfg, 3, Device::CPU).unwrap();
        assert_eq!(head.labels(), &["A".to_string(), "B".to_string(), "C".to_string()]);

        let bare = BertConfig::bert_base_uncased();
        let fallback = BertForSequenceClassification::on_device(&bare, 2, Device::CPU).unwrap();
        assert_eq!(fallback.labels(), &["LABEL_0".to_string(), "LABEL_1".to_string()]);
    }

    /// Smoke: forward through the full classification graph produces
    /// `[batch, num_labels]` logits.
    #[test]
    fn sequence_classification_forward_shape_smoke() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let head = BertForSequenceClassification::on_device(&config, 5, dev).unwrap();
        head.graph().eval();

        let batch = 2;
        let seq = 4;
        let ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4, 5, 6, 7, 0], &[batch, seq], dev).unwrap(),
            false,
        );
        let pos = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3, 0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(
            Tensor::from_i64(&[0; 8], &[batch, seq], dev).unwrap(),
            false,
        );
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions {
            dtype: DType::Float32, device: dev,
        }).unwrap();
        let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let out = head.graph().forward_multi(&[ids, pos, tt, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, 5]);
    }

    /// `BertForTokenClassification` adds two classifier keys on top of
    /// an un-pooled backbone. Pooler must be absent.
    #[test]
    fn token_classification_parameter_keys_match_hf() {
        let config = BertConfig::bert_base_uncased();
        let head = BertForTokenClassification::on_device(&config, 9, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();
        assert!(!keys.iter().any(|k| k.starts_with("bert.pooler.")),
            "token classification must not carry pooler params");
        assert!(keys.contains(&"classifier.weight"));
        assert!(keys.contains(&"classifier.bias"));

        let by_key: std::collections::HashMap<&str, &[i64]> = expected
            .iter().map(|p| (p.key.as_str(), p.shape.as_slice())).collect();
        assert_eq!(by_key["classifier.weight"], &[9, 768]);
        assert_eq!(by_key["classifier.bias"],   &[9]);
    }

    /// Smoke: token classifier emits per-token logits of shape
    /// `[batch, seq, num_labels]`.
    #[test]
    fn token_classification_forward_shape_smoke() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let head = BertForTokenClassification::on_device(&config, 7, dev).unwrap();
        head.graph().eval();

        let batch = 2;
        let seq = 4;
        let ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4, 5, 6, 7, 0], &[batch, seq], dev).unwrap(),
            false,
        );
        let pos = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3, 0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(Tensor::from_i64(&[0; 8], &[batch, seq], dev).unwrap(), false);
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions {
            dtype: DType::Float32, device: dev,
        }).unwrap();
        let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let out = head.graph().forward_multi(&[ids, pos, tt, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, seq, 7]);
    }

    /// `BertForQuestionAnswering` adds exactly `qa_outputs.{weight,bias}`
    /// of shape `[2, H]` / `[2]`.
    #[test]
    fn question_answering_parameter_keys_match_hf() {
        let config = BertConfig::bert_base_uncased();
        let head = BertForQuestionAnswering::on_device(&config, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let mut head_keys: Vec<&str> = expected
            .iter().map(|p| p.key.as_str()).filter(|k| !k.starts_with("bert.")).collect();
        head_keys.sort();
        assert_eq!(head_keys, vec!["qa_outputs.bias", "qa_outputs.weight"]);

        let by_key: std::collections::HashMap<&str, &[i64]> = expected
            .iter().map(|p| (p.key.as_str(), p.shape.as_slice())).collect();
        assert_eq!(by_key["qa_outputs.weight"], &[2, 768]);
        assert_eq!(by_key["qa_outputs.bias"],   &[2]);
    }

    /// Smoke: QA head emits `[batch, seq, 2]` (start, end logits).
    #[test]
    fn question_answering_forward_shape_smoke() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let head = BertForQuestionAnswering::on_device(&config, dev).unwrap();
        head.graph().eval();

        let batch = 1;
        let seq = 4;
        let ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4], &[batch, seq], dev).unwrap(),
            false,
        );
        let pos = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(Tensor::from_i64(&[0; 4], &[batch, seq], dev).unwrap(), false);
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions {
            dtype: DType::Float32, device: dev,
        }).unwrap();
        let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let out = head.graph().forward_multi(&[ids, pos, tt, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, seq, 2]);
    }

    /// Zero labels must error rather than produce a zero-width head
    /// that silently passes through later shape checks.
    #[test]
    fn task_heads_reject_zero_labels() {
        let config = BertConfig::bert_base_uncased();
        let dev = Device::CPU;
        assert!(BertForSequenceClassification::on_device(&config, 0, dev).is_err());
        assert!(BertForTokenClassification::on_device(&config, 0, dev).is_err());
    }

    /// `HasGraph` should return a reference to the same underlying graph
    /// the head owns (pointer equality) — the contract graph-aware
    /// framework paths rely on to reach the head's real parameters.
    #[test]
    fn has_graph_returns_inner_graph_by_reference() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let seq   = BertForSequenceClassification::on_device(&config, 3, dev).unwrap();
        let token = BertForTokenClassification::on_device(&config, 5, dev).unwrap();
        let qa    = BertForQuestionAnswering::on_device(&config, dev).unwrap();
        assert!(std::ptr::eq(seq.graph(),   <BertForSequenceClassification as HasGraph>::graph(&seq)));
        assert!(std::ptr::eq(token.graph(), <BertForTokenClassification as HasGraph>::graph(&token)));
        assert!(std::ptr::eq(qa.graph(),    <BertForQuestionAnswering as HasGraph>::graph(&qa)));
    }

    /// End-to-end single-device training step on CPU: build a head, wire
    /// the optimizer + training mode on its graph, run a full forward →
    /// loss → backward → step cycle, confirm no error and the loss is
    /// finite. Multi-GPU DDP over heads rides the cooperative
    /// (`into_worker`) tier, covered by flodl's own DDP suite + `hf-ddp`.
    #[test]
    fn head_drives_cpu_training_step() {
        use flodl::Adam;
        use crate::task_heads::sequence_classification_loss;

        let config = tiny_bert_config();
        let dev = Device::CPU;
        let head = BertForSequenceClassification::on_device(&config, 3, dev).unwrap();

        // Single-device wiring: set the optimizer over the head's graph
        // parameters and enable training mode (what the retired
        // `Trainer::setup_head` did on the no-cluster path).
        head.graph().set_optimizer(|p| Adam::new(p, 1e-3));
        head.graph().set_training(true);

        let batch = 2;
        let seq = 4;
        let ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4, 5, 6, 7, 0], &[batch, seq], dev).unwrap(),
            false,
        );
        let pos = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3, 0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(Tensor::from_i64(&[0; 8], &[batch, seq], dev).unwrap(), false);
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions {
            dtype: DType::Float32, device: dev,
        }).unwrap();
        let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let logits = head.graph().forward_multi(&[ids, pos, tt, mask]).unwrap();
        assert_eq!(logits.shape(), vec![batch, 3]);

        let labels = Variable::new(Tensor::from_i64(&[0, 1], &[batch], dev).unwrap(), false);
        let loss = sequence_classification_loss(&logits, &labels).unwrap();
        loss.backward().unwrap();
        head.graph().step().unwrap();

        let loss_val = loss.item().unwrap();
        assert!(loss_val.is_finite(), "loss must be finite, got {loss_val}");
    }

    // ── BertForMaskedLM ──────────────────────────────────────────────

    /// `BertForMaskedLM` weight-ties its decoder to the word-embedding
    /// table, and its state_dict should reflect that: a single
    /// `bert.embeddings.word_embeddings.weight` key with shape
    /// `[vocab_size, hidden]`, **no** `cls.predictions.decoder.weight`
    /// key, plus the transform and fresh-bias keys.
    #[test]
    fn masked_lm_parameter_keys_match_hf_tied_layout() {
        let config = BertConfig::bert_base_uncased();
        let head = BertForMaskedLM::on_device(&config, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();

        // Tying contract: exactly one copy of the vocab-sized weight,
        // routed under the embeddings tag (first visit wins).
        assert!(
            keys.contains(&"bert.embeddings.word_embeddings.weight"),
            "tied weight must surface under embeddings tag: {keys:?}",
        );
        assert!(
            !keys.contains(&"cls.predictions.decoder.weight"),
            "decoder.weight must be absent (tied, dedup kept embeddings entry)",
        );

        // Pooler must not appear — MLM uses no_pooler backbone.
        assert!(
            !keys.iter().any(|k| k.starts_with("bert.pooler.")),
            "MLM must not carry pooler params",
        );

        // Transform + fresh decoder bias.
        let mut head_keys: Vec<&str> = keys
            .iter()
            .copied()
            .filter(|k| k.starts_with("cls."))
            .collect();
        head_keys.sort();
        assert_eq!(
            head_keys,
            vec![
                "cls.predictions.decoder.bias",
                "cls.predictions.transform.LayerNorm.bias",
                "cls.predictions.transform.LayerNorm.weight",
                "cls.predictions.transform.dense.bias",
                "cls.predictions.transform.dense.weight",
            ],
        );

        let by_key: std::collections::HashMap<&str, &[i64]> = expected
            .iter().map(|p| (p.key.as_str(), p.shape.as_slice())).collect();
        let v = config.vocab_size;
        let h = config.hidden_size;
        assert_eq!(by_key["bert.embeddings.word_embeddings.weight"], &[v, h]);
        assert_eq!(by_key["cls.predictions.transform.dense.weight"], &[h, h]);
        assert_eq!(by_key["cls.predictions.transform.dense.bias"],   &[h]);
        assert_eq!(by_key["cls.predictions.transform.LayerNorm.weight"], &[h]);
        assert_eq!(by_key["cls.predictions.transform.LayerNorm.bias"],   &[h]);
        assert_eq!(by_key["cls.predictions.decoder.bias"],           &[v]);
    }

    /// The tied decoder's underlying `Variable` must share its `Rc`
    /// with the embedding table's — otherwise gradient accumulation
    /// would silently split across two independent tensors and the
    /// `named_parameters()` dedup would have been a coincidence.
    /// Structural check: exactly one `[vocab_size, hidden]`-shaped
    /// Parameter in the graph. An untied decoder would double it.
    ///
    /// Uses `bert-base-uncased` config (not `tiny_bert_config`) so the
    /// shape test is unambiguous — in the tiny preset
    /// `intermediate_size == vocab_size` collides with the FFN weight
    /// shape.
    #[test]
    fn masked_lm_decoder_shares_embedding_rc() {
        let config = BertConfig::bert_base_uncased();
        let head = BertForMaskedLM::on_device(&config, Device::CPU).unwrap();

        let named = head.graph().named_parameters();
        let embed_w = named
            .iter()
            .find(|(k, _)| k == "bert.embeddings/word_embeddings.weight")
            .map(|(_, p)| p.clone())
            .expect("embeddings word_embeddings.weight must be present");
        assert_eq!(
            embed_w.variable.shape(),
            vec![config.vocab_size, config.hidden_size],
        );

        let vocab_shaped_count = named
            .iter()
            .filter(|(_, p)| p.variable.shape() == vec![config.vocab_size, config.hidden_size])
            .count();
        assert_eq!(
            vocab_shaped_count, 1,
            "exactly one [V, H]-shaped Parameter expected under tying",
        );
    }

    /// Smoke: MLM head emits `[batch, seq, vocab_size]` logits.
    #[test]
    fn masked_lm_forward_shape_smoke() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let head = BertForMaskedLM::on_device(&config, dev).unwrap();
        head.graph().eval();

        let batch = 2;
        let seq = 4;
        let ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4, 5, 6, 7, 0], &[batch, seq], dev).unwrap(),
            false,
        );
        let pos = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3, 0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(Tensor::from_i64(&[0; 8], &[batch, seq], dev).unwrap(), false);
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions {
            dtype: DType::Float32, device: dev,
        }).unwrap();
        let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let out = head.graph().forward_multi(&[ids, pos, tt, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, seq, config.vocab_size]);
    }

    /// `HasGraph` impl points to the MLM head's inner graph by
    /// reference, matching the contract the other heads satisfy.
    #[test]
    fn masked_lm_has_graph_returns_inner_graph_by_reference() {
        let config = tiny_bert_config();
        let head = BertForMaskedLM::on_device(&config, Device::CPU).unwrap();
        assert!(std::ptr::eq(head.graph(), <BertForMaskedLM as HasGraph>::graph(&head)));
    }

    /// Backward through the tied decoder must produce a gradient on
    /// the shared embedding weight — if tying were broken, the
    /// decoder path's gradient would land on a different (phantom)
    /// tensor and the embedding weight would see only its own
    /// position-lookup contribution.
    #[test]
    fn masked_lm_backward_accumulates_on_tied_weight() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let head = BertForMaskedLM::on_device(&config, dev).unwrap();
        head.graph().train();

        let batch = 1;
        let seq = 4;
        let ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4], &[batch, seq], dev).unwrap(),
            false,
        );
        let pos = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(Tensor::from_i64(&[0; 4], &[batch, seq], dev).unwrap(), false);
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions {
            dtype: DType::Float32, device: dev,
        }).unwrap();
        let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let logits = head.graph().forward_multi(&[ids, pos, tt, mask]).unwrap();
        let loss = logits.sum().unwrap();
        loss.backward().unwrap();

        let named = head.graph().named_parameters();
        let embed_w = named
            .iter()
            .find(|(k, _)| k == "bert.embeddings/word_embeddings.weight")
            .map(|(_, p)| p.clone())
            .expect("tied weight must be present");
        assert!(
            embed_w.variable.grad().is_some(),
            "tied embedding/decoder weight must receive gradient",
        );
    }

    /// HF checkpoints save `cls.predictions.decoder.weight` alongside
    /// `bert.embeddings.word_embeddings.weight` even though the two
    /// are the same tied tensor. flodl-hf emits only the first
    /// (dedup by pointer identity in `named_parameters()`), so the
    /// loader must *tolerate* the redundant decoder key rather than
    /// error on it.
    ///
    /// This is a CPU-only synthetic-safetensors test — no network,
    /// no real checkpoint needed. The existing
    /// `bert_mlm_parity.rs` live test exercises the same path
    /// end-to-end but is `#[ignore]` + `_live` (network-gated), so
    /// this unit test closes the coverage gap for default runs.
    ///
    /// Covers two assertions:
    /// 1. `load_safetensors_into_graph_with_rename_allow_unused`
    ///    accepts a blob carrying the redundant key and reports it
    ///    back in the `unused` list.
    /// 2. `load_safetensors_into_graph_with_rename` (strict) rejects
    ///    the same blob — confirming the tolerance is opt-in, not
    ///    accidental.
    #[test]
    fn mlm_loader_tolerates_redundant_tied_decoder_key() {
        use std::collections::HashMap as StdHashMap;

        use safetensors::{tensor::TensorView, Dtype};

        use crate::safetensors_io::{
            expected_from_graph, load_safetensors_into_graph_with_rename,
            load_safetensors_into_graph_with_rename_allow_unused,
        };

        let config = tiny_bert_config();
        let dev = Device::CPU;
        let head = BertForMaskedLM::on_device(&config, dev).unwrap();

        // Drive the checkpoint's key set from the graph itself. Each
        // expected key gets a zero-filled payload of the right shape;
        // the loader only validates shapes, not values.
        let expected = expected_from_graph(head.graph());
        let mut entries: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut embed_weight_shape: Vec<usize> = Vec::new();
        for p in &expected {
            let shape_usize: Vec<usize> = p.shape.iter().map(|&d| d as usize).collect();
            let numel: usize = shape_usize.iter().product();
            let payload = vec![0u8; numel * 4]; // f32 zeros
            if p.key == "bert.embeddings.word_embeddings.weight" {
                embed_weight_shape = shape_usize.clone();
            }
            entries.push((p.key.clone(), Dtype::F32, shape_usize, payload));
        }
        assert!(
            !embed_weight_shape.is_empty(),
            "bert.embeddings.word_embeddings.weight must be an expected key",
        );

        // The load-bearing addition: a redundant `cls.predictions.decoder.weight`
        // with the same shape as the tied word-embedding weight, as HF
        // checkpoints typically ship it.
        let decoder_numel: usize = embed_weight_shape.iter().product();
        entries.push((
            "cls.predictions.decoder.weight".to_string(),
            Dtype::F32,
            embed_weight_shape,
            vec![0u8; decoder_numel * 4],
        ));

        // Serialize into an in-memory safetensors blob.
        let views: StdHashMap<String, TensorView<'_>> = entries
            .iter()
            .map(|(n, d, s, b)| {
                (n.clone(), TensorView::new(*d, s.clone(), b).unwrap())
            })
            .collect();
        let bytes = safetensors::serialize(&views, &None).unwrap();

        // 1. `allow_unused` variant: accepts the redundant key and
        //    returns it in the `unused` list.
        let unused = load_safetensors_into_graph_with_rename_allow_unused(
            head.graph(),
            &bytes,
            |k| k.to_string(),
        )
        .expect("allow_unused loader must accept redundant tied decoder key");
        assert!(
            unused.iter().any(|k| k == "cls.predictions.decoder.weight"),
            "redundant decoder key must be reported in `unused`; got: {unused:?}",
        );

        // 2. Strict variant: same blob must fail, confirming the
        //    tolerance is opt-in.
        let strict_err = load_safetensors_into_graph_with_rename(
            head.graph(),
            &bytes,
            |k| k.to_string(),
        )
        .expect_err("strict loader must reject the redundant decoder key");
        let msg = strict_err.to_string();
        assert!(
            msg.contains("cls.predictions.decoder.weight"),
            "strict-loader error must name the offending key; got: {msg}",
        );
    }
