    use super::*;
    use crate::safetensors_io::expected_from_graph;

    /// Round-trip: preset -> to_json_str -> from_json_str recovers the
    /// same config. DeBERTa-v2's writer emits a wide set of invariants
    /// the parser validates (`relative_attention: true`,
    /// `share_att_key: true`, `position_biased_input: false`,
    /// `pos_att_type: "p2c|c2p"`, `norm_rel_ebd: "layer_norm"`,
    /// `legacy: false`, `type_vocab_size: 0`) — this test catches any
    /// drift between the emitted JSON and what the parser accepts.
    #[test]
    fn deberta_v2_config_to_json_str_round_trip() {
        let preset = DebertaV2Config::deberta_v3_base();
        let s = preset.to_json_str();
        let recovered = DebertaV2Config::from_json_str(&s).unwrap();
        assert_eq!(preset.to_json_str(), recovered.to_json_str());
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(
            v.get("model_type").and_then(|x| x.as_str()),
            Some("deberta-v2"),
        );
        // Validator-trip fields the parser rejects without them.
        assert_eq!(
            v.get("relative_attention").and_then(|x| x.as_bool()),
            Some(true),
        );
        assert_eq!(v.get("share_att_key").and_then(|x| x.as_bool()), Some(true));
        assert_eq!(
            v.get("position_biased_input").and_then(|x| x.as_bool()),
            Some(false),
        );
        assert_eq!(
            v.get("pos_att_type").and_then(|x| x.as_str()),
            Some("p2c|c2p"),
        );
        assert_eq!(
            v.get("norm_rel_ebd").and_then(|x| x.as_str()),
            Some("layer_norm"),
        );
        assert_eq!(v.get("legacy").and_then(|x| x.as_bool()), Some(false));
        assert_eq!(v.get("type_vocab_size").and_then(|x| x.as_i64()), Some(0));
        // pooler_hidden_act present (separate field from hidden_act).
        assert!(v.get("pooler_hidden_act").is_some());
    }

    fn mini_config() -> DebertaV2Config {
        // Small dims so tests run fast while still exercising the
        // disentangled-attention + rel-embeddings structure.
        DebertaV2Config {
            vocab_size: 64,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: 16,
            layer_norm_eps: 1e-7,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            pad_token_id: Some(0),
            position_buckets: 4,
            max_relative_positions: 8,
            hidden_act: GeluApprox::Exact,
            pooler_hidden_act: GeluApprox::Exact,
            num_labels: None,
            id2label: None,
            architectures: None,
        }
    }

    fn v3_base_config_json() -> &'static str {
        // microsoft/deberta-v3-base actual config.json (pinned).
        r#"{
            "model_type": "deberta-v2",
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "relative_attention": true,
            "position_buckets": 256,
            "norm_rel_ebd": "layer_norm",
            "share_att_key": true,
            "pos_att_type": "p2c|c2p",
            "layer_norm_eps": 1e-7,
            "max_relative_positions": -1,
            "position_biased_input": false,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 0,
            "vocab_size": 128100
        }"#
    }

    /// Round-trip the real v3-base config through `from_json_str` and
    /// check every load-bearing field. `max_relative_positions` must
    /// resolve from `-1` to `max_position_embeddings`.
    #[test]
    fn from_json_str_parses_v3_base() {
        let cfg = DebertaV2Config::from_json_str(v3_base_config_json()).unwrap();
        assert_eq!(cfg.vocab_size, 128_100);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.position_buckets, 256);
        assert_eq!(
            cfg.max_relative_positions, 512,
            "-1 must resolve to max_position_embeddings",
        );
        assert_eq!(cfg.layer_norm_eps, 1e-7);
    }

    /// Unsupported knobs must surface specific errors with the
    /// offending field name so users targeting v1 / experimental
    /// configs can file a clear bug.
    #[test]
    fn from_json_str_rejects_share_att_key_false() {
        let json = v3_base_config_json().replace(
            "\"share_att_key\": true",
            "\"share_att_key\": false",
        );
        let err = DebertaV2Config::from_json_str(&json).unwrap_err().to_string();
        assert!(err.contains("share_att_key"), "got: {err}");
    }

    #[test]
    fn from_json_str_rejects_position_biased_input_true() {
        let json = v3_base_config_json().replace(
            "\"position_biased_input\": false",
            "\"position_biased_input\": true",
        );
        let err = DebertaV2Config::from_json_str(&json).unwrap_err().to_string();
        assert!(err.contains("position_biased_input"), "got: {err}");
    }

    #[test]
    fn from_json_str_rejects_missing_p2c() {
        let json = v3_base_config_json().replace(
            "\"pos_att_type\": \"p2c|c2p\"",
            "\"pos_att_type\": \"c2p\"",
        );
        let err = DebertaV2Config::from_json_str(&json).unwrap_err().to_string();
        assert!(err.contains("pos_att_type"), "got: {err}");
    }

    #[test]
    fn from_json_str_rejects_legacy_mlm() {
        let json = v3_base_config_json().replace(
            "\"type_vocab_size\": 0",
            "\"type_vocab_size\": 0, \"legacy\": true",
        );
        let err = DebertaV2Config::from_json_str(&json).unwrap_err().to_string();
        assert!(err.contains("legacy"), "got: {err}");
    }

    #[test]
    fn from_json_str_rejects_token_types() {
        let json = v3_base_config_json().replace(
            "\"type_vocab_size\": 0",
            "\"type_vocab_size\": 2",
        );
        let err = DebertaV2Config::from_json_str(&json).unwrap_err().to_string();
        assert!(err.contains("type_vocab_size"), "got: {err}");
    }

    /// Backbone parameter keys must match HF state_dict exactly.
    #[test]
    fn backbone_parameter_keys_match_hf() {
        let graph = DebertaV2Model::on_device(&mini_config(), Device::CPU).unwrap();
        let expected = expected_from_graph(&graph);
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();

        for must_have in &[
            "deberta.embeddings.word_embeddings.weight",
            "deberta.embeddings.LayerNorm.weight",
            "deberta.encoder.rel_embeddings.weight",
            "deberta.encoder.LayerNorm.weight",
            "deberta.encoder.layer.0.attention.self.query_proj.weight",
            "deberta.encoder.layer.0.attention.self.key_proj.weight",
            "deberta.encoder.layer.0.attention.self.value_proj.weight",
            "deberta.encoder.layer.0.attention.output.dense.weight",
            "deberta.encoder.layer.0.attention.output.LayerNorm.weight",
            "deberta.encoder.layer.0.intermediate.dense.weight",
            "deberta.encoder.layer.0.output.dense.weight",
            "deberta.encoder.layer.0.output.LayerNorm.weight",
            "deberta.encoder.layer.1.attention.self.query_proj.weight",
        ] {
            assert!(
                keys.iter().any(|k| k == must_have),
                "missing HF key {must_have} in {keys:?}",
            );
        }
    }

    /// v3-base-specific negatives: no position_embeddings (position_biased_input=false)
    /// and no token_type_embeddings (type_vocab_size=0).
    #[test]
    fn backbone_has_no_absolute_position_or_token_type_embeddings() {
        let graph = DebertaV2Model::on_device(&mini_config(), Device::CPU).unwrap();
        let expected = expected_from_graph(&graph);
        for p in &expected {
            assert!(
                !p.key.contains("position_embeddings"),
                "v3 has no absolute position embeddings; got {}", p.key,
            );
            assert!(
                !p.key.contains("token_type_embeddings"),
                "v3 has no token-type embeddings; got {}", p.key,
            );
        }
    }

    /// MLM head: tied `[V, H]` weight surfaces once under the embeddings
    /// tag, fresh `[V]` decoder bias surfaces under `lm_predictions.lm_head.bias`.
    #[test]
    fn mlm_head_ties_weight_and_emits_separate_bias() {
        let cfg = mini_config();
        let head = DebertaV2ForMaskedLM::on_device(&cfg, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();

        assert!(
            keys.contains(&"deberta.embeddings.word_embeddings.weight"),
            "tied word_embeddings must surface once under embeddings tag: {keys:?}",
        );
        // No "decoder." infix — HF stores the tied-decoder bias directly
        // as `lm_predictions.lm_head.bias`.
        assert!(
            keys.contains(&"lm_predictions.lm_head.bias"),
            "tied-decoder bias must surface as lm_predictions.lm_head.bias: {keys:?}",
        );
        assert!(
            !keys.iter().any(|k| k.contains("lm_predictions.lm_head.decoder")),
            "no .decoder. key should appear in MLM head: {keys:?}",
        );
        assert!(
            keys.contains(&"lm_predictions.lm_head.dense.weight"),
            "MLM transform dense must surface: {keys:?}",
        );
        assert!(
            keys.contains(&"lm_predictions.lm_head.LayerNorm.weight"),
            "MLM transform LayerNorm must surface: {keys:?}",
        );

        // Only one [V, H]-shaped Parameter (the tied word_embedding).
        let named = head.graph().named_parameters();
        let v_h_shaped = named
            .iter()
            .filter(|(_, p)| p.variable.shape() == vec![cfg.vocab_size, cfg.hidden_size])
            .count();
        assert_eq!(
            v_h_shaped, 1,
            "exactly one [V, H]-shaped parameter expected (tied)",
        );
    }

    /// End-to-end forward shape smoke for the backbone.
    #[test]
    fn backbone_forward_shape() {
        let cfg = mini_config();
        let dev = Device::CPU;
        let graph = DebertaV2Model::on_device(&cfg, dev).unwrap();
        graph.eval();

        let batch = 1;
        let seq = 4;
        let input_ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4], &[batch, seq], dev).unwrap(),
            false,
        );
        let mask = Variable::new(
            Tensor::ones(&[batch, seq], TensorOptions {
                dtype: DType::Int64, device: dev,
            }).unwrap(),
            false,
        );
        let out = graph.forward_multi(&[input_ids, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, seq, cfg.hidden_size]);
    }

    /// Backbone forward runs end-to-end in f16: cast every parameter
    /// to half-precision, then run the same shape smoke. Guards the
    /// `build_deberta_attention_mask` + embeddings mask-gate dtype
    /// threading — without those, the additive bias / mask multiply
    /// would be Float while the hidden states are Half, tripping
    /// libtorch's same-dtype check.
    #[test]
    fn backbone_forward_shape_f16() {
        use flodl::nn::{cast_parameters, Module};

        let cfg = mini_config();
        let dev = Device::CPU;
        let graph = DebertaV2Model::on_device(&cfg, dev).unwrap();
        cast_parameters(&graph.parameters(), DType::Float16);
        graph.eval();

        let batch = 1;
        let seq = 4;
        let input_ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4], &[batch, seq], dev).unwrap(),
            false,
        );
        let mask = Variable::new(
            Tensor::ones(&[batch, seq], TensorOptions {
                dtype: DType::Int64, device: dev,
            }).unwrap(),
            false,
        );
        let out = graph.forward_multi(&[input_ids, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, seq, cfg.hidden_size]);
        assert_eq!(out.data().dtype(), DType::Float16,
            "forward output must remain f16 throughout the encoder");
    }

    /// Sequence-classification head: [B, num_labels] output.
    #[test]
    fn seqcls_head_forward_shape() {
        let mut cfg = mini_config();
        cfg.num_labels = Some(3);
        let dev = Device::CPU;
        let head = DebertaV2ForSequenceClassification::on_device(&cfg, 3, dev).unwrap();
        head.graph().eval();

        let batch = 2;
        let seq = 4;
        let ids_data: Vec<i64> = (1..=(batch * seq)).collect();
        let input_ids = Variable::new(
            Tensor::from_i64(&ids_data, &[batch, seq], dev).unwrap(),
            false,
        );
        let mask = Variable::new(
            Tensor::ones(&[batch, seq], TensorOptions {
                dtype: DType::Int64, device: dev,
            }).unwrap(),
            false,
        );
        let out = head.graph().forward_multi(&[input_ids, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, 3]);
    }

    /// MLM head: [B, S, V] logits via the tied-decoder matmul path.
    #[test]
    fn mlm_head_forward_shape() {
        let cfg = mini_config();
        let dev = Device::CPU;
        let head = DebertaV2ForMaskedLM::on_device(&cfg, dev).unwrap();
        head.graph().eval();

        let batch = 1;
        let seq = 3;
        let input_ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let mask = Variable::new(
            Tensor::ones(&[batch, seq], TensorOptions {
                dtype: DType::Int64, device: dev,
            }).unwrap(),
            false,
        );
        let out = head.graph().forward_multi(&[input_ids, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, seq, cfg.vocab_size]);
    }

    /// `build_deberta_attention_mask` produces the expected `[B, 1, S, S]`
    /// additive form from a `[B, S]` flat mask.
    #[test]
    fn deberta_mask_shape_and_values() {
        let dev = Device::CPU;
        let flat = Tensor::from_f32(&[1.0, 1.0, 0.0], &[1, 3], dev).unwrap();
        let extended = build_deberta_attention_mask(&flat, DType::Float32).unwrap();
        assert_eq!(extended.shape(), vec![1, 1, 3, 3]);
        let data = extended.to_f32_vec().unwrap();
        // Row-major [1, 1, 3, 3]: (q, k)
        // Valid tokens: 0, 1.  Padding: 2.
        // Attend allowed <=> q AND k both valid.
        let at = |q: usize, k: usize| data[q * 3 + k];
        assert_eq!(at(0, 0), 0.0);
        assert_eq!(at(0, 1), 0.0);
        assert!(at(0, 2) < -1e30, "q=0 attending to pad key blocked");
        assert_eq!(at(1, 1), 0.0);
        assert!(at(1, 2) < -1e30, "q=1 attending to pad key blocked");
        assert!(at(2, 0) < -1e30, "pad query blocks its own row");
        assert!(at(2, 2) < -1e30, "pad attending to pad blocked");
    }
