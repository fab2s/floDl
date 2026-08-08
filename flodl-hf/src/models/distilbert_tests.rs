use super::*;
use crate::models::bert::build_extended_attention_mask;
use crate::safetensors_io::expected_from_graph;
use flodl::HasGraph;

/// 16 parameter keys every encoder layer exposes, template-formatted
/// for a given layer index.
fn expected_layer_keys(i: i64) -> Vec<String> {
    let suffixes = [
        "attention.k_lin.bias",
        "attention.k_lin.weight",
        "attention.out_lin.bias",
        "attention.out_lin.weight",
        "attention.q_lin.bias",
        "attention.q_lin.weight",
        "attention.v_lin.bias",
        "attention.v_lin.weight",
        "ffn.lin1.bias",
        "ffn.lin1.weight",
        "ffn.lin2.bias",
        "ffn.lin2.weight",
        "output_layer_norm.bias",
        "output_layer_norm.weight",
        "sa_layer_norm.bias",
        "sa_layer_norm.weight",
    ];
    suffixes
        .iter()
        .map(|s| format!("distilbert.transformer.layer.{i}.{s}"))
        .collect()
}

/// Round-trip: preset -> to_json_str -> from_json_str recovers the
/// same config. DistilBERT uses HF's short field names (`dim`,
/// `n_layers`, `n_heads`, `hidden_dim`, `activation`) — this test
/// catches any drift between the reader and writer.
#[test]
fn distilbert_config_to_json_str_round_trip() {
    let preset = DistilBertConfig::distilbert_base_uncased();
    let s = preset.to_json_str();
    let recovered = DistilBertConfig::from_json_str(&s).unwrap();
    assert_eq!(preset.to_json_str(), recovered.to_json_str());
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert_eq!(
        v.get("model_type").and_then(|x| x.as_str()),
        Some("distilbert"),
    );
    // Short-form field names present (DistilBERT-specific).
    assert!(v.get("dim").is_some());
    assert!(v.get("n_layers").is_some());
    assert!(v.get("activation").is_some());
}

/// Backbone keys: 4 embeddings + 16 × n_layers encoder keys.
/// DistilBERT has no pooler and no token_type_embeddings.
#[test]
fn distilbert_parameter_keys_match_hf_dotted_form() {
    let config = DistilBertConfig::distilbert_base_uncased();
    let graph = DistilBertModel::build(&config).unwrap();
    let expected = expected_from_graph(&graph);

    let mut keys: Vec<String> = expected.iter().map(|p| p.key.clone()).collect();
    keys.sort();

    let mut want: Vec<String> = vec![
        "distilbert.embeddings.LayerNorm.bias".into(),
        "distilbert.embeddings.LayerNorm.weight".into(),
        "distilbert.embeddings.position_embeddings.weight".into(),
        "distilbert.embeddings.word_embeddings.weight".into(),
    ];
    for i in 0..config.n_layers {
        want.extend(expected_layer_keys(i));
    }
    want.sort();

    // 4 embedding keys + 16 × 6 layer keys = 100 backbone keys.
    assert_eq!(want.len(), 100, "expected-key list size drift");
    assert_eq!(
        keys, want,
        "DistilBERT parameter keys must match HF exactly"
    );
}

/// Parameter shapes must match the distilbert/distilbert-base-uncased reference.
#[test]
fn distilbert_parameter_shapes_match_base_uncased() {
    let config = DistilBertConfig::distilbert_base_uncased();
    let graph = DistilBertModel::build(&config).unwrap();
    let expected = expected_from_graph(&graph);
    let by_key: std::collections::HashMap<&str, &[i64]> = expected
        .iter()
        .map(|p| (p.key.as_str(), p.shape.as_slice()))
        .collect();

    assert_eq!(
        by_key["distilbert.embeddings.word_embeddings.weight"],
        &[30522, 768]
    );
    assert_eq!(
        by_key["distilbert.embeddings.position_embeddings.weight"],
        &[512, 768]
    );
    assert_eq!(by_key["distilbert.embeddings.LayerNorm.weight"], &[768]);
    assert_eq!(by_key["distilbert.embeddings.LayerNorm.bias"], &[768]);

    for i in 0..config.n_layers {
        let p = format!("distilbert.transformer.layer.{i}");
        assert_eq!(by_key[&*format!("{p}.attention.q_lin.weight")], &[768, 768]);
        assert_eq!(by_key[&*format!("{p}.attention.q_lin.bias")], &[768]);
        assert_eq!(by_key[&*format!("{p}.attention.k_lin.weight")], &[768, 768]);
        assert_eq!(by_key[&*format!("{p}.attention.v_lin.weight")], &[768, 768]);
        assert_eq!(
            by_key[&*format!("{p}.attention.out_lin.weight")],
            &[768, 768]
        );
        assert_eq!(by_key[&*format!("{p}.sa_layer_norm.weight")], &[768]);
        assert_eq!(by_key[&*format!("{p}.ffn.lin1.weight")], &[3072, 768]);
        assert_eq!(by_key[&*format!("{p}.ffn.lin1.bias")], &[3072]);
        assert_eq!(by_key[&*format!("{p}.ffn.lin2.weight")], &[768, 3072]);
        assert_eq!(by_key[&*format!("{p}.ffn.lin2.bias")], &[768]);
        assert_eq!(by_key[&*format!("{p}.output_layer_norm.weight")], &[768]);
    }
}

/// Encoder stack honours `config.n_layers`.
#[test]
fn distilbert_layer_count_scales_with_config() {
    for n in [1_i64, 3, 6] {
        let config = DistilBertConfig {
            n_layers: n,
            ..DistilBertConfig::distilbert_base_uncased()
        };
        let graph = DistilBertModel::build(&config).unwrap();
        let expected = expected_from_graph(&graph);
        let total = expected.len();
        let want_total = 4 + 16 * n as usize;
        assert_eq!(
            total, want_total,
            "n_layers={n}: got {total} keys, expected {want_total}",
        );
    }
}

/// `DistilBertForSequenceClassification` adds exactly 4 head keys on
/// top of the 100-key backbone: `pre_classifier.{w,b}` and
/// `classifier.{w,b}`.
#[test]
fn seqcls_head_adds_four_keys() {
    let config = DistilBertConfig {
        num_labels: Some(3),
        ..DistilBertConfig::distilbert_base_uncased()
    };
    let head = DistilBertForSequenceClassification::on_device(&config, 3, Device::CPU).unwrap();
    let expected = expected_from_graph(head.graph());
    let keys: Vec<String> = expected.iter().map(|p| p.key.clone()).collect();

    assert_eq!(expected.len(), 100 + 4, "backbone + SeqCls head key count");
    assert!(keys.iter().any(|k| k == "pre_classifier.weight"));
    assert!(keys.iter().any(|k| k == "pre_classifier.bias"));
    assert!(keys.iter().any(|k| k == "classifier.weight"));
    assert!(keys.iter().any(|k| k == "classifier.bias"));
}

/// The seq-classification head implements [`flodl::Module`] by delegating
/// every method to its inner [`flodl::Graph`] (the trait defaults introspect
/// the head struct's own fields and would find nothing — training a
/// zero-parameter model). This delegation is what lets the head train as
/// `M` through `Trainer::builder(...).into_worker()` (see
/// `examples/distilbert_finetune_ddp.rs`).
#[test]
fn seqcls_head_module_delegates_to_graph() {
    use flodl::Module;
    let config = DistilBertConfig {
        num_labels: Some(2),
        ..DistilBertConfig::distilbert_base_uncased()
    };
    let head = DistilBertForSequenceClassification::on_device(&config, 2, Device::CPU).unwrap();
    let g = head.graph();
    assert!(
        !head.parameters().is_empty(),
        "Module::parameters must be non-empty (else the optimizer trains nothing)"
    );
    assert_eq!(
        head.parameters().len(),
        g.parameters().len(),
        "Module::parameters must delegate to the inner graph"
    );
    assert_eq!(
        head.buffers().len(),
        g.buffers().len(),
        "Module::buffers must delegate to the inner graph"
    );
    assert!(
        head.as_any()
            .and_then(|a| a.downcast_ref::<flodl::Graph>())
            .is_some(),
        "Module::as_any must present the inner Graph (DDP multi-input replica path)"
    );
}

/// `DistilBertForTokenClassification` adds 2 head keys: `classifier.{w,b}`.
#[test]
fn tokencls_head_adds_two_keys() {
    let config = DistilBertConfig {
        num_labels: Some(9),
        ..DistilBertConfig::distilbert_base_uncased()
    };
    let head = DistilBertForTokenClassification::on_device(&config, 9, Device::CPU).unwrap();
    let expected = expected_from_graph(head.graph());
    let keys: Vec<String> = expected.iter().map(|p| p.key.clone()).collect();

    assert_eq!(
        expected.len(),
        100 + 2,
        "backbone + TokenCls head key count"
    );
    assert!(keys.iter().any(|k| k == "classifier.weight"));
    assert!(keys.iter().any(|k| k == "classifier.bias"));
}

/// `DistilBertForQuestionAnswering` adds 2 head keys: `qa_outputs.{w,b}`,
/// with `classifier`-shaped `[2, dim]` output (start/end).
#[test]
fn qa_head_adds_two_keys_shape_2_dim() {
    let config = DistilBertConfig::distilbert_base_uncased();
    let head = DistilBertForQuestionAnswering::on_device(&config, Device::CPU).unwrap();
    let expected = expected_from_graph(head.graph());
    let by_key: std::collections::HashMap<&str, &[i64]> = expected
        .iter()
        .map(|p| (p.key.as_str(), p.shape.as_slice()))
        .collect();

    assert_eq!(expected.len(), 100 + 2, "backbone + QA head key count");
    assert_eq!(by_key["qa_outputs.weight"], &[2, 768]);
    assert_eq!(by_key["qa_outputs.bias"], &[2]);
}

/// Seqcls head errors if `num_labels` can't be inferred from config.
#[test]
fn seqcls_num_labels_required() {
    let config = DistilBertConfig::distilbert_base_uncased();
    let err = DistilBertForSequenceClassification::num_labels_from_config(&config).unwrap_err();
    assert!(format!("{err}").contains("num_labels"), "got: {err}");
}

#[test]
fn parses_distilbert_base_uncased_config() {
    // Real config.json from distilbert/distilbert-base-uncased, pinned
    // as a literal so the test is offline.
    let json = r#"{
            "activation": "gelu",
            "architectures": ["DistilBertForMaskedLM"],
            "attention_dropout": 0.1,
            "dim": 768,
            "dropout": 0.1,
            "hidden_dim": 3072,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "model_type": "distilbert",
            "n_heads": 12,
            "n_layers": 6,
            "pad_token_id": 0,
            "qa_dropout": 0.1,
            "seq_classif_dropout": 0.2,
            "sinusoidal_pos_embds": false,
            "tie_weights_": true,
            "vocab_size": 30522
        }"#;
    let cfg = DistilBertConfig::from_json_str(json).unwrap();
    assert_eq!(cfg.vocab_size, 30522);
    assert_eq!(cfg.dim, 768);
    assert_eq!(cfg.n_layers, 6);
    assert_eq!(cfg.n_heads, 12);
    assert_eq!(cfg.hidden_dim, 3072);
    assert_eq!(cfg.max_position_embeddings, 512);
    assert_eq!(cfg.pad_token_id, 0);
    assert!((cfg.dropout - 0.1).abs() < 1e-12);
    assert!((cfg.attention_dropout - 0.1).abs() < 1e-12);
    assert!((cfg.qa_dropout - 0.1).abs() < 1e-12);
    assert!((cfg.seq_classif_dropout - 0.2).abs() < 1e-12);
    assert!(!cfg.sinusoidal_pos_embds);
    assert!((cfg.layer_norm_eps - 1e-12).abs() < 1e-18);
    assert!(cfg.num_labels.is_none());
    assert!(cfg.id2label.is_none());
}

#[test]
fn parses_cased_distilled_squad_config() {
    // `sinusoidal_pos_embds = true` — verify we capture it without
    // tripping over the flag. (Cosmetic at load time; see doc on
    // the field.)
    let json = r#"{
            "activation": "gelu",
            "architectures": ["DistilBertForQuestionAnswering"],
            "attention_dropout": 0.1,
            "dim": 768,
            "dropout": 0.1,
            "hidden_dim": 3072,
            "max_position_embeddings": 512,
            "model_type": "distilbert",
            "n_heads": 12,
            "n_layers": 6,
            "pad_token_id": 0,
            "qa_dropout": 0.1,
            "seq_classif_dropout": 0.2,
            "sinusoidal_pos_embds": true,
            "vocab_size": 28996
        }"#;
    let cfg = DistilBertConfig::from_json_str(json).unwrap();
    assert_eq!(cfg.vocab_size, 28996);
    assert!(cfg.sinusoidal_pos_embds);
}

#[test]
fn parses_finetuned_seqcls_config() {
    // 3-class sentiment head from lxyuan's student. Exercises the
    // num_labels + id2label derivation paths.
    let json = r#"{
            "activation": "gelu",
            "architectures": ["DistilBertForSequenceClassification"],
            "attention_dropout": 0.1,
            "dim": 768,
            "dropout": 0.1,
            "hidden_dim": 3072,
            "id2label": {"0": "positive", "1": "neutral", "2": "negative"},
            "label2id": {"positive": 0, "neutral": 1, "negative": 2},
            "max_position_embeddings": 512,
            "model_type": "distilbert",
            "n_heads": 12,
            "n_layers": 6,
            "pad_token_id": 0,
            "qa_dropout": 0.1,
            "seq_classif_dropout": 0.2,
            "sinusoidal_pos_embds": false,
            "vocab_size": 119547
        }"#;
    let cfg = DistilBertConfig::from_json_str(json).unwrap();
    assert_eq!(cfg.vocab_size, 119547);
    assert_eq!(cfg.num_labels, Some(3));
    let labels = cfg.id2label.unwrap();
    assert_eq!(labels, vec!["positive", "neutral", "negative"]);
}

#[test]
fn missing_required_field_errors() {
    // Drop `n_layers` — must surface a clear error.
    let json = r#"{
            "vocab_size": 30522, "dim": 768, "n_heads": 12,
            "hidden_dim": 3072, "max_position_embeddings": 512
        }"#;
    let err = DistilBertConfig::from_json_str(json).unwrap_err();
    assert!(format!("{err}").contains("n_layers"), "got: {err}");
}

// ── DistilBertForMaskedLM ────────────────────────────────────────

/// `DistilBertForMaskedLM` ties its projector weight to the
/// word-embedding table. State_dict must carry
/// `distilbert.embeddings.word_embeddings.weight` but **not**
/// `vocab_projector.weight`; the flat LM head contributes three
/// tagged nodes.
#[test]
fn masked_lm_parameter_keys_match_hf_tied_layout() {
    let config = DistilBertConfig::distilbert_base_uncased();
    let head = DistilBertForMaskedLM::on_device(&config, Device::CPU).unwrap();
    let expected = expected_from_graph(head.graph());
    let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();

    assert!(
        keys.contains(&"distilbert.embeddings.word_embeddings.weight"),
        "tied weight must surface under embeddings tag: {keys:?}",
    );
    assert!(
        !keys.contains(&"vocab_projector.weight"),
        "vocab_projector.weight must be absent (tied, dedup kept embeddings entry)",
    );

    // No pooler in DistilBERT at all.
    assert!(
        !keys.iter().any(|k| k.contains("pooler")),
        "DistilBERT carries no pooler",
    );

    let mut head_keys: Vec<&str> = keys
        .iter()
        .copied()
        .filter(|k| {
            k.starts_with("vocab_transform.")
                || k.starts_with("vocab_layer_norm.")
                || k.starts_with("vocab_projector.")
        })
        .collect();
    head_keys.sort();
    assert_eq!(
        head_keys,
        vec![
            "vocab_layer_norm.bias",
            "vocab_layer_norm.weight",
            "vocab_projector.bias",
            "vocab_transform.bias",
            "vocab_transform.weight",
        ],
    );

    let by_key: std::collections::HashMap<&str, &[i64]> = expected
        .iter()
        .map(|p| (p.key.as_str(), p.shape.as_slice()))
        .collect();
    let v = config.vocab_size;
    let d = config.dim;
    assert_eq!(
        by_key["distilbert.embeddings.word_embeddings.weight"],
        &[v, d]
    );
    assert_eq!(by_key["vocab_transform.weight"], &[d, d]);
    assert_eq!(by_key["vocab_transform.bias"], &[d]);
    assert_eq!(by_key["vocab_layer_norm.weight"], &[d]);
    assert_eq!(by_key["vocab_layer_norm.bias"], &[d]);
    assert_eq!(by_key["vocab_projector.bias"], &[v]);
}

/// Structural tying check: exactly one `[vocab, dim]`-shaped
/// Parameter in the graph.
#[test]
fn masked_lm_projector_shares_embedding_rc() {
    let config = DistilBertConfig::distilbert_base_uncased();
    let head = DistilBertForMaskedLM::on_device(&config, Device::CPU).unwrap();

    let named = head.graph().named_parameters();
    let embed_w = named
        .iter()
        .find(|(k, _)| k == "distilbert.embeddings/word_embeddings.weight")
        .map(|(_, p)| p.clone())
        .expect("embeddings word_embeddings.weight must be present");
    assert_eq!(
        embed_w.variable.shape(),
        vec![config.vocab_size, config.dim],
    );

    let vocab_shaped_count = named
        .iter()
        .filter(|(_, p)| p.variable.shape() == vec![config.vocab_size, config.dim])
        .count();
    assert_eq!(
        vocab_shaped_count, 1,
        "exactly one [V, dim]-shaped Parameter expected under tying",
    );
}

/// Smoke: DistilBERT MLM head emits `[batch, seq, vocab_size]`
/// logits. Batch and seq kept tiny so the forward pass stays
/// cheap on the full distilbert-base config.
#[test]
fn masked_lm_forward_shape_smoke() {
    let config = DistilBertConfig::distilbert_base_uncased();
    let dev = Device::CPU;
    let head = DistilBertForMaskedLM::on_device(&config, dev).unwrap();
    head.graph().eval();

    let batch = 1;
    let seq = 4;
    let ids = Variable::new(
        Tensor::from_i64(&[101, 200, 300, 102], &[batch, seq], dev).unwrap(),
        false,
    );
    let mask_flat = Tensor::ones(
        &[batch, seq],
        TensorOptions {
            dtype: DType::Float32,
            device: dev,
        },
    )
    .unwrap();
    let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

    let out = head.graph().forward_multi(&[ids, mask]).unwrap();
    assert_eq!(out.shape(), vec![batch, seq, config.vocab_size]);
}

/// `HasGraph` impl points to the MLM head's inner graph by reference.
#[test]
fn masked_lm_has_graph_returns_inner_graph_by_reference() {
    let config = DistilBertConfig::distilbert_base_uncased();
    let head = DistilBertForMaskedLM::on_device(&config, Device::CPU).unwrap();
    assert!(std::ptr::eq(
        head.graph(),
        <DistilBertForMaskedLM as HasGraph>::graph(&head),
    ));
}

/// Backward through the tied projector must produce a gradient on
/// the shared embedding weight.
#[test]
fn masked_lm_backward_accumulates_on_tied_weight() {
    let config = DistilBertConfig::distilbert_base_uncased();
    let dev = Device::CPU;
    let head = DistilBertForMaskedLM::on_device(&config, dev).unwrap();
    head.graph().train();

    let batch = 1;
    let seq = 4;
    let ids = Variable::new(
        Tensor::from_i64(&[101, 200, 300, 102], &[batch, seq], dev).unwrap(),
        false,
    );
    let mask_flat = Tensor::ones(
        &[batch, seq],
        TensorOptions {
            dtype: DType::Float32,
            device: dev,
        },
    )
    .unwrap();
    let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

    let logits = head.graph().forward_multi(&[ids, mask]).unwrap();
    let loss = logits.sum().unwrap();
    loss.backward().unwrap();

    let named = head.graph().named_parameters();
    let embed_w = named
        .iter()
        .find(|(k, _)| k == "distilbert.embeddings/word_embeddings.weight")
        .map(|(_, p)| p.clone())
        .expect("tied weight must be present");
    assert!(
        embed_w.variable.grad().is_some(),
        "tied embedding/projector weight must receive gradient",
    );
}

#[test]
fn preset_roundtrips_through_parser() {
    // Sanity: the preset values are the same as a fresh parse of
    // the canonical config.
    let preset = DistilBertConfig::distilbert_base_uncased();
    // Parse a stripped-down config asserting the same values.
    let json = r#"{
            "vocab_size": 30522, "dim": 768, "n_layers": 6, "n_heads": 12,
            "hidden_dim": 3072, "max_position_embeddings": 512, "pad_token_id": 0
        }"#;
    let parsed = DistilBertConfig::from_json_str(json).unwrap();
    assert_eq!(preset.vocab_size, parsed.vocab_size);
    assert_eq!(preset.dim, parsed.dim);
    assert_eq!(preset.n_layers, parsed.n_layers);
    assert_eq!(preset.n_heads, parsed.n_heads);
    assert_eq!(preset.hidden_dim, parsed.hidden_dim);
    assert_eq!(preset.pad_token_id, parsed.pad_token_id);
}
