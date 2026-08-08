//! Round-trip: `HfTokenizer::from_pretrained("bert-base-uncased")` must
//! produce the input_ids that `parity_bert.py` pinned into the BERT parity
//! fixture. Closes the loop between step 4 (forward parity on hardcoded
//! token IDs) and step 5 (tokenizer that reproduces those IDs from text).
//!
//! `_live` because it pulls `tokenizer.json` from the HuggingFace Hub.
//! Run with `fdl test-live`.

use std::path::Path;

use safetensors::SafeTensors;

use flodl_hf::tokenizer::HfTokenizer;

const FIXTURE: &str = "tests/fixtures/bert_base_uncased_parity.safetensors";

#[test]
#[ignore = "network"]
fn bert_tokenizer_matches_parity_fixture_live() {
    let bytes =
        std::fs::read(Path::new(FIXTURE)).unwrap_or_else(|e| panic!("reading {FIXTURE}: {e}"));
    let st = SafeTensors::deserialize(&bytes).expect("parse parity fixture");

    let view = st.tensor("inputs.input_ids").unwrap();
    let expected_ids: Vec<i64> = view
        .data()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected_shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();

    let tok = HfTokenizer::from_pretrained("bert-base-uncased").unwrap();
    let enc = tok.encode(&["hello world"]).unwrap();

    assert_eq!(enc.input_ids.shape(), expected_shape, "shape mismatch");

    let actual_ids = enc.input_ids.data().to_i64_vec().unwrap();
    assert_eq!(
        actual_ids, expected_ids,
        "token ids diverge from parity fixture",
    );

    // Attention mask and token-type-ids should also match the fixture's
    // hardcoded values for a single unpadded sentence.
    let attn_view = st.tensor("inputs.attention_mask").unwrap();
    let expected_attn: Vec<i64> = attn_view
        .data()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let actual_attn = enc.attention_mask.data().to_i64_vec().unwrap();
    assert_eq!(actual_attn, expected_attn, "attention_mask mismatch");

    let tt_view = st.tensor("inputs.token_type_ids").unwrap();
    let expected_tt: Vec<i64> = tt_view
        .data()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let actual_tt = enc.token_type_ids.data().to_i64_vec().unwrap();
    assert_eq!(actual_tt, expected_tt, "token_type_ids mismatch");
}
