//! PyTorch parity for `DistilBertForSequenceClassification`.
//!
//! Loads the fixture from
//! `flodl-hf/scripts/parity_distilbert_seqcls.py` (`fdl flodl-hf parity
//! distilbert-seqcls`) and compares flodl's logits against the
//! HF Python reference. Fixture model:
//! `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
//! (3-label sentiment).
//!
//! `_live` — pulls real weights from the Hub. Run with `fdl test-live`.

use std::path::Path;

use safetensors::SafeTensors;

use flodl::nn::Module;
use flodl::{Device, Tensor, Variable};
use flodl_hf::models::bert::build_extended_attention_mask;
use flodl_hf::models::distilbert::DistilBertForSequenceClassification;

const FIXTURE: &str = "tests/fixtures/distilbert_seqcls_parity.safetensors";
const LOGITS_TOL: f32 = 1e-5;

mod parity_common;
use parity_common::{max_abs_diff, parse_f32, parse_i64, shape_i64};

#[test]
#[ignore = "network + ~550MB cache write"]
fn distilbert_seqcls_parity_vs_pytorch_live() {
    let dev = Device::CPU;

    let bytes = std::fs::read(Path::new(FIXTURE)).unwrap_or_else(|e| {
        panic!("reading {FIXTURE}: {e} (run `fdl flodl-hf parity distilbert-seqcls` to regenerate)")
    });
    let st = SafeTensors::deserialize(&bytes).expect("parse parity fixture");

    let ids = st.tensor("inputs.input_ids").unwrap();
    let mk = st.tensor("inputs.attention_mask").unwrap();
    let logits_ref = st.tensor("outputs.logits").unwrap();

    let input_ids = Variable::new(
        Tensor::from_i64(&parse_i64(&ids), &shape_i64(&ids), dev).unwrap(),
        false,
    );
    let mask_flat_f32: Vec<f32> = parse_i64(&mk).iter().map(|&x| x as f32).collect();
    let mask_flat = Tensor::from_f32(&mask_flat_f32, &shape_i64(&mk), dev).unwrap();
    let attention_mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

    let logits_ref_data = parse_f32(&logits_ref);
    let logits_ref_shape = shape_i64(&logits_ref);

    let model_id = "lxyuan/distilbert-base-multilingual-cased-sentiments-student";
    let clf = DistilBertForSequenceClassification::from_pretrained(model_id).unwrap();
    clf.graph().eval();

    let out = clf
        .graph()
        .forward_multi(&[input_ids, attention_mask])
        .unwrap();
    assert_eq!(out.shape(), logits_ref_shape, "logits shape mismatch");

    let actual = out.data().to_f32_vec().unwrap();
    let diff = max_abs_diff(&actual, &logits_ref_data);
    eprintln!("{model_id} logits max_abs_diff = {diff:.3e} (tol {LOGITS_TOL:.0e})");
    assert!(
        diff <= LOGITS_TOL,
        "SeqCls parity: max_abs_diff = {diff:.3e} exceeds tol {LOGITS_TOL:.0e}",
    );
}
