//! PyTorch parity for `BertModel::from_pretrained("google-bert/bert-base-uncased")`.
//!
//! Loads a committed safetensors fixture built by
//! `flodl-hf/scripts/parity_bert.py` (run via `fdl flodl-hf parity bert`) and
//! compares flodl's pooled output against the Python `BertModel` reference on
//! the same pinned inputs.
//!
//! `_live` because it pulls the real `google-bert/bert-base-uncased` weights from the
//! HuggingFace Hub (first run only; hf-hub caches). Run with `fdl test-live`.
//!
//! See `.claude/projects/memory/project_flodl_hf_bert_state.md` for the
//! tolerance rationale and the layer-isolation debug ladder if parity fails.
//!
//! Fixture layout (keys in the safetensors file):
//! - `inputs.input_ids` / `inputs.position_ids` / `inputs.token_type_ids` /
//!   `inputs.attention_mask` — all `i64 [1, 4]`
//! - `outputs.last_hidden_state` — `f32 [1, 4, 768]`
//! - `outputs.pooler_output`    — `f32 [1, 768]`

use std::path::Path;

use safetensors::SafeTensors;

use flodl::nn::Module;
use flodl::{Device, Tensor, Variable};
use flodl_hf::models::bert::{BertModel, build_extended_attention_mask};

const FIXTURE: &str = "tests/fixtures/bert_base_uncased_parity.safetensors";

/// Max absolute diff we accept between flodl and HF Python on the pooled
/// output. Same libtorch kernels under the hood; observed agreement is ~1e-6
/// on the reference host, so 1e-5 leaves 10x headroom for cross-hardware
/// scheduling noise while still flagging a real regression in any layer.
const POOLER_TOL: f32 = 1e-5;

mod parity_common;
use parity_common::{max_abs_diff, parse_f32, parse_i64, shape_i64};

#[test]
#[ignore = "network + ~440MB cache write"]
fn bert_parity_vs_pytorch_live() {
    let dev = Device::CPU;

    let fixture_bytes = std::fs::read(Path::new(FIXTURE)).unwrap_or_else(|e| {
        panic!("reading {FIXTURE}: {e} (run `fdl flodl-hf parity bert` to regenerate)")
    });
    let st = SafeTensors::deserialize(&fixture_bytes).expect("parse parity fixture");

    let input_ids_view = st.tensor("inputs.input_ids").unwrap();
    let position_ids_view = st.tensor("inputs.position_ids").unwrap();
    let token_type_ids_view = st.tensor("inputs.token_type_ids").unwrap();
    let attention_mask_view = st.tensor("inputs.attention_mask").unwrap();
    let pooler_ref_view = st.tensor("outputs.pooler_output").unwrap();

    let input_ids = Variable::new(
        Tensor::from_i64(
            &parse_i64(&input_ids_view),
            &shape_i64(&input_ids_view),
            dev,
        )
        .unwrap(),
        false,
    );
    let position_ids = Variable::new(
        Tensor::from_i64(
            &parse_i64(&position_ids_view),
            &shape_i64(&position_ids_view),
            dev,
        )
        .unwrap(),
        false,
    );
    let token_type_ids = Variable::new(
        Tensor::from_i64(
            &parse_i64(&token_type_ids_view),
            &shape_i64(&token_type_ids_view),
            dev,
        )
        .unwrap(),
        false,
    );

    // `build_extended_attention_mask` expects f32 — upcast the fixture's
    // i64 mask so future fixtures with non-all-ones masks keep working.
    let mask_flat_f32: Vec<f32> = parse_i64(&attention_mask_view)
        .iter()
        .map(|&x| x as f32)
        .collect();
    let mask_flat =
        Tensor::from_f32(&mask_flat_f32, &shape_i64(&attention_mask_view), dev).unwrap();
    let attention_mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

    let pooler_ref = parse_f32(&pooler_ref_view);
    let pooler_ref_shape = shape_i64(&pooler_ref_view);

    let graph = BertModel::from_pretrained("google-bert/bert-base-uncased").unwrap();
    graph.eval();

    let out = graph
        .forward_multi(&[input_ids, position_ids, token_type_ids, attention_mask])
        .unwrap();
    assert_eq!(out.shape(), pooler_ref_shape, "pooler shape mismatch");

    let pooler_actual = out.data().to_f32_vec().unwrap();
    let diff = max_abs_diff(&pooler_actual, &pooler_ref);
    eprintln!("pooler_output max_abs_diff = {diff:.3e} (tol {POOLER_TOL:.0e})");
    assert!(
        diff <= POOLER_TOL,
        "pooler parity: max_abs_diff = {diff:.3e} exceeds tol {POOLER_TOL:.0e}",
    );
}
