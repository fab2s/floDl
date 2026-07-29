//! Weight-space divergence, computed one way for every averaging backend.
//!
//! The ElChe convergence guard compares the divergence number *across*
//! backends (a run may switch between NCCL, CPU-reduce, and the cluster
//! wire path). If each backend carried its own copy of the norm math, a
//! one-copy tweak would silently make the numbers non-comparable. So the
//! triple is defined exactly once, here, and every backend injects its
//! pre/post tensors into it.

use crate::tensor::{Result, Tensor, TensorError};

/// Compute the weight-space divergence triple
/// `(||pre - post|| / ||post||, post_norm, pre_norm)`.
///
/// - `pre` is the pre-sync parameter snapshot. **Mutated in place**: each
///   element becomes `pre - post`, so callers treat it as round scratch.
/// - `post` is the averaged (post-sync) parameters.
///
/// `pre_norm` is captured before the in-place subtraction. The divergence
/// is guarded against a ~zero post-norm (returns 0.0). An empty parameter
/// set returns `(0.0, None, None)`.
pub(crate) fn divergence_triple(
    pre: &[Tensor],
    post: &[Tensor],
) -> Result<(f64, Option<f64>, Option<f64>)> {
    if pre.is_empty() {
        return Ok((0.0, None, None));
    }
    if pre.len() != post.len() {
        return Err(TensorError::new(&format!(
            "divergence_triple: pre.len() ({}) must equal post.len() ({})",
            pre.len(),
            post.len(),
        )));
    }

    // A bf16-wire consensus arrives as bf16 tensors (the decode lands
    // verbatim in the bf16 snapshot staging). The foreach fast route
    // needs uniform dtypes, and a whole-list upcast would materialize a
    // model-sized f32 transient per window — the exact size class the
    // pinned-decode rig regression indicted. The mixed path below runs
    // the SAME math one tensor at a time (foreach over single-element
    // slices), holding at most one upcast transient; norms are computed
    // on the f32 image so the triple keeps f32 precision (a norm READ
    // from a bf16 result tensor would round to ~3 digits). Values are
    // identical either way: the bf16 tensors hold bf16-representable
    // numbers whichever dtype carries them. This once-per-window CPU
    // work loses only foreach batching, which buys nothing here.
    let uniform_f32 = post
        .iter()
        .all(|t| t.dtype() == crate::tensor::DType::Float32);

    let (pre_sq, diff_sq, post_sq) = if uniform_f32 {
        // pre_norm BEFORE the foreach_add_list_ subtracts post from `pre`.
        let pre_norm_tensors = Tensor::foreach_norm(pre, 2.0)?;
        let mut pre_sq = 0.0f64;
        for n in &pre_norm_tensors {
            let v: f64 = n.item()?;
            pre_sq += v * v;
        }

        // pre[i] += -1 * post[i]  →  pre[i] = pre - post.
        Tensor::foreach_add_list_(pre, post, -1.0)?;
        let diff_norms = Tensor::foreach_norm(pre, 2.0)?;
        let post_norms = Tensor::foreach_norm(post, 2.0)?;

        let mut diff_sq = 0.0f64;
        for n in &diff_norms {
            let v: f64 = n.item()?;
            diff_sq += v * v;
        }
        let mut post_sq = 0.0f64;
        for n in &post_norms {
            let v: f64 = n.item()?;
            post_sq += v * v;
        }
        (pre_sq, diff_sq, post_sq)
    } else {
        let mut pre_sq = 0.0f64;
        let mut diff_sq = 0.0f64;
        let mut post_sq = 0.0f64;
        for (p, q) in pre.iter().zip(post.iter()) {
            let pre_n: f64 =
                Tensor::foreach_norm(std::slice::from_ref(p), 2.0)?[0].item()?;
            pre_sq += pre_n * pre_n;
            // One upcast transient at a time; shallow clone when
            // already f32 (mixed lists — f32 buffers among bf16 params
            // — never reach here today, but per-tensor is free).
            let q32 = if q.dtype() == crate::tensor::DType::Float32 {
                q.clone()
            } else {
                q.to_dtype(crate::tensor::DType::Float32)?
            };
            Tensor::foreach_add_list_(
                std::slice::from_ref(p),
                std::slice::from_ref(&q32),
                -1.0,
            )?;
            let diff_n: f64 =
                Tensor::foreach_norm(std::slice::from_ref(p), 2.0)?[0].item()?;
            diff_sq += diff_n * diff_n;
            let post_n: f64 =
                Tensor::foreach_norm(std::slice::from_ref(&q32), 2.0)?[0].item()?;
            post_sq += post_n * post_n;
        }
        (pre_sq, diff_sq, post_sq)
    };

    let pre_norm = pre_sq.sqrt();
    let post_norm = post_sq.sqrt();
    let divergence = if post_norm > 1e-10 {
        diff_sq.sqrt() / post_norm
    } else {
        0.0
    };

    Ok((divergence, Some(post_norm), Some(pre_norm)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{test_device, Tensor};

    fn t(data: &[f32]) -> Tensor {
        Tensor::from_f32(data, &[data.len() as i64], test_device()).unwrap()
    }

    #[test]
    fn empty_is_zero_none_none() {
        let (d, post, pre) = divergence_triple(&[], &[]).unwrap();
        assert_eq!(d, 0.0);
        assert!(post.is_none() && pre.is_none());
    }

    #[test]
    fn length_mismatch_errors() {
        let pre = [t(&[1.0])];
        let post = [t(&[1.0]), t(&[2.0])];
        assert!(divergence_triple(&pre, &post).is_err());
    }

    #[test]
    fn matches_hand_computed_triple() {
        // pre = [3,4] (norm 5), post = [0,0] guarded → divergence 0 when
        // post_norm ~ 0; use a non-degenerate post to exercise the ratio.
        let pre = [t(&[3.0, 4.0])]; // ||pre|| = 5
        let post = [t(&[6.0, 8.0])]; // ||post|| = 10; ||pre-post|| = ||[-3,-4]|| = 5
        let (d, post_n, pre_n) = divergence_triple(&pre, &post).unwrap();
        assert!((pre_n.unwrap() - 5.0).abs() < 1e-5);
        assert!((post_n.unwrap() - 10.0).abs() < 1e-5);
        assert!((d - 0.5).abs() < 1e-5, "divergence = ||pre-post||/||post|| = 5/10");
    }

    #[test]
    fn zero_post_norm_is_guarded() {
        let pre = [t(&[1.0, 1.0])];
        let post = [t(&[0.0, 0.0])];
        let (d, post_n, _) = divergence_triple(&pre, &post).unwrap();
        assert_eq!(d, 0.0, "post_norm ~ 0 must not divide");
        assert!(post_n.unwrap() < 1e-9);
    }

    /// A bf16 post list (the decode-into-request consensus under
    /// `bf16_wire` lands verbatim in the bf16 snapshot staging) must
    /// produce the SAME triple as the equivalent f32 list — the mixed
    /// path runs the identical math per tensor, and bf16-exact values
    /// leave nothing to rounding. Multi-tensor, so the per-tensor
    /// accumulation across the list is exercised too.
    #[test]
    fn bf16_post_matches_f32_reference() {
        // All values bf16-exact.
        let pre_a = [t(&[3.0, 4.0]), t(&[1.0, 2.0])];
        let post_f32 = [t(&[6.0, 8.0]), t(&[2.0, 4.0])];
        let reference = divergence_triple(&pre_a, &post_f32).unwrap();

        let pre_b = [t(&[3.0, 4.0]), t(&[1.0, 2.0])];
        let post_bf16: Vec<Tensor> = post_f32
            .iter()
            .map(|p| p.to_dtype(crate::tensor::DType::BFloat16).unwrap())
            .collect();
        let mixed = divergence_triple(&pre_b, &post_bf16).unwrap();

        assert!((mixed.0 - reference.0).abs() < 1e-6, "divergence");
        assert!(
            (mixed.1.unwrap() - reference.1.unwrap()).abs() < 1e-6,
            "post_norm"
        );
        assert!(
            (mixed.2.unwrap() - reference.2.unwrap()).abs() < 1e-6,
            "pre_norm"
        );
        // The scratch mutation contract holds on the mixed path too:
        // pre became pre - post, in f32.
        assert_eq!(pre_b[0].to_f32_vec().unwrap(), vec![-3.0, -4.0]);
    }
}
