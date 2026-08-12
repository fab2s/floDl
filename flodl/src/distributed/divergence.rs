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
        let mut sums = (0.0f64, 0.0f64, 0.0f64);
        for (p, q) in pre.iter().zip(post.iter()) {
            observe_pair(p, q, &mut sums)?;
        }
        sums
    };

    Ok(triple_from_sums(pre_sq, diff_sq, post_sq))
}

/// Accumulate one `(pre_f32, post)` pair's contribution into the three
/// running sums `(pre_sq, diff_sq, post_sq)`. `pre_f32` must be an f32
/// tensor and is MUTATED into `pre - post` (the same round-scratch
/// contract as [`divergence_triple`]). `post` may be bf16: one upcast
/// transient at a time, norms computed on the f32 image so the triple
/// keeps f32 precision (a norm READ from a bf16 result tensor would
/// round to ~3 digits).
///
/// This is the per-pair definition both [`divergence_triple`]'s mixed
/// path and [`DivergenceAccum`] share, so the streaming and the
/// list-at-once callers can never drift apart.
fn observe_pair(pre_f32: &Tensor, post: &Tensor, sums: &mut (f64, f64, f64)) -> Result<()> {
    let pre_n: f64 = Tensor::foreach_norm(std::slice::from_ref(pre_f32), 2.0)?[0].item()?;
    sums.0 += pre_n * pre_n;
    // Bring `post` to the pre-image's device + f32. Same-device f32 is
    // a shallow clone (the production CPU-averaging pair, and the
    // NCCL path's device-resident triple, both stay move-free); the
    // cross-device leg exists for [`DivergenceAccum`], whose transient
    // is CPU by design while a caller's post may live elsewhere.
    let q32 = if post.device() == pre_f32.device() && post.dtype() == crate::tensor::DType::Float32
    {
        post.clone()
    } else {
        let moved = if post.device() == pre_f32.device() {
            post.clone()
        } else {
            post.to_device(pre_f32.device())?
        };
        if moved.dtype() == crate::tensor::DType::Float32 {
            moved
        } else {
            moved.to_dtype(crate::tensor::DType::Float32)?
        }
    };
    Tensor::foreach_add_list_(
        std::slice::from_ref(pre_f32),
        std::slice::from_ref(&q32),
        -1.0,
    )?;
    let diff_n: f64 = Tensor::foreach_norm(std::slice::from_ref(pre_f32), 2.0)?[0].item()?;
    sums.1 += diff_n * diff_n;
    let post_n: f64 = Tensor::foreach_norm(std::slice::from_ref(&q32), 2.0)?[0].item()?;
    sums.2 += post_n * post_n;
    Ok(())
}

/// Finish the triple from the three accumulated sums — the single
/// definition of the ratio and its ~zero-post-norm guard.
fn triple_from_sums(pre_sq: f64, diff_sq: f64, post_sq: f64) -> (f64, Option<f64>, Option<f64>) {
    let pre_norm = pre_sq.sqrt();
    let post_norm = post_sq.sqrt();
    let divergence = if post_norm > 1e-10 {
        diff_sq.sqrt() / post_norm
    } else {
        0.0
    };
    (divergence, Some(post_norm), Some(pre_norm))
}

/// Streaming counterpart of [`divergence_triple`]: same triple, one
/// tensor pair at a time, holding at most ONE pre-image transient (the
/// current tensor's f32 copy) instead of a resident model-sized
/// scratch. This is what retired `param_bridge`'s `pre_scratch` (a full
/// f32 CPU model copy per rank, steady for the whole run, whose only
/// output was these three scalars).
///
/// Protocol per tensor, in decode order: [`Self::capture_pre`] BEFORE
/// the consensus decode overwrites the staging tensor (under
/// decode-into-request the staging IS the destination, so the pre-image
/// is destroyed by the decode), then [`Self::observe_post`] with the
/// decoded consensus. [`Self::finish`] yields the exact triple;
/// [`Self::finish_keep_local`] yields the keep-local round's triple
/// `(0, pre_norm, pre_norm)` — pre == post by definition there, and the
/// post-side sums may hold a zero-mass reply's meaningless payloads, so
/// only the pre sums are trusted.
#[derive(Debug)]
pub(crate) struct DivergenceAccum {
    sums: (f64, f64, f64),
    pairs: usize,
    /// The current tensor's pre-image (f32 CPU copy), between
    /// `capture_pre` and its matching `observe_post`.
    held: Option<Tensor>,
}

impl DivergenceAccum {
    pub(crate) fn new() -> Self {
        Self {
            sums: (0.0, 0.0, 0.0),
            pairs: 0,
            held: None,
        }
    }

    /// Copy `pre`'s current values into the held f32 transient. Must be
    /// called before the decode lands in `pre`'s storage. Upcasts bf16
    /// exactly (same explicit-f32 + `copy_` shape the retired scratch
    /// used).
    pub(crate) fn capture_pre(&mut self, pre: &Tensor) -> Result<()> {
        if self.held.is_some() {
            return Err(TensorError::new(
                "DivergenceAccum::capture_pre: previous pre-image not yet observed",
            ));
        }
        let copy = Tensor::zeros(
            &pre.shape(),
            crate::tensor::TensorOptions {
                dtype: crate::tensor::DType::Float32,
                device: crate::tensor::Device::CPU,
            },
        )?;
        copy.copy_(pre, false)?;
        self.held = Some(copy);
        Ok(())
    }

    /// Fold the held pre-image and the decoded `post` into the sums,
    /// dropping the transient.
    pub(crate) fn observe_post(&mut self, post: &Tensor) -> Result<()> {
        let Some(held) = self.held.take() else {
            return Err(TensorError::new(
                "DivergenceAccum::observe_post: no captured pre-image",
            ));
        };
        observe_pair(&held, post, &mut self.sums)?;
        self.pairs += 1;
        Ok(())
    }

    /// The exact triple over every observed pair. Empty (no pairs)
    /// mirrors [`divergence_triple`]'s empty contract: `(0, None, None)`.
    pub(crate) fn finish(self) -> (f64, Option<f64>, Option<f64>) {
        if self.pairs == 0 {
            return (0.0, None, None);
        }
        triple_from_sums(self.sums.0, self.sums.1, self.sums.2)
    }

    /// Keep-local round (zero realized mass — the caller keeps its own
    /// tensors, so pre == post): `(0, pre_norm, pre_norm)`, trusting
    /// only the pre-side sums (the reply's payloads were meaningless).
    pub(crate) fn finish_keep_local(self) -> (f64, Option<f64>, Option<f64>) {
        if self.pairs == 0 {
            return (0.0, None, None);
        }
        let pre_norm = self.sums.0.sqrt();
        (0.0, Some(pre_norm), Some(pre_norm))
    }
}

/// Exact global L2 norm of a tensor list, computed on the f32 image
/// (per-tensor upcast when needed — same precision contract as the
/// triple). Used by the all-idle path, where no reduce runs and the
/// triple degenerates to `(0, n, n)` without any pre-image copy.
pub(crate) fn exact_norm(tensors: &[Tensor]) -> Result<Option<f64>> {
    if tensors.is_empty() {
        return Ok(None);
    }
    let mut sq = 0.0f64;
    for t in tensors {
        let t32 = if t.dtype() == crate::tensor::DType::Float32 {
            t.clone()
        } else {
            t.to_dtype(crate::tensor::DType::Float32)?
        };
        let n: f64 = Tensor::foreach_norm(std::slice::from_ref(&t32), 2.0)?[0].item()?;
        sq += n * n;
    }
    Ok(Some(sq.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device};

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
        assert!(
            (d - 0.5).abs() < 1e-5,
            "divergence = ||pre-post||/||post|| = 5/10"
        );
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

    /// The streaming accumulator must produce the SAME triple as
    /// `divergence_triple` over the same pairs — they share the
    /// per-pair math by construction, and this pins the sharing.
    #[test]
    fn accum_matches_divergence_triple() {
        let pre = [t(&[3.0, 4.0]), t(&[1.0, 2.0])];
        let post = [t(&[6.0, 8.0]), t(&[2.0, 4.0])];
        let reference = divergence_triple(&pre, &post).unwrap();

        let pre_b = [t(&[3.0, 4.0]), t(&[1.0, 2.0])];
        let mut accum = DivergenceAccum::new();
        for (p, q) in pre_b.iter().zip(post.iter()) {
            accum.capture_pre(p).unwrap();
            accum.observe_post(q).unwrap();
        }
        let streamed = accum.finish();
        assert!((streamed.0 - reference.0).abs() < 1e-12, "divergence");
        assert!(
            (streamed.1.unwrap() - reference.1.unwrap()).abs() < 1e-12,
            "post_norm"
        );
        assert!(
            (streamed.2.unwrap() - reference.2.unwrap()).abs() < 1e-12,
            "pre_norm"
        );
        // capture_pre copies — the caller's tensors are NOT mutated
        // (unlike divergence_triple's scratch contract).
        assert_eq!(pre_b[0].to_f32_vec().unwrap(), vec![3.0, 4.0]);
    }

    /// bf16 pre AND post (the decode-into staging under `bf16_wire`
    /// holds bf16 both before and after the decode) must match the f32
    /// reference: `capture_pre` upcasts exactly, `observe_pair` upcasts
    /// the post side.
    #[test]
    fn accum_bf16_staging_matches_f32_reference() {
        let pre_f32 = [t(&[3.0, 4.0]), t(&[1.0, 2.0])];
        let post_f32 = [t(&[6.0, 8.0]), t(&[2.0, 4.0])];
        let reference = divergence_triple(&pre_f32, &post_f32).unwrap();

        let to_bf16 = |ts: &[Tensor]| -> Vec<Tensor> {
            ts.iter()
                .map(|p| p.to_dtype(crate::tensor::DType::BFloat16).unwrap())
                .collect()
        };
        // Rebuild pre (divergence_triple mutated the f32 originals).
        let pre_bf16 = to_bf16(&[t(&[3.0, 4.0]), t(&[1.0, 2.0])]);
        let post_bf16 = to_bf16(&post_f32);
        let mut accum = DivergenceAccum::new();
        for (p, q) in pre_bf16.iter().zip(post_bf16.iter()) {
            accum.capture_pre(p).unwrap();
            accum.observe_post(q).unwrap();
        }
        let streamed = accum.finish();
        assert!((streamed.0 - reference.0).abs() < 1e-6, "divergence");
        assert!(
            (streamed.1.unwrap() - reference.1.unwrap()).abs() < 1e-6,
            "post_norm"
        );
        assert!(
            (streamed.2.unwrap() - reference.2.unwrap()).abs() < 1e-6,
            "pre_norm"
        );
    }

    /// Keep-local rounds trust only the pre sums: divergence exactly 0,
    /// both norms = the pre norm — regardless of what garbage the
    /// zero-mass reply decoded into the post side.
    #[test]
    fn accum_keep_local_is_zero_with_pre_norms() {
        let pre = [t(&[3.0, 4.0])]; // ||pre|| = 5
        let garbage_post = [t(&[0.0, 0.0])];
        let mut accum = DivergenceAccum::new();
        accum.capture_pre(&pre[0]).unwrap();
        accum.observe_post(&garbage_post[0]).unwrap();
        let (d, post_n, pre_n) = accum.finish_keep_local();
        assert_eq!(d, 0.0);
        assert!((pre_n.unwrap() - 5.0).abs() < 1e-5);
        assert!((post_n.unwrap() - 5.0).abs() < 1e-5, "post_norm = pre_norm");
    }

    /// Empty accumulator mirrors divergence_triple's empty contract.
    #[test]
    fn accum_empty_is_zero_none_none() {
        let (d, post, pre) = DivergenceAccum::new().finish();
        assert_eq!(d, 0.0);
        assert!(post.is_none() && pre.is_none());
        let (d, post, pre) = DivergenceAccum::new().finish_keep_local();
        assert_eq!(d, 0.0);
        assert!(post.is_none() && pre.is_none());
    }

    /// Protocol misuse errors loudly rather than corrupting the sums.
    #[test]
    fn accum_protocol_misuse_errors() {
        let a = t(&[1.0]);
        let mut accum = DivergenceAccum::new();
        assert!(accum.observe_post(&a).is_err(), "observe without capture");
        accum.capture_pre(&a).unwrap();
        assert!(accum.capture_pre(&a).is_err(), "double capture");
    }

    /// exact_norm: f32 image, empty contract, bf16 parity.
    #[test]
    fn exact_norm_contracts() {
        assert!(exact_norm(&[]).unwrap().is_none());
        let ts = [t(&[3.0, 4.0]), t(&[0.0, 0.0])];
        assert!((exact_norm(&ts).unwrap().unwrap() - 5.0).abs() < 1e-6);
        let bf: Vec<Tensor> = ts
            .iter()
            .map(|p| p.to_dtype(crate::tensor::DType::BFloat16).unwrap())
            .collect();
        assert!((exact_norm(&bf).unwrap().unwrap() - 5.0).abs() < 1e-6);
    }
}
