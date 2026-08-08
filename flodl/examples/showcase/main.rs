//! Showcase: exercises every method of the graph fluent builder API
//! in a single coherent graph, plus training tools and observation.
//!
//! Builder methods exercised:
//!
//!     From, Through, Tag, Using (backward ref), Using (forward ref),
//!     Split, Merge, Also, Map.Slices, Map.Each, Map.Over, Map.Batched,
//!     Loop.For, Loop.While, Loop.Until,
//!     Gate, Gate.Using, Switch, Switch.Using,
//!     Fork, Input, TagGroup, Build
//!
//! Graph methods exercised:
//!
//!     forward, forward_multi, parameters, train, eval, set_training, reset_state,
//!     detach_state, dot, tagged, flush, trend, enable_profiling,
//!     flush_timings, timing_trend
//!
//! Variable ops exercised (via custom modules):
//!
//!     add, sub, mul, div, matmul, mul_scalar, add_scalar, div_scalar,
//!     relu, sigmoid, tanh, gelu, silu,
//!     sum, mean, sum_dim, mean_dim,
//!     exp, log, neg, abs, sqrt, pow_scalar, clamp,
//!     softmax, log_softmax,
//!     transpose, reshape, narrow, flatten, squeeze, unsqueeze, expand,
//!     cat, select, index_select, permute,
//!     sin, cos, reciprocal, var, std, gather, chunk, repeat, pad,
//!     topk, sort, min, max
//!
//! Training tools exercised:
//!
//!     Adam optimizer, CosineScheduler, mse_loss, clip_grad_norm,
//!     save/load checkpoint, no_grad, observation, profiling, trends
//!
//! Also demonstrates Graph-as-Module: sub-graphs used as reusable blocks
//! inside Split branches and Loop bodies.

use std::collections::HashMap;

use flodl::Monitor;
use flodl::{
    Adam, BatchNorm, CosineScheduler, Device, Dropout, FlowBuilder, GELU, Graph, LayerNorm,
    LearnedHalt, Linear, MergeOp, Module, NamedInputModule, Optimizer, Reshape, SiLU,
    SoftmaxRouter, StateAdd, Tensor, ThresholdHalt, Variable, clip_grad_norm, load_checkpoint_file,
    modules, mse_loss, no_grad, save_checkpoint_file,
};

// ---------------------------------------------------------------------------
// Reusable sub-graph builders
// ---------------------------------------------------------------------------

/// Feed-forward block: Linear -> GELU -> LayerNorm.
fn ffn_block(dim: i64) -> flodl::Result<Graph> {
    FlowBuilder::from(Linear::new(dim, dim)?)
        .through(GELU)
        .through(LayerNorm::new(dim)?)
        .build()
}

/// Projection head: Linear -> LayerNorm.
fn read_head(dim: i64) -> flodl::Result<Graph> {
    FlowBuilder::from(Linear::new(dim, dim)?)
        .through(LayerNorm::new(dim)?)
        .build()
}

/// SiLU block: Linear -> SiLU -> BatchNorm.
fn silu_block(dim: i64) -> flodl::Result<Graph> {
    FlowBuilder::from(Linear::new(dim, dim)?)
        .through(SiLU)
        .through(BatchNorm::new(dim)?)
        .build()
}

// ---------------------------------------------------------------------------
// Custom modules exercising Variable ops
// ---------------------------------------------------------------------------

/// RMS normalization: x / sqrt(mean(x^2) + eps).
/// Exercises: pow_scalar, mean_dim, add_scalar, sqrt, div.
struct RmsNorm {
    eps: f64,
}

impl RmsNorm {
    fn new() -> Self {
        RmsNorm { eps: 1e-6 }
    }
}

impl Module for RmsNorm {
    fn name(&self) -> &str {
        "rmsnorm"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let sq = input.pow_scalar(2.0)?; // pow_scalar
        let ms = sq.mean_dim(-1, true)?; // mean_dim
        let shifted = ms.add_scalar(self.eps)?; // add_scalar
        let rms = shifted.sqrt()?; // sqrt
        input.div(&rms) // div
    }
}

/// Soft clamping: clamp(x * scale, -bound, bound) then abs.
/// Exercises: mul_scalar, clamp, abs.
struct SoftClamp {
    scale: f64,
    bound: f64,
}

impl SoftClamp {
    fn new(scale: f64, bound: f64) -> Self {
        SoftClamp { scale, bound }
    }
}

impl Module for SoftClamp {
    fn name(&self) -> &str {
        "softclamp"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let scaled = input.mul_scalar(self.scale)?; // mul_scalar
        let clamped = scaled.clamp(-self.bound, self.bound)?; // clamp
        clamped.abs() // abs
    }
}

/// Log-space transform: log(exp(x) + 1) (softplus).
/// Exercises: exp, add_scalar (as +1), log.
struct Softplus;

impl Module for Softplus {
    fn name(&self) -> &str {
        "softplus"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let ex = input.exp()?; // exp
        let shifted = ex.add_scalar(1.0)?; // add_scalar (+1)
        shifted.log() // log
    }
}

/// Negated sigmoid gate: sigmoid(-x) * x.
/// Exercises: neg, sigmoid (direct op), mul.
struct NegSigmoidGate;

impl Module for NegSigmoidGate {
    fn name(&self) -> &str {
        "neg_sigmoid_gate"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let negated = input.neg()?; // neg
        let gate = negated.sigmoid()?; // sigmoid
        input.mul(&gate) // mul
    }
}

/// Shape gymnastics: flatten -> unsqueeze -> squeeze -> transpose.
/// Input [B, D] -> flatten [B*D] -> unsqueeze [1, B*D] -> squeeze [B*D]
/// -> reshape back to [B, D].
/// Exercises: flatten, unsqueeze, squeeze, reshape.
struct ShapeOps {
    batch: i64,
    dim: i64,
}

impl ShapeOps {
    fn new(batch: i64, dim: i64) -> Self {
        ShapeOps { batch, dim }
    }
}

impl Module for ShapeOps {
    fn name(&self) -> &str {
        "shape_ops"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let flat = input.flatten(0, -1)?; // flatten
        let expanded = flat.unsqueeze(0)?; // unsqueeze
        let squeezed = expanded.squeeze(0)?; // squeeze
        squeezed.reshape(&[self.batch, self.dim]) // reshape (back)
    }
}

/// Log-softmax along last dim, then sum_dim to scalar per batch.
/// Exercises: log_softmax, sum_dim.
struct LogSoftmaxReduce;

impl Module for LogSoftmaxReduce {
    fn name(&self) -> &str {
        "log_softmax_reduce"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let lsm = input.log_softmax(-1)?; // log_softmax
        lsm.sum_dim(-1, true) // sum_dim (keepdim)
    }
}

/// Transpose dim 0 and dim 1 (exercises transpose + permute).
/// For [A, B] input: transpose(0,1) -> [B, A], permute back to [A, B].
struct TransposeRoundTrip;

impl Module for TransposeRoundTrip {
    fn name(&self) -> &str {
        "transpose_rt"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let t = input.transpose(0, 1)?; // transpose
        t.permute(&[1, 0]) // permute (back)
    }
}

/// Context blending: uses auxiliary input to modulate the stream.
/// Exercises: div_scalar, sigmoid, mul, add (via NamedInputModule).
struct ContextBlend;

impl Module for ContextBlend {
    fn name(&self) -> &str {
        "context_blend"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        Ok(input.clone())
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }
}

impl NamedInputModule for ContextBlend {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> flodl::Result<Variable> {
        let ctx = &refs["ctx"];
        let scaled = ctx.div_scalar(2.0)?; // div_scalar
        let gate = scaled.sigmoid()?; // sigmoid
        let modulated = input.mul(&gate)?; // mul
        modulated.add(input) // add
    }
}

/// Spectral basis: sin/cos/reciprocal projections.
/// Used as a fork side-branch (output captured, stream continues unchanged).
/// Exercises: sin, cos, reciprocal, tanh.
struct SpectralBasis;

impl Module for SpectralBasis {
    fn name(&self) -> &str {
        "spectral_basis"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let s = input.sin()?; // sin
        let c = input.cos()?; // cos
        let sc = s.add(&c)?;
        let r = sc.reciprocal()?; // reciprocal
        r.tanh() // tanh
    }
}

/// Variance gate: gate stream by normalized variance.
/// Exercises: mean (scalar), var, std, expand.
struct VarianceGate {
    dim: i64,
}

impl VarianceGate {
    fn new(dim: i64) -> Self {
        VarianceGate { dim }
    }
}

impl Module for VarianceGate {
    fn name(&self) -> &str {
        "variance_gate"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let m = input.mean()?; // mean (scalar)
        let _v = input.var()?; // var (scalar)
        let s = input.std()?; // std (scalar)
        let gate_val = m.add(&s)?;
        let gate = gate_val.expand(&[1, self.dim])?; // expand
        input.mul(&gate) // mul
    }
}

/// Chunk-recombine: split along last dim, process, cat back.
/// Exercises: chunk, relu (Variable op), cat.
struct ChunkRecombine;

impl Module for ChunkRecombine {
    fn name(&self) -> &str {
        "chunk_recombine"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let chunks = input.chunk(2, -1)?; // chunk
        let a = chunks[0].relu()?; // relu (Variable op)
        let b = chunks[1].neg()?;
        a.cat(&b, -1) // cat
    }
}

/// Attention-like op exercise: softmax, select, narrow, index_select.
/// Input [1, D] -> exercises each op then returns [1, D].
struct AttentionLikeOps {
    dim: i64,
}

impl AttentionLikeOps {
    fn new(dim: i64) -> Self {
        AttentionLikeOps { dim }
    }
}

impl Module for AttentionLikeOps {
    fn name(&self) -> &str {
        "attention_ops"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let weights = input.softmax(-1)?; // softmax

        // select dim 0, index 0 -> [D]
        let row = input.select(0, 0)?; // select
        let row2d = row.unsqueeze(0)?;

        // narrow: take first half along last dim
        let half_dim = self.dim / 2;
        let first_half = row2d.narrow(-1, 0, half_dim)?; // narrow

        // index_select: pick specific indices from first_half [1, half_dim]
        let idx = Tensor::from_i64(&[0, 1], &[2], Device::CPU)?;
        let selected = first_half.index_select(-1, &idx)?; // index_select

        // Combine: scale weights by mean of selected values
        let scale = selected.mean()?; // scalar
        let scale_expanded = scale.expand(&[1, self.dim])?; // expand (scalar -> [1,D])
        weights.add(&scale_expanded)
    }
}

/// TopK/sort/gather/min/max/pad exercise.
/// Input [1, D] -> exercises each op then returns [1, D].
struct TopKFilterOps {
    dim: i64,
}

impl TopKFilterOps {
    fn new(dim: i64) -> Self {
        TopKFilterOps { dim }
    }
}

impl Module for TopKFilterOps {
    fn name(&self) -> &str {
        "topk_filter"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        // topk: get top 4 values
        let (values, indices) = input.topk(4, -1, true, true)?; // topk

        // sort the top values
        let (sorted, _sort_idx) = values.sort(-1, false)?; // sort

        // gather: use indices to rearrange
        let gathered = input.gather(-1, &indices)?; // gather

        // min/max as scalar ops
        let mn = gathered.min()?; // min
        let mx = gathered.max()?; // max
        let range = mx.sub(&mn)?;

        // pad: pad sorted [1,4] to [1, D] with zeros on the right
        let pad_amount = self.dim - 4;
        let padded = sorted.pad(&[0, pad_amount], 0.0)?; // pad

        padded.add(&range.expand(&[1, self.dim])?)
    }
}

/// Repeat exercise: repeat tensor along dims.
/// Input [1, D] -> repeat [1, 2] -> [1, 2D] -> narrow back to [1, D].
struct RepeatNarrow {
    dim: i64,
}

impl RepeatNarrow {
    fn new(dim: i64) -> Self {
        RepeatNarrow { dim }
    }
}

impl Module for RepeatNarrow {
    fn name(&self) -> &str {
        "repeat_narrow"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        let repeated = input.repeat(&[1, 2])?; // repeat
        repeated.narrow(-1, 0, self.dim) // narrow (trim back)
    }
}

/// Resettable module: exercises Module::reset() auto-detection by loops.
/// Accumulates a call counter, reset clears it.
struct CounterModule {
    count: std::cell::Cell<u32>,
}

impl CounterModule {
    fn new() -> Self {
        CounterModule {
            count: std::cell::Cell::new(0),
        }
    }
}

impl Module for CounterModule {
    fn name(&self) -> &str {
        "counter"
    }

    fn forward(&self, input: &Variable) -> flodl::Result<Variable> {
        self.count.set(self.count.get() + 1);
        Ok(input.clone())
    }

    fn reset(&self) {
        self.count.set(0);
    }
}

// ---------------------------------------------------------------------------
// Custom Switch selector (user-defined NamedInputModule)
// ---------------------------------------------------------------------------

/// Picks branch 0 (lightweight) or branch 1 (heavy) based on
/// activation magnitude of the "refined" reference.
struct HeavyPathSelector;

impl Module for HeavyPathSelector {
    fn name(&self) -> &str {
        "heavy_path_selector"
    }

    fn forward(&self, _input: &Variable) -> flodl::Result<Variable> {
        let t = Tensor::from_f32(&[0.0], &[1], Device::CPU)?;
        Ok(Variable::new(t, false))
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }
}

impl NamedInputModule for HeavyPathSelector {
    fn forward_named(
        &self,
        _input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> flodl::Result<Variable> {
        let refined = &refs["refined"];
        let data = refined.data().to_f32_vec()?;
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let branch = if max_val > 5.0 { 1.0_f32 } else { 0.0 };
        let t = Tensor::from_f32(&[branch], &[1], Device::CPU)?;
        Ok(Variable::new(t, false))
    }
}

// ---------------------------------------------------------------------------
// Sub-graph builder for fork side-branch
// ---------------------------------------------------------------------------

/// Spectral monitor sub-graph: SpectralBasis -> Linear projection.
/// Used as a fork target — its output gets tagged but doesn't affect the main stream.
fn spectral_monitor(dim: i64) -> flodl::Result<Graph> {
    FlowBuilder::from(SpectralBasis)
        .through(Linear::new(dim, dim)?)
        .build()
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

/// Build the extended showcase graph.
///
/// Data flow `[B,2]` -> `[B,2]` (B=2 for BatchNorm compatibility):
///
/// ```text
/// Input("ctx")                                     auxiliary input port
/// From(Linear 2->8)                                Tag("input")
/// Through(GELU) -> Through(LayerNorm)
/// Through(RmsNorm)                                 pow_scalar, mean_dim, add_scalar, sqrt, div
/// Through(ContextBlend).Using("ctx")               div_scalar, sigmoid, mul, add (NamedInputModule)
/// Fork(spectral_monitor).Tag("spectral")           sin, cos, reciprocal, tanh (side branch)
/// Split(read_head, read_head) -> Mean()            multi-head read
/// Also(Linear 8->8)                                residual
/// Through(Dropout(0.1))
/// Through(SoftClamp(0.5, 3.0))                     mul_scalar, clamp, abs
/// Through(Softplus)                                exp, add_scalar, log
/// Through(VarianceGate)                            mean, var, std, expand
/// Map(read_head(2)).Slices(4)                      per-position processing
/// Through(Reshape [2,4])
/// Map(Linear 4->4).Each()                          Tag("halves")
/// Map(Linear 4->4).Over("halves")                  refine tagged halves
/// Map(Linear 4->4).Batched().Each()                batched fast-path map
/// Through(Reshape [1,8])
/// Through(ShapeOps)                                flatten, unsqueeze, squeeze, reshape
/// Through(NegSigmoidGate)                          neg, sigmoid, mul
/// Through(TransposeRoundTrip)                      transpose, permute
/// Through(CounterModule)                           exercises Module::reset()
/// Through(ChunkRecombine)                          chunk, relu, cat
/// Through(AttentionLikeOps)                        softmax, select, narrow, index_select
/// Through(TopKFilterOps)                           topk, sort, gather, min, max, pad
/// Through(RepeatNarrow)                            repeat
/// Loop(silu_block).For(2)                          SiLU + BatchNorm, Tag("refined")
/// Gate(SoftmaxRouter, Linear, Linear)              .Using("input")
/// Switch(HeavyPathSelector, Linear, ffn_block)     .Using("refined")
/// Through(StateAdd).Using("memory").Tag("memory")  forward ref
/// Loop(Linear).While(ThresholdHalt(100), 5)
/// Loop(Linear).Until(LearnedHalt(8), 7)
/// Through(LogSoftmaxReduce)                        log_softmax, sum_dim
/// Through(Linear 1->8)                             widen back
/// Split(Linear, Linear).TagGroup("final_heads") -> Add()
/// Through(Linear 8->2)                             output projection   Tag("output")
/// ```
fn build_showcase() -> flodl::Result<Graph> {
    const B: i64 = 2; // batch size (>= 2 for BatchNorm training mode)
    const H: i64 = 8;

    FlowBuilder::from(Linear::new(2, H)?)
        // input() declares auxiliary graph inputs — forward_multi receives them
        .input(&["ctx"])
        // Tag names a position in the stream for later reference via .using()
        .tag("input")
        // .through() chains modules sequentially: stream -> module -> stream
        .through(GELU)
        .through(LayerNorm::new(H)?)
        .through(RmsNorm::new())
        // ContextBlend is a NamedInputModule that reads the "ctx" auxiliary input
        .through(ContextBlend)
        .using(&["ctx"])
        // .fork() runs a side-branch: output can be tagged, main stream unchanged
        .fork(spectral_monitor(H)?)
        .tag("spectral")
        // .split() forks the stream into parallel branches, .merge() recombines.
        // modules![] is shorthand for vec![Box::new(...) as Box<dyn Module>, ...]
        .split(modules![read_head(H)?, read_head(H)?])
        .merge(MergeOp::Mean)
        // .also() adds a residual connection: output = stream + module(stream)
        .also(Linear::new(H, H)?)
        .through(Dropout::new(0.1))
        .through(SoftClamp::new(0.5, 3.0))
        .through(Softplus)
        // VarianceGate exercises mean/var/std/expand
        .through(VarianceGate::new(H))
        // .map().slices(n) decomposes [B,D] -> [B*n,D/n], applies body, recomposes
        .map(read_head(2)?)
        .slices(H / 2)
        // Reshape changes tensor dimensions without copying data
        .through(Reshape::new(&[B * 2, H / 2]))
        // .map().each() applies body independently to each element in a multi-stream
        .map(Linear::new(H / 2, H / 2)?)
        .each()
        .tag("halves")
        // .map().over(tag) iterates over a tagged tensor (backward ref) instead
        // of the current stream — useful for refining previously computed features
        .map(Linear::new(H / 2, H / 2)?)
        .over("halves")
        // .map().batched().each() — fast path: full batch in one call
        .map(Linear::new(H / 2, H / 2)?)
        .batched()
        .each()
        .through(Reshape::new(&[B, H]))
        .through(ShapeOps::new(B, H))
        .through(NegSigmoidGate)
        .through(TransposeRoundTrip)
        // CounterModule overrides reset() — loops auto-call it before iterating
        .through(CounterModule::new())
        // ChunkRecombine: chunk, relu (Variable op), cat
        .through(ChunkRecombine)
        // AttentionLikeOps: softmax, select, narrow, index_select
        .through(AttentionLikeOps::new(H))
        // TopKFilterOps: topk, sort, gather, min, max, pad
        .through(TopKFilterOps::new(H))
        // RepeatNarrow: repeat
        .through(RepeatNarrow::new(H))
        // .loop_body().for_n(n) repeats the body n times, feeding output back as input.
        // silu_block is a sub-graph (Graph implements Module) — graphs compose freely.
        .loop_body(silu_block(H)?)
        .for_n(2)
        .tag("refined")
        // .gate() is soft routing (mixture of experts): all experts run, router
        // produces weights, outputs are combined. .using() feeds the tagged "input"
        // stream to the router as a backward reference.
        .gate(
            SoftmaxRouter::new(H, 2)?,
            modules![Linear::new(H, H)?, Linear::new(H, H)?],
        )
        .using(&["input"])
        // .switch() is hard routing: router picks one branch, others are skipped.
        // HeavyPathSelector is a custom NamedInputModule — it receives the "refined"
        // ref via forward_named() and decides which branch to activate.
        .switch(
            HeavyPathSelector,
            modules![Linear::new(H, H)?, ffn_block(H)?],
        )
        .using(&["refined"])
        // Forward reference: .using("memory") reads a tag that doesn't exist yet —
        // the framework creates a state buffer. .tag("memory") writes to it.
        // On the first pass, the state is zero (pass-through). On subsequent passes,
        // StateAdd accumulates: stream + previous_memory.
        .through(StateAdd)
        .using(&["memory"])
        .tag("memory")
        // .while_cond() repeats until the halt module signals stop (or max iterations).
        // ThresholdHalt stops when the stream's L2 norm exceeds the threshold.
        .loop_body(Linear::new(H, H)?)
        .while_cond(ThresholdHalt::new(100.0), 5)
        // .until_cond() is the inverse: repeats until halt signals true.
        // LearnedHalt has trainable parameters — it learns when to stop.
        .loop_body(Linear::new(H, H)?)
        .until_cond(LearnedHalt::new(H)?, 7)
        .through(LogSoftmaxReduce)
        .through(Linear::new(1, H)?)
        // Split with tag_group: names each branch ("final_heads_0", "final_heads_1")
        .split(vec![
            Box::new(Linear::new(H, H)?),
            Box::new(Linear::new(H, H)?),
        ])
        .tag_group("final_heads")
        .merge(MergeOp::Add)
        // Final projection and output tag for observation
        .through(Linear::new(H, 2)?)
        .tag("output")
        .build()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_input(requires_grad: bool) -> Variable {
    let t = Tensor::from_f32(&[1.0, 2.0, 0.5, -1.0], &[2, 2], Device::CPU).unwrap();
    Variable::new(t, requires_grad)
}

fn make_context() -> Variable {
    let t = Tensor::from_f32(
        &[
            0.5, -0.3, 0.8, 1.2, -0.5, 0.1, 0.9, -0.7, 0.2, 0.7, -0.4, 0.6, 1.0, -0.8, 0.3, -0.1,
        ],
        &[2, 8],
        Device::CPU,
    )
    .unwrap();
    Variable::new(t, false)
}

fn make_target() -> Variable {
    let t = Tensor::from_f32(&[0.5, -0.5, -0.3, 0.8], &[2, 2], Device::CPU).unwrap();
    Variable::new(t, false)
}

#[cfg(test)]
fn count_grads(params: &[flodl::Parameter]) -> usize {
    params
        .iter()
        .filter(|p| {
            p.variable
                .grad()
                .and_then(|g| g.to_f32_vec().ok())
                .is_some_and(|d| d.iter().any(|v| *v != 0.0))
        })
        .count()
}

// ---------------------------------------------------------------------------
// Main: demo run
// ---------------------------------------------------------------------------

fn main() {
    flodl::manual_seed(42);
    println!("=== floDl showcase ===\n");

    // -- Build --
    println!("Building graph...");
    let g = build_showcase().expect("build failed");
    let n_params = g.parameters().len();
    println!("Parameters: {}", n_params);

    // -- Forward (with auxiliary input) --
    let result = g
        .forward_multi(&[make_input(false), make_context()])
        .expect("forward failed");
    println!(
        "Output: {:?} (shape {:?})",
        result.data().to_f32_vec().unwrap(),
        result.shape()
    );

    // -- Forward ref carries state --
    g.reset_state();
    let r1 = g
        .forward_multi(&[make_input(false), make_context()])
        .unwrap();
    let v1 = r1.data().to_f32_vec().unwrap();
    let r2 = g
        .forward_multi(&[make_input(false), make_context()])
        .unwrap();
    let v2 = r2.data().to_f32_vec().unwrap();
    println!("State drift: pass2 differs = {}", v1 != v2);

    // -- Reset --
    g.reset_state();
    let r3 = g
        .forward_multi(&[make_input(false), make_context()])
        .unwrap();
    let v3 = r3.data().to_f32_vec().unwrap();
    println!("Reset restores: {}", v1 == v3);

    // -- DOT + SVG (structural) --
    let dot = g.dot();
    println!("DOT: {} bytes", dot.len());

    // Write structural DOT
    let dot_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/showcase/showcase.dot"
    );
    std::fs::write(dot_path, &dot).expect("write showcase.dot");
    println!("Wrote {}", dot_path);

    // Write structural SVG
    let svg_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/showcase/showcase.svg"
    );
    let svg = g.svg(Some(svg_path)).expect("write showcase.svg");
    println!("Wrote {} ({} bytes)", svg_path, svg.len());

    // -- Training loop with observation + profiling + monitor --
    println!("\n--- Training (5 epochs x 4 steps) ---");
    g.train();
    g.reset_state();
    g.enable_profiling();

    let params = g.parameters();
    let mut optimizer = Adam::new(&params, 0.001);
    let num_epochs = 5;
    let total_steps = num_epochs * 4;
    let sched = CosineScheduler::new(0.001, 1e-5, total_steps);
    let mut monitor = Monitor::new(num_epochs);

    let mut step_idx = 0;
    for epoch in 0..num_epochs {
        let t = std::time::Instant::now();
        for _ in 0..4 {
            optimizer.zero_grad();
            let input = make_input(true);
            let ctx = make_context();
            let target = make_target();

            let pred = g.forward_multi(&[input, ctx]).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();

            loss.backward().unwrap();
            clip_grad_norm(&params, 1.0).unwrap();
            optimizer.set_lr(sched.lr(step_idx));
            optimizer.step().unwrap();
            step_idx += 1;

            g.record_scalar("loss", loss.item().unwrap());
            g.record_scalar("lr", sched.lr(step_idx - 1));
            g.end_step();
        }

        g.end_epoch();
        monitor.log(epoch, t.elapsed(), &g);
    }

    // -- Trends --
    let trend = g.trend("loss");
    println!(
        "\nLoss trend: {} epochs, slope={:.4}, improving={}",
        trend.len(),
        trend.slope(0),
        trend.improving(0),
    );

    // Timing trends use node IDs — pick the first tagged one
    let timing = g.timing_trend("input");
    println!(
        "Timing trend (input node): {} epochs, mean={:.1}us",
        timing.len(),
        timing.mean() * 1e6,
    );

    // -- Write profiling DOT + SVG --
    let profile_dot = g.dot_with_profile();
    let profile_dot_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/showcase/showcase_profile.dot"
    );
    std::fs::write(profile_dot_path, &profile_dot).expect("write showcase_profile.dot");
    println!("Wrote {}", profile_dot_path);

    let profile_svg_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/showcase/showcase_profile.svg"
    );
    let profile_svg = g
        .svg_with_profile(Some(profile_svg_path))
        .expect("write showcase_profile.svg");
    println!("Wrote {} ({} bytes)", profile_svg_path, profile_svg.len());

    // -- Write training HTML --
    let html_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/showcase/showcase_training.html"
    );
    g.plot_html(html_path, &["loss"])
        .expect("write showcase_training.html");
    println!("Wrote {}", html_path);

    // -- Write training log --
    let log_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/showcase/showcase_training.log"
    );
    g.write_log(log_path, 5, &["loss"])
        .expect("write showcase_training.log");
    println!("Wrote {}", log_path);

    // -- Checkpoint round-trip --
    let path = "/tmp/flodl_showcase_checkpoint.fdl";
    let named = g.named_parameters();
    let named_bufs = g.named_buffers();
    save_checkpoint_file(path, &named, &named_bufs, Some(g.structural_hash()))
        .expect("save failed");
    let report = load_checkpoint_file(path, &named, &named_bufs, Some(g.structural_hash()))
        .expect("load failed");
    println!(
        "\nCheckpoint save/load: OK ({} loaded)",
        report.loaded.len()
    );

    // -- no_grad inference (eval mode works now — BatchNorm has running stats from training) --
    g.eval();
    g.reset_state();
    let final_out = no_grad(|| g.forward_multi(&[make_input(false), make_context()])).unwrap();
    let final_vals = final_out.data().to_f32_vec().unwrap();
    println!("no_grad inference: {:?}", final_vals);
    assert!(
        final_vals.iter().all(|v| v.is_finite()),
        "no_grad output should be finite"
    );

    println!("\nAll showcase checks passed.");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build() {
        let g = build_showcase().unwrap();
        let result = g
            .forward_multi(&[make_input(false), make_context()])
            .unwrap();
        let vals = result.data().to_f32_vec().unwrap();
        assert_eq!(
            vals.len(),
            4,
            "expected 4 outputs (2x2), got {}",
            vals.len()
        );
    }

    #[test]
    fn test_forward_ref_carries_state() {
        let g = build_showcase().unwrap();

        let r1 = g
            .forward_multi(&[make_input(false), make_context()])
            .unwrap();
        let v1 = r1.data().to_f32_vec().unwrap();

        let r2 = g
            .forward_multi(&[make_input(false), make_context()])
            .unwrap();
        let v2 = r2.data().to_f32_vec().unwrap();

        assert_ne!(v1, v2, "pass 2 should differ from pass 1");
    }

    #[test]
    fn test_reset_state() {
        let g = build_showcase().unwrap();

        // Populate BatchNorm running stats, then switch to eval mode
        // so forward passes don't update running stats (deterministic).
        g.forward_multi(&[make_input(false), make_context()])
            .unwrap();
        g.eval();
        g.reset_state();

        let r1 = g
            .forward_multi(&[make_input(false), make_context()])
            .unwrap();
        let v1 = r1.data().to_f32_vec().unwrap();

        g.forward_multi(&[make_input(false), make_context()])
            .unwrap();

        g.reset_state();
        let r3 = g
            .forward_multi(&[make_input(false), make_context()])
            .unwrap();
        let v3 = r3.data().to_f32_vec().unwrap();

        assert_eq!(v1, v3, "after reset should match pass 1");
    }

    #[test]
    fn test_detach_state() {
        let g = build_showcase().unwrap();

        g.forward_multi(&[make_input(false), make_context()])
            .unwrap();
        g.detach_state();

        let result = g
            .forward_multi(&[make_input(false), make_context()])
            .unwrap();
        assert_eq!(result.data().to_f32_vec().unwrap().len(), 4);
    }

    #[test]
    fn test_backward() {
        let g = build_showcase().unwrap();

        let result = g
            .forward_multi(&[make_input(true), make_context()])
            .unwrap();
        let loss = result.sum().unwrap();
        loss.backward().unwrap();

        let with_grad = count_grads(&g.parameters());
        assert!(with_grad > 0, "no parameters received gradients");
    }

    #[test]
    fn test_parameters() {
        let g = build_showcase().unwrap();
        let params = g.parameters();
        assert!(
            params.len() > 44,
            "expected more than 44 params (extended graph), got {}",
            params.len()
        );
    }

    #[test]
    fn test_set_training() {
        let g = build_showcase().unwrap();

        // Run one training pass to populate BatchNorm running stats
        g.forward_multi(&[make_input(false), make_context()])
            .unwrap();

        // Now eval mode works (running stats populated)
        g.set_training(false);
        g.reset_state();
        let r1 = g
            .forward_multi(&[make_input(false), make_context()])
            .unwrap();

        // Switch back to training
        g.set_training(true);
        g.reset_state();
        let r2 = g
            .forward_multi(&[make_input(false), make_context()])
            .unwrap();

        assert_eq!(r1.data().to_f32_vec().unwrap().len(), 4);
        assert_eq!(r2.data().to_f32_vec().unwrap().len(), 4);
    }

    #[test]
    fn test_dot() {
        let g = build_showcase().unwrap();
        let dot = g.dot();
        assert!(!dot.is_empty(), "DOT output is empty");
        assert!(dot.contains("digraph"), "DOT should contain digraph");
    }

    #[test]
    fn test_training_loop() {
        let g = build_showcase().unwrap();
        g.train();

        let params = g.parameters();
        let mut optimizer = Adam::new(&params, 0.01);

        let mut losses = Vec::new();
        for _ in 0..3 {
            let input = make_input(true);
            let ctx = make_context();
            let target = make_target();

            let pred = g.forward_multi(&[input, ctx]).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            losses.push(loss.item().unwrap());

            loss.backward().unwrap();
            clip_grad_norm(&params, 1.0).unwrap();
            optimizer.step().unwrap();
            optimizer.zero_grad();
            g.end_step();
        }

        // Loss should be finite across all steps
        for (i, &l) in losses.iter().enumerate() {
            assert!(l.is_finite(), "loss at step {} is not finite: {}", i, l);
        }
    }

    #[test]
    fn test_observation() {
        let g = build_showcase().unwrap();

        // Run forward — tagged outputs should be captured
        let out = g
            .forward_multi(&[make_input(false), make_context()])
            .unwrap();

        // "output" tag should have a captured value
        let tagged = g.tagged("output");
        assert!(tagged.is_some(), "tagged 'output' not captured");
        assert_eq!(tagged.unwrap().shape(), &[2, 2]);

        // Record a scalar metric manually (output is [1,2], not scalar)
        let loss_val = out
            .data()
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .sum::<f64>();
        g.record("test_loss", &[loss_val]);
        g.flush(&["test_loss"]);
        assert_eq!(g.flush_count(), 1);

        // Run another epoch
        let out2 = g
            .forward_multi(&[make_input(false), make_context()])
            .unwrap();
        let loss_val2 = out2
            .data()
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .sum::<f64>();
        g.record("test_loss", &[loss_val2]);
        g.flush(&["test_loss"]);
        assert_eq!(g.flush_count(), 2);

        // Trend should have 2 epochs
        let trend = g.trend("test_loss");
        assert_eq!(trend.len(), 2, "expected 2 epochs in trend");
    }

    #[test]
    fn test_profiling() {
        let g = build_showcase().unwrap();
        g.enable_profiling();

        g.forward_multi(&[make_input(false), make_context()])
            .unwrap();
        g.collect_timings(&[]); // snapshot node timings to buffer
        g.flush_timings(&[]); // flush buffer to epoch history

        let timing = g.timing_trend("input");
        assert_eq!(timing.len(), 1, "expected 1 timing epoch");
        assert!(timing.latest() > 0.0, "timing should be positive");
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let g = build_showcase().unwrap();
        let params = g.parameters();
        let named = g.named_parameters();

        // Populate BatchNorm running stats, then use eval mode for deterministic output
        g.forward_multi(&[make_input(false), make_context()])
            .unwrap();
        g.eval();
        g.reset_state();

        // Save checkpoint and capture baseline output
        let path = "/tmp/flodl_showcase_test_ckpt.fdl";
        let named_bufs = g.named_buffers();
        save_checkpoint_file(path, &named, &named_bufs, Some(g.structural_hash())).unwrap();

        let before = g
            .forward_multi(&[make_input(false), make_context()])
            .unwrap();
        let v_before = before.data().to_f32_vec().unwrap();
        assert!(
            v_before.iter().all(|v| v.is_finite()),
            "pre-train output NaN"
        );

        // Capture first parameter tensor for direct comparison
        let p0_before = params[0].variable.data().to_f32_vec().unwrap();

        // Mutate parameters via optimizer step
        g.reset_state();
        g.train();
        let pred = g
            .forward_multi(&[make_input(true), make_context()])
            .unwrap();
        let loss = pred.sum().unwrap();
        loss.backward().unwrap();
        let mut opt = Adam::new(&params, 0.1);
        opt.step().unwrap();

        // Verify parameters actually changed
        let p0_after = params[0].variable.data().to_f32_vec().unwrap();
        assert_ne!(p0_before, p0_after, "training should change parameters");

        // Restore checkpoint and verify parameters match original
        let report =
            load_checkpoint_file(path, &named, &named_bufs, Some(g.structural_hash())).unwrap();
        assert_eq!(report.loaded.len(), named.len());
        let p0_restored = params[0].variable.data().to_f32_vec().unwrap();
        assert_eq!(
            p0_before, p0_restored,
            "checkpoint restore should match original params"
        );

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_no_grad() {
        let g = build_showcase().unwrap();

        let result = no_grad(|| g.forward_multi(&[make_input(true), make_context()])).unwrap();
        let vals = result.data().to_f32_vec().unwrap();
        assert_eq!(vals.len(), 4);
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "no_grad should produce finite values"
        );
    }

    #[test]
    fn test_visualization() {
        let g = build_showcase().unwrap();

        // Structural DOT
        let dot = g.dot();
        assert!(dot.contains("digraph"), "DOT should contain digraph");
        assert!(dot.contains("#input"), "DOT should contain #input tag");

        let dot_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/showcase/showcase.dot"
        );
        std::fs::write(dot_path, &dot).unwrap();

        // Structural SVG
        let svg_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/showcase/showcase.svg"
        );
        let svg = g.svg(Some(svg_path)).unwrap();
        assert!(svg.len() > 100, "SVG should have content");

        // Run a forward pass with profiling for timing DOT
        g.enable_profiling();
        g.forward_multi(&[make_input(false), make_context()])
            .unwrap();

        let profile_dot = g.dot_with_profile();
        assert!(
            profile_dot.contains("Forward:"),
            "profile DOT should show total time"
        );

        let profile_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/showcase/showcase_profile.dot"
        );
        std::fs::write(profile_path, &profile_dot).unwrap();

        // Profile SVG
        let profile_svg_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/showcase/showcase_profile.svg"
        );
        let profile_svg = g.svg_with_profile(Some(profile_svg_path)).unwrap();
        assert!(profile_svg.len() > 100, "profile SVG should have content");

        // Training with observation for HTML + log
        g.train();
        g.reset_state();
        let params = g.parameters();
        let mut optimizer = Adam::new(&params, 0.01);

        for _epoch in 0..3 {
            for _ in 0..4 {
                optimizer.zero_grad();
                let pred = g
                    .forward_multi(&[make_input(true), make_context()])
                    .unwrap();
                let loss = mse_loss(&pred, &make_target()).unwrap();
                loss.backward().unwrap();
                optimizer.step().unwrap();

                g.record_scalar("loss", loss.item().unwrap());
                g.end_step();
            }
            g.end_epoch();
        }

        // Training HTML plot
        let html_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/showcase/showcase_training.html"
        );
        g.plot_html(html_path, &["loss"]).unwrap();

        // Training log
        let log_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/showcase/showcase_training.log"
        );
        g.write_log(log_path, 3, &["loss"]).unwrap();

        // Verify files exist and have content
        assert!(std::fs::metadata(dot_path).unwrap().len() > 100);
        assert!(std::fs::metadata(svg_path).unwrap().len() > 100);
        assert!(std::fs::metadata(profile_path).unwrap().len() > 100);
        assert!(std::fs::metadata(profile_svg_path).unwrap().len() > 100);
        assert!(std::fs::metadata(html_path).unwrap().len() > 100);
        assert!(std::fs::metadata(log_path).unwrap().len() > 10);
    }

    #[test]
    fn test_cosine_scheduler() {
        let sched = CosineScheduler::new(0.01, 1e-5, 10);

        let lr_start = sched.lr(0);
        let lr_end = sched.lr(10);

        assert!(
            lr_end < lr_start,
            "LR should decrease: {} -> {}",
            lr_start,
            lr_end
        );
        assert!((lr_end - 1e-5).abs() < 1e-4, "LR should reach min_lr");
    }

    #[test]
    fn test_fork_tag() {
        let g = build_showcase().unwrap();
        g.forward_multi(&[make_input(false), make_context()])
            .unwrap();

        // Fork output should be captured via tag
        let spectral = g.tagged("spectral");
        assert!(spectral.is_some(), "fork tag 'spectral' not captured");
    }
}
