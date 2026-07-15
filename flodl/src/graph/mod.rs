//! Computation graph: fluent builder, parallel execution, observation, profiling,
//! visualization, and hierarchical composition.
//!
//! Build graphs with `FlowBuilder`, execute via the `Module` trait.
//! Label subgraphs for tree features: selective freeze/thaw, subgraph
//! checkpoints, cross-boundary observation, and per-subgraph optimizer groups.
//!
//! ```ignore
//! let encoder = FlowBuilder::from(Linear::new(4, 8)?)
//!     .through(GELU)
//!     .label("encoder")
//!     .build()?;
//!
//! let model = FlowBuilder::from(encoder)
//!     .through(Linear::new(8, 2)?)
//!     .build()?;
//!
//! let y = model.forward(&x)?;
//! model.freeze("encoder")?;  // freeze by label path
//! ```

pub mod node;
pub mod flow;
pub mod loop_node;
pub mod switch;
pub mod gate;
pub mod map;
pub mod observe;
pub mod trend;
pub mod profile;
pub mod dot;
pub mod plot;
pub mod router;
pub mod halt;
pub mod reshape;
pub mod state;
pub mod snapshot;
pub mod tree;
pub mod verbose;
#[allow(clippy::module_inception)]
mod graph;
mod distributed;

use std::collections::HashMap;

use crate::autograd::Variable;
use crate::data::Batch;

/// Context passed to the per-batch loss closure during El Che distributed
/// training. All fields carry live autograd graphs, so the returned loss
/// scalar can be backpropagated immediately.
///
/// ```ignore
/// model.set_loss_fn(|ctx: &LossContext| {
///     let cls  = cross_entropy_loss(&ctx.tags["head"], &ctx.batch["label"])?;
///     let rec  = mse_loss(&ctx.tags["recon"], &ctx.batch["image"])?;
///     Ok(cls + rec)
/// });
/// ```
pub struct LossContext<'a> {
    /// Forward output (live autograd).
    pub output: &'a Variable,
    /// The per-device batch with all named fields (inputs + targets).
    pub batch: &'a Batch,
    /// Tagged outputs from this forward pass (live autograd).
    pub tags: &'a HashMap<String, Variable>,
    /// Loop traces keyed by tag name (live autograd).
    pub traces: &'a HashMap<String, Vec<Variable>>,
}

/// Graph-side view of any [`Module`](crate::nn::Module): `.as_graph()`
/// recovers the [`Graph`] behind a `dyn Module` when there is one.
///
/// This is where the graph↔module downcast lives — the `Module` trait
/// itself stays graph-agnostic (its [`as_any`](crate::nn::Module::as_any)
/// identity hook carries no graph vocabulary), and this blanket
/// extension turns that hook back into the ergonomic `.as_graph()`
/// call wherever `GraphExt` is in scope (it is re-exported at the
/// crate root, so `use flodl::*` brings it in).
///
/// Returns `Some` for [`Graph`] itself and for transparent wrappers
/// that present their inner graph through `as_any`; `None` for plain
/// leaf modules.
pub trait GraphExt {
    /// Downcast to [`Graph`] for hierarchical tree composition and
    /// graph-aware framework paths.
    fn as_graph(&self) -> Option<&Graph>;
}

impl<M: crate::nn::Module + ?Sized> GraphExt for M {
    fn as_graph(&self) -> Option<&Graph> {
        self.as_any().and_then(|a| a.downcast_ref::<Graph>())
    }
}

pub use flow::FlowBuilder;
pub use loop_node::LoopBuilder;
pub use map::MapBuilder;
pub use trend::{Trend, TrendGroup};
pub use profile::{Profile, NodeTiming, LevelTiming};
pub use plot::format_duration;
pub use router::{SoftmaxRouter, SigmoidRouter, FixedSelector, ArgmaxSelector};
pub use halt::{ThresholdHalt, LearnedHalt};
pub use reshape::Reshape;
pub use state::StateAdd;
pub use observe::Reduce;
pub use tree::PathKind;
pub use snapshot::ModelSnapshot;
pub use graph::*;

/// Merge operation for combining split branches.
pub enum MergeOp {
    /// Element-wise sum of all branches.
    Add,
    /// Element-wise mean of all branches.
    Mean,
}
