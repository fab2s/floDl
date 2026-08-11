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

mod checkpoint;
mod distributed;
pub mod dot;
mod execution;
pub mod flow;
pub mod gate;
#[allow(clippy::module_inception)]
mod graph;
pub mod halt;
pub mod loop_node;
pub mod map;
pub mod node;
pub mod observe;
pub mod plot;
pub mod profile;
pub mod reshape;
pub mod router;
pub mod snapshot;
pub mod state;
pub mod switch;
pub mod tree;
pub mod trend;
pub mod verbose;

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

pub use execution::{ActiveGraphEpochIterator, GraphEpochIterator};
pub use flow::FlowBuilder;
pub use graph::*;
pub use halt::{LearnedHalt, ThresholdHalt};
pub use loop_node::LoopBuilder;
pub use map::MapBuilder;
pub use observe::Reduce;
pub use plot::format_duration;
pub use profile::{LevelTiming, NodeTiming, Profile, ProfileSource};
pub use reshape::Reshape;
pub use router::{ArgmaxSelector, FixedSelector, SigmoidRouter, SoftmaxRouter};
pub use snapshot::ModelSnapshot;
pub use state::StateAdd;
pub use tree::PathKind;
pub use trend::{Trend, TrendGroup};

/// Merge operation for combining split branches.
pub enum MergeOp {
    /// Element-wise sum of all branches.
    Add,
    /// Element-wise mean of all branches.
    Mean,
}
