//! DDP run-mode orchestrator: spawns GPU worker threads and a coordinator thread.

mod rank_entry;
mod coord_config;
mod single_host;
mod handle;
mod builder;

pub use handle::DdpHandle;
pub use builder::DdpBuilder;

#[cfg(test)]
pub(super) use coord_config::build_coord_config_from_builder;
