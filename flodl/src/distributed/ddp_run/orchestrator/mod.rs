//! DDP run-mode orchestrator: spawns GPU worker threads and a coordinator thread.

mod builder;
mod coord_config;
mod handle;
mod rank_entry;
mod single_host;

pub use builder::DdpBuilder;
pub use handle::DdpHandle;
/// Crate-wide export: every force-exiting cluster role (launcher, relay,
/// agent, rank death, cooperative Worker drop) must terminate through this
/// instead of `std::process::exit` — see its doc for the libtorch
/// static-teardown GP-fault it avoids.
pub(crate) use handle::clean_process_exit;

#[cfg(test)]
pub(super) use coord_config::build_coord_config_from_builder;
