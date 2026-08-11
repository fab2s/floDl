//! Subcommand modules carved out of `main.rs` — one file per major command surface.

pub mod cargo;
pub mod config;
pub mod install;
pub mod libtorch;
pub mod nccl;
pub mod schema;
