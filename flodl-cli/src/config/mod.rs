//! fdl.yaml configuration loading and discovery.
//!
//! Walks up from CWD to find the project manifest, parses YAML/JSON,
//! and loads sub-command configs from registered command directories.
//!
//! The implementation is split across topic files; this `mod.rs`
//! re-exports the public surface so external callers can continue
//! using `crate::config::Type` paths unchanged.

mod cluster;
mod command;
mod loading;
mod schema;
mod validation;

#[cfg(test)]
mod tests;

pub use cluster::{
    ClusterConfig, ClusterController, ClusterWorker, DdpConfig, LocalDevices, OutputConfig,
    SpeedHint, SshConfig, TrainingConfig, WorkerJoin, DEFAULT_CONTROLLER_PORT,
    DEFAULT_DATA_PATH,
};
pub use command::{load_command, load_command_with_env};
pub use loading::{
    cluster_dispatch_enabled, config_layer_sources, find_config, find_config_in,
    find_project_config, load_merged_value, load_project, load_project_with_env,
    resolve_cluster_dispatch, resolve_config_layers,
};
pub use schema::{
    validate_schema, ArgSpec, CommandConfig, CommandKind, CommandSpec, OptionSpec,
    ProjectConfig, Schema,
};
pub use validation::{
    defaults_only, merge_preset, schema_to_args_spec, validate_preset_for_exec,
    validate_preset_values, validate_presets_strict, validate_tail, ResolvedConfig,
};
