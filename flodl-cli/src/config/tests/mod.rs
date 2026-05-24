//! Tests for [`super`] (`config` module). Hosts shared test helpers
//! plus mod-decls for topic test files. Helpers are `pub(super)` so
//! child modules pull them via `use super::*;`.

pub(super) use super::*;

use std::path::PathBuf;

mod cluster_tests;
mod command_tests;
mod loading_tests;
mod schema_tests;
mod validation_tests;

/// Resolve the project root (where fdl.yml / fdl.yml.example live) starting
/// from CARGO_MANIFEST_DIR. The CLI crate sits one level down.
pub(super) fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("flodl-cli parent must be project root")
        .to_path_buf()
}

pub(super) fn load_example() -> ProjectConfig {
    let path = project_root().join("fdl.yml.example");
    assert!(
        path.is_file(),
        "fdl.yml.example missing at {} -- the CLI depends on it as the canonical config template",
        path.display()
    );
    load_project(&path).expect("fdl.yml.example must parse as a valid ProjectConfig")
}

pub(super) fn opt(ty: &str) -> OptionSpec {
    OptionSpec {
        ty: ty.into(),
        description: None,
        default: None,
        choices: None,
        short: None,
        env: None,
        completer: None,
    }
}

pub(super) fn arg(name: &str, ty: &str) -> ArgSpec {
    ArgSpec {
        name: name.into(),
        ty: ty.into(),
        description: None,
        required: true,
        variadic: false,
        default: None,
        choices: None,
        completer: None,
    }
}

pub(super) fn schema_with_model_option(strict: bool) -> Schema {
    let mut s = Schema {
        strict,
        ..Schema::default()
    };
    let mut model = opt("string");
    model.short = Some("m".into());
    model.choices = Some(vec![
        serde_json::json!("mlp"),
        serde_json::json!("resnet"),
    ]);
    s.options.insert("model".into(), model);
    s.options.insert("epochs".into(), opt("int"));
    // A bool flag, no value.
    s.options.insert("validate".into(), opt("bool"));
    s
}

pub(super) fn strict_schema_with_model_option() -> Schema {
    schema_with_model_option(true)
}

/// Minimal tempdir helper — matches the pattern used across the crate.
pub(super) struct TempDir(PathBuf);
impl TempDir {
pub(super)     fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "fdl-cfg-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) fn filenames(layers: &[(PathBuf, serde_yaml::Value)]) -> Vec<String> {
    layers
        .iter()
        .map(|(p, _)| {
            p.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string()
        })
        .collect()
}

pub(super) fn canonical_cluster_yaml() -> &'static str {
    "\
cluster:
  controller:
    host: 192.168.122.1
    port: 29500
    path: /opt/flodl
    nccl_socket_ifname: virbr0
  workers:
    - host: master-host
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: virbr0
      path: /opt/flodl
      arch: precompiled/cu128
    - host: worker-host
      ssh:
        target: worker-host
      ranks: [1, 2]
      local_devices: [0, 1]
      nccl_socket_ifname: enp1s0
      path: /srv/flodl
      arch: builds/sm61-sm120

commands:
  cuda-test:
    cluster: true
  train:
    cluster: true
    run: cargo run --release --bin my-training-app
"
}

