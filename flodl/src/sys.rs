//! System-level detection that does NOT touch libtorch / CUDA.
//!
//! flodl's main CUDA APIs (e.g. [`crate::tensor::cuda_device_count`])
//! initialize libtorch on first call. Once libtorch latches onto a
//! device list, `CUDA_VISIBLE_DEVICES` is ignored and — critically for
//! cluster mode — the spawned children inherit a corrupted CUDA
//! context on heterogeneous-GPU rigs.
//!
//! [`detect_gpus`] avoids both problems by shelling out to `nvidia-smi`
//! and parsing its CSV output. Use this when you need GPU info for
//! pre-`Trainer::run` decisions (mode filtering, log banners,
//! CLI-flag validation) — see the "no CUDA before `Trainer::run`"
//! invariant in the [`crate::distributed::Trainer`] docs.
//!
//! For *runtime* GPU queries (after dispatch), the libtorch-backed
//! APIs in [`crate::tensor`] remain the right tool.
//!
//! Returns an empty `Vec` when `nvidia-smi` is missing or fails —
//! callers can treat that as "no CUDA visible" without a separate
//! "did we have a driver" branch.
//!
//! # Where this lives
//!
//! The implementation is the dependency-free [`flodl_hw`] crate, which
//! `flodl-cli` also depends on. `fdl` needs the same answers *before
//! libtorch exists at all* (to pick which variant to install) and so
//! cannot depend on `flodl`; both used to carry a hand-synchronized copy
//! of this struct and this parser. This module is a re-export of the
//! single source, kept as `flodl::sys` because that is the published
//! path.

pub use flodl_hw::{detect_gpus, detect_gpus_physical, GpuInfo, MemInfo, mem_info};
