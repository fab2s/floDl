//! Hardware detection for flodl: GPUs and host RAM, with **no libtorch,
//! no CUDA runtime, and no dependencies**.
//!
//! # Why this is its own crate
//!
//! Two consumers need the same answers and cannot share code any other
//! way:
//!
//! - **`flodl`** needs GPU identity *before* libtorch is initialized.
//!   flodl's CUDA APIs (e.g. `flodl::tensor::cuda_device_count`)
//!   initialize libtorch on first call, and once libtorch latches onto a
//!   device list, `CUDA_VISIBLE_DEVICES` is ignored. Critically for
//!   cluster mode, the launcher's spawned children then inherit a
//!   corrupted CUDA context on heterogeneous-GPU rigs. See the
//!   "no CUDA before `Trainer::run`" invariant.
//! - **`fdl`** (`flodl-cli`) needs the same answers *before libtorch
//!   exists at all*, to decide which libtorch variant to download or
//!   build. It therefore cannot depend on `flodl`, which would pull
//!   `flodl-sys` and libtorch onto the install path.
//!
//! Both used to carry a hand-synchronized copy of the same struct and
//! the same parser, kept aligned by a comment. This crate is the single
//! source, and the one place a second GPU vendor is added.
//!
//! # Two enumerations, deliberately
//!
//! [`detect_gpus`] reports what the **runtime** will see (visibility
//! masks applied). [`detect_gpus_physical`] reports what is
//! **installed** (masks ignored). Provisioning decisions want the
//! physical set; runtime decisions want the visible set. Conflating them
//! is a real bug in both directions, so they are named apart rather than
//! separated by a boolean argument.
//!
//! # A sweep returns findings, it does not print them
//!
//! Both of the above are shorthands for [`survey`] / [`survey_visible`],
//! which return a [`GpuSurvey`]: the devices **plus** what the sweep
//! learned that a device list cannot express. An empty list has at least
//! four causes needing different responses (CPU-only box, driver present
//! but tool broken, card present but stack not installed, masked away),
//! and the one that matters most for a second vendor is the third: an
//! AMD card with no ROCm is a common state and the user needs told, with
//! an action.
//!
//! Detection therefore records [`SurveyNote`]s and lets the caller
//! decide what to surface, rather than `eprintln!`ing from inside a
//! library. [`GpuSurvey::require_devices`] turns an empty sweep into the
//! best available explanation, which is what a command with an explicit
//! GPU request (`--gpus all`) wants.
//!
//! # Vendor is not device
//!
//! [`GpuVendor`] is an identity. It is deliberately *not* the device
//! string a tensor library is handed: ROCm libtorch keeps `kCUDA`, so an
//! AMD device is `GpuVendor::Amd` here while still being addressed as
//! CUDA at the API surface. Vendor drives detection, diagnostics,
//! packaging and feature derivation; the API surface is a different
//! axis, and assuming they are the same one does not survive contact
//! with Intel (whose libtorch device type genuinely differs).
//!
//! # Contract
//!
//! Nothing here panics, spawns a thread, or initializes a GPU runtime.
//! Every probe is a std-only filesystem read or a bounded subprocess,
//! and each vendor's subprocess is gated behind cheap filesystem checks,
//! so a CPU-only box spawns nothing at all.

mod gpu;
mod mem;
mod nvidia;
mod report;
mod vendor;

pub use gpu::{detect_gpus, detect_gpus_physical, survey, survey_visible, GpuInfo};
pub use mem::{mem_info, MemInfo};
pub use nvidia::nvidia_driver_version;
pub use report::{GpuSurvey, NoteKind, SurveyNote};
pub use vendor::{GpuArch, GpuVendor};
