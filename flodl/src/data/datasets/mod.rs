//! Standard dataset parsers (MNIST, CIFAR-10, Shakespeare).
//!
//! Pure parsers: accept raw bytes, return tensors. No download logic.
//! Implement [`super::BatchDataSet`] for direct use with [`super::DataLoader`].
//!
//! Disk-backed readers ([`Cifar10Disk`]) read the same raw files per
//! sample instead of parsing everything into RAM — the
//! larger-than-RAM path. They implement [`super::DataSet`], so the
//! sample-keyed tiers (RAM cache, disk stage, reservation staging)
//! compose above them automatically.

pub mod mnist;
pub mod cifar10;
pub mod shakespeare;

pub use mnist::Mnist;
pub use cifar10::{Cifar10, Cifar10Disk};
pub use shakespeare::Shakespeare;
