//! Standard dataset parsers (MNIST, CIFAR-10, Shakespeare, token shards).
//!
//! Pure parsers: accept raw bytes, return tensors. No download logic.
//! Implement [`super::BatchDataSet`] for direct use with [`super::DataLoader`].
//!
//! Disk-backed readers ([`Cifar10Disk`], [`TokenShards`]) read the raw
//! files per sample instead of parsing everything into RAM — the
//! larger-than-RAM path. They implement [`super::DataSet`], so the
//! sample-keyed tiers (RAM cache, disk stage, reservation staging)
//! compose above them automatically.

pub mod mnist;
pub mod cifar10;
pub mod shakespeare;
pub mod token_shards;

pub use mnist::Mnist;
pub use cifar10::{Cifar10, Cifar10Disk};
pub use shakespeare::Shakespeare;
pub use token_shards::{TokenDtype, TokenShards};
