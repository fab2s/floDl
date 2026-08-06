//! Within-cohort model identity: a SHA-256 over the parameter and
//! buffer manifest (names, dtypes, shapes, in declaration order) of the
//! model a rank trains.
//!
//! This deliberately hashes what the averaging plane depends on, not
//! the binary (which legitimately differs per host: libtorch variant,
//! GPU arch, rustc) and not a source tree (only meaningful when every
//! box built the same one). Two ranks with equal signatures can average
//! each other's parameters; two ranks with different signatures would
//! hang or corrupt each other at the first collective, which is exactly
//! why the coordinator refuses formation on a mismatch instead.
//!
//! The signature is an equality token within one live formation, never
//! persisted: nothing outside a running cohort may store or compare it,
//! so its byte layout is free to change with [`CONTROL_PROTOCOL_VERSION`].
//!
//! [`CONTROL_PROTOCOL_VERSION`]: crate::distributed::wire::CONTROL_PROTOCOL_VERSION

use crate::nn::{Buffer, Parameter};
use hmac_sha256::Hash as Sha256;

/// Hash the manifest in declaration order. Averaging is
/// position-addressed (`parameters()` order IS the reduce order), so an
/// order difference is a real mismatch here, unlike the name-addressed
/// checkpoint hash which sorts.
pub(crate) fn model_sig(params: &[Parameter], buffers: &[Buffer]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for p in params {
        hasher.update(b"P");
        hasher.update(p.name.as_bytes());
        hasher.update(b"\0");
        let data = p.variable.data();
        hasher.update((data.dtype() as i32).to_le_bytes());
        let shape = data.shape();
        hasher.update((shape.len() as u32).to_le_bytes());
        for &dim in &shape {
            hasher.update(dim.to_le_bytes());
        }
    }
    for b in buffers {
        hasher.update(b"B");
        hasher.update(b.name.as_bytes());
        hasher.update(b"\0");
        let t = b.get();
        hasher.update((t.dtype() as i32).to_le_bytes());
        let shape = t.shape();
        hasher.update((shape.len() as u32).to_le_bytes());
        for &dim in &shape {
            hasher.update(dim.to_le_bytes());
        }
    }
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_opts};

    fn param(name: &str, shape: &[i64]) -> Parameter {
        Parameter::new(Tensor::zeros(shape, test_opts()).unwrap(), name)
    }

    fn buffer(name: &str, shape: &[i64]) -> Buffer {
        Buffer::new(Tensor::zeros(shape, test_opts()).unwrap(), name)
    }

    #[test]
    fn same_manifest_same_sig() {
        let a = model_sig(&[param("w", &[3, 2]), param("b", &[2])], &[buffer("rm", &[2])]);
        let b = model_sig(&[param("w", &[3, 2]), param("b", &[2])], &[buffer("rm", &[2])]);
        assert_eq!(a, b);
    }

    #[test]
    fn name_shape_and_dtype_all_bind() {
        let base = model_sig(&[param("w", &[3, 2])], &[]);
        assert_ne!(base, model_sig(&[param("w2", &[3, 2])], &[]), "name must bind");
        assert_ne!(base, model_sig(&[param("w", &[2, 3])], &[]), "shape must bind");
        let f64_w = Parameter::new(
            Tensor::zeros(
                &[3, 2],
                crate::tensor::TensorOptions {
                    dtype: crate::tensor::DType::Float64,
                    ..test_opts()
                },
            )
            .unwrap(),
            "w",
        );
        assert_ne!(base, model_sig(&[f64_w], &[]), "dtype must bind");
    }

    #[test]
    fn declaration_order_binds() {
        let ab = model_sig(&[param("a", &[2]), param("b", &[2])], &[]);
        let ba = model_sig(&[param("b", &[2]), param("a", &[2])], &[]);
        assert_ne!(ab, ba, "averaging is position-addressed; order is identity");
    }

    #[test]
    fn params_and_buffers_are_distinct_namespaces() {
        let as_param = model_sig(&[param("rm", &[2])], &[]);
        let as_buffer = model_sig(&[], &[buffer("rm", &[2])]);
        assert_ne!(as_param, as_buffer);
    }
}
