//! Shape manipulation operations: reshape, transpose, expand, permute,
//! narrow, select, squeeze, unsqueeze, flatten, cat, stack, repeat, pad, chunk, batches, meshgrid.
//!
//! View semantics (PyTorch parity): `transpose`, `permute`, `narrow`, `select`,
//! `squeeze`, and `unsqueeze` return views sharing the source storage; `reshape`
//! and `flatten` return a view when the layout allows, a copy otherwise. In-place
//! ops on a view write through to the source — that is what makes slice-write
//! idioms like `t.narrow(0, i, 1)?.copy_(&src)` work. [`Tensor::contiguous`]
//! copies only when the layout is non-contiguous (also PyTorch semantics).

use std::ptr;
use flodl_sys::{self as ffi, FlodlTensor};
use super::{Tensor, TensorError, check_err, Result, ffi_call};

impl Tensor {
    /// Reshape to a new shape (must have same total elements).
    /// Use -1 for one inferred dimension.
    /// Returns a view when the layout allows, a copy otherwise.
    ///
    /// ```ignore
    /// let flat = t.reshape(&[-1])?; // [2, 3] -> [6]
    /// ```
    pub fn reshape(&self, shape: &[i64]) -> Result<Tensor> {
        let mut shape = shape.to_vec();
        ffi_call!(flodl_reshape, self.handle, shape.as_mut_ptr(), shape.len() as i32)
    }

    /// Swap two dimensions. Returns a view (shares storage).
    ///
    /// ```ignore
    /// let t = x.transpose(0, 1)?; // [M, N] -> [N, M]
    /// ```
    pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Tensor> {
        ffi_call!(flodl_transpose, self.handle, dim0, dim1)
    }

    /// Broadcast to a larger shape.
    pub fn expand(&self, shape: &[i64]) -> Result<Tensor> {
        let mut shape = shape.to_vec();
        ffi_call!(flodl_expand, self.handle, shape.as_mut_ptr(), shape.len() as i32)
    }

    /// Permute dimensions. Returns a view (shares storage).
    pub fn permute(&self, dims: &[i64]) -> Result<Tensor> {
        let mut dims = dims.to_vec();
        ffi_call!(flodl_permute, self.handle, dims.as_mut_ptr(), dims.len() as i32)
    }

    /// Narrow (slice) along a dimension. Returns a view (shares storage),
    /// so `t.narrow(0, i, 1)?.copy_(&src)` writes into `t`.
    pub fn narrow(&self, dim: i32, start: i64, length: i64) -> Result<Tensor> {
        ffi_call!(flodl_narrow, self.handle, dim, start, length)
    }

    /// Scatter a narrow slice back into a tensor (for narrow backward).
    pub fn narrow_scatter(&self, src: &Tensor, dim: i32, start: i64) -> Result<Tensor> {
        ffi_call!(flodl_narrow_scatter, self.handle, src.handle, dim, start)
    }

    /// Select a single index along a dimension (reduces that dim).
    /// Returns a view (shares storage).
    pub fn select(&self, dim: i32, index: i64) -> Result<Tensor> {
        ffi_call!(flodl_select, self.handle, dim, index)
    }

    /// Squeeze (remove) a dimension of size 1. Returns a view (shares storage).
    pub fn squeeze(&self, dim: i32) -> Result<Tensor> {
        ffi_call!(flodl_squeeze, self.handle, dim)
    }

    /// Unsqueeze (insert) a dimension of size 1. Returns a view (shares storage).
    pub fn unsqueeze(&self, dim: i32) -> Result<Tensor> {
        ffi_call!(flodl_unsqueeze, self.handle, dim)
    }

    /// Insert multiple dimensions of size 1.
    /// Dims are sorted ascending and applied sequentially.
    pub fn unsqueeze_many(&self, dims: &[i32]) -> Result<Tensor> {
        let mut sorted = dims.to_vec();
        sorted.sort();
        let mut t = self.unsqueeze(sorted[0])?;
        for &d in &sorted[1..] {
            t = t.unsqueeze(d)?;
        }
        Ok(t)
    }

    /// Flatten dimensions `[start_dim..=end_dim]` into one.
    /// Returns a view when the layout allows, a copy otherwise.
    pub fn flatten(&self, start_dim: i32, end_dim: i32) -> Result<Tensor> {
        ffi_call!(flodl_flatten, self.handle, start_dim, end_dim)
    }

    /// Concatenate two tensors along a dimension.
    pub fn cat(&self, other: &Tensor, dim: i32) -> Result<Tensor> {
        ffi_call!(flodl_cat2, self.handle, other.handle, dim)
    }

    /// Concatenate multiple tensors along an existing dimension.
    ///
    /// All tensors must have the same shape except in the concatenation dimension.
    /// Uses a single kernel launch regardless of the number of tensors.
    pub fn cat_many(tensors: &[&Tensor], dim: i32) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(TensorError::new("cat_many: empty tensor list"));
        }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut result: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_cat(handles.as_mut_ptr(), handles.len() as i32, dim, &mut result)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(result))
    }

    /// Stack tensors along a new dimension.
    ///
    /// All tensors must have the same shape. A new dimension is inserted at `dim`.
    pub fn stack(tensors: &[&Tensor], dim: i32) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(TensorError::new("stack: empty tensor list"));
        }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut result: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_stack(handles.as_mut_ptr(), handles.len() as i32, dim, &mut result)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(result))
    }

    /// Repeat the tensor along each dimension.
    pub fn repeat(&self, repeats: &[i64]) -> Result<Tensor> {
        let mut repeats = repeats.to_vec();
        ffi_call!(flodl_repeat, self.handle, repeats.as_mut_ptr(), repeats.len() as i32)
    }

    /// Constant-value padding. Padding format matches PyTorch: [left, right, top, bottom, ...].
    pub fn pad(&self, padding: &[i64], value: f64) -> Result<Tensor> {
        let mut padding = padding.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_pad(
                self.handle, padding.as_mut_ptr(), padding.len() as i32,
                value, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Pad with configurable mode. Padding format matches PyTorch: [left, right, ...].
    ///
    /// `mode`: 0=constant, 1=reflect, 2=replicate, 3=circular.
    /// `value`: fill value (only used when mode=constant).
    pub fn pad_mode(&self, padding: &[i64], mode: i32, value: f64) -> Result<Tensor> {
        let mut padding = padding.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_pad_mode(
                self.handle, padding.as_mut_ptr(), padding.len() as i32,
                mode, value, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Reverse the order of elements along the given dimensions.
    pub fn flip(&self, dims: &[i64]) -> Result<Tensor> {
        let mut dims = dims.to_vec();
        ffi_call!(flodl_flip, self.handle, dims.as_mut_ptr(), dims.len() as i32)
    }

    /// Roll elements along a dimension by `shift` positions.
    pub fn roll(&self, shift: i64, dim: i32) -> Result<Tensor> {
        ffi_call!(flodl_roll, self.handle, shift, dim)
    }

    /// Extract the diagonal of a 2D tensor, or a diagonal from a batch.
    pub fn diagonal(&self, offset: i64, dim1: i32, dim2: i32) -> Result<Tensor> {
        ffi_call!(flodl_diagonal, self.handle, offset, dim1, dim2)
    }

    /// Move a dimension from `src` to `dst`.
    pub fn movedim(&self, src: i64, dst: i64) -> Result<Tensor> {
        ffi_call!(flodl_movedim, self.handle, src, dst)
    }

    /// Repeat tensor by tiling the given number of times per dimension.
    pub fn tile(&self, reps: &[i64]) -> Result<Tensor> {
        let mut reps_buf = reps.to_vec();
        ffi_call!(flodl_tile, self.handle, reps_buf.as_mut_ptr(), reps.len() as i32)
    }

    /// Split tensor into pieces of `split_size` along a dimension.
    /// The last piece may be smaller.
    pub fn split(&self, split_size: i64, dim: i32) -> Result<Vec<Tensor>> {
        let mut results_ptr: *mut FlodlTensor = ptr::null_mut();
        let mut count: i32 = 0;
        let err = unsafe {
            ffi::flodl_split(self.handle, split_size, dim, &mut results_ptr, &mut count)
        };
        check_err(err)?;
        let mut tensors = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let handle = unsafe { *results_ptr.add(i) };
            tensors.push(Tensor::from_raw(handle));
        }
        if !results_ptr.is_null() {
            unsafe { ffi::flodl_free_string(results_ptr as *mut std::ffi::c_char) };
        }
        Ok(tensors)
    }

    /// Remove a dimension by unpacking the tensor into a Vec of slices.
    pub fn unbind(&self, dim: i32) -> Result<Vec<Tensor>> {
        let mut results_ptr: *mut FlodlTensor = ptr::null_mut();
        let mut count: i32 = 0;
        let err = unsafe {
            ffi::flodl_unbind(self.handle, dim, &mut results_ptr, &mut count)
        };
        check_err(err)?;
        let mut tensors = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let handle = unsafe { *results_ptr.add(i) };
            tensors.push(Tensor::from_raw(handle));
        }
        if !results_ptr.is_null() {
            unsafe { ffi::flodl_free_string(results_ptr as *mut std::ffi::c_char) };
        }
        Ok(tensors)
    }

    /// Return a contiguous copy if the tensor is not already contiguous.
    pub fn contiguous(&self) -> Result<Tensor> {
        ffi_call!(flodl_contiguous, self.handle)
    }

    /// Split tensor into chunks along a dimension.
    pub fn chunk(&self, chunks: i32, dim: i32) -> Result<Vec<Tensor>> {
        let mut results_ptr: *mut FlodlTensor = ptr::null_mut();
        let mut count: i32 = 0;
        let err = unsafe {
            ffi::flodl_chunk(self.handle, chunks, dim, &mut results_ptr, &mut count)
        };
        check_err(err)?;
        let mut tensors = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let handle = unsafe { *results_ptr.add(i) };
            tensors.push(Tensor::from_raw(handle));
        }
        if !results_ptr.is_null() {
            // Free the C-allocated array (tensors are now owned by Rust).
            // flodl_free_string is just free() -- safe for any malloc'd pointer.
            unsafe { ffi::flodl_free_string(results_ptr as *mut std::ffi::c_char) };
        }
        Ok(tensors)
    }

    /// Split tensor into batches of `batch_size` along dimension 0.
    /// The last batch may be smaller if the tensor size isn't evenly divisible.
    ///
    /// ```ignore
    /// let data = Tensor::randn(&[100, 4], opts)?;
    /// for batch in data.batches(32)? {
    ///     let x = Variable::new(batch, false);
    ///     // ...
    /// }
    /// ```
    pub fn batches(&self, batch_size: i64) -> Result<Vec<Tensor>> {
        let n = self.shape()[0];
        let mut result = Vec::new();
        let mut start = 0i64;
        while start < n {
            let len = (batch_size).min(n - start);
            result.push(self.narrow(0, start, len)?);
            start += len;
        }
        Ok(result)
    }

    /// Compute meshgrid from a slice of 1-D tensors (always "ij" indexing).
    pub fn meshgrid(tensors: &[&Tensor]) -> Result<Vec<Tensor>> {
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut results_ptr: *mut FlodlTensor = ptr::null_mut();
        let mut count: i32 = 0;
        let err = unsafe {
            ffi::flodl_meshgrid(
                handles.as_mut_ptr(), handles.len() as i32,
                &mut results_ptr, &mut count,
            )
        };
        check_err(err)?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let handle = unsafe { *results_ptr.add(i) };
            out.push(Tensor::from_raw(handle));
        }
        if !results_ptr.is_null() {
            unsafe { ffi::flodl_free_string(results_ptr as *mut std::ffi::c_char) };
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_flatten() {
        let t = Tensor::ones(&[2, 3, 4], test_opts()).unwrap();
        let f = t.flatten(1, 2).unwrap();
        assert_eq!(f.shape(), vec![2, 12]);
    }

    #[test]
    fn test_stack() {
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0], &[2], test_device()).unwrap();
        let c = Tensor::from_f32(&[5.0, 6.0], &[2], test_device()).unwrap();

        // Stack along dim 0: [3, 2]
        let s = Tensor::stack(&[&a, &b, &c], 0).unwrap();
        assert_eq!(s.shape(), vec![3, 2]);
        let data = s.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Stack along dim 1: [2, 3]
        let s1 = Tensor::stack(&[&a, &b, &c], 1).unwrap();
        assert_eq!(s1.shape(), vec![2, 3]);
        let data1 = s1.to_f32_vec().unwrap();
        assert_eq!(data1, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_reshape_transpose_narrow_select() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], test_device()).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape(), vec![3, 2]);
        assert_eq!(r.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let tr = t.transpose(0, 1).unwrap();
        assert_eq!(tr.shape(), vec![3, 2]);
        assert_eq!(tr.to_f32_vec().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let n = t.narrow(1, 0, 2).unwrap();
        assert_eq!(n.shape(), vec![2, 2]);
        assert_eq!(n.to_f32_vec().unwrap(), vec![1.0, 2.0, 4.0, 5.0]);

        let s = t.select(0, 1).unwrap();
        assert_eq!(s.shape(), vec![3]);
        assert_eq!(s.to_f32_vec().unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_shape_ops_view_semantics_write_through() {
        // View semantics are deterministic (PyTorch parity), never
        // layout-dependent: narrow aliases along EVERY dim, not just the
        // ones whose slice happens to be contiguous.
        let t = Tensor::zeros(&[2, 3], test_opts()).unwrap();
        let row = Tensor::from_f32(&[7.0, 8.0, 9.0], &[1, 3], test_device()).unwrap();
        t.narrow(0, 1, 1).unwrap().copy_(&row, false).unwrap();
        assert_eq!(t.to_f32_vec().unwrap(), vec![0.0, 0.0, 0.0, 7.0, 8.0, 9.0]);

        let col = Tensor::from_f32(&[5.0, 6.0], &[2, 1], test_device()).unwrap();
        t.narrow(1, 0, 1).unwrap().copy_(&col, false).unwrap();
        assert_eq!(t.to_f32_vec().unwrap(), vec![5.0, 0.0, 0.0, 6.0, 8.0, 9.0]);

        // transpose is a view: writing through it lands in the source.
        let u = Tensor::zeros(&[2, 2], test_opts()).unwrap();
        u.transpose(0, 1).unwrap().narrow(0, 0, 1).unwrap().fill_(3.0).unwrap();
        // Column 0 of u was filled via row 0 of the transpose.
        assert_eq!(u.to_f32_vec().unwrap(), vec![3.0, 0.0, 3.0, 0.0]);

        // Readout of a non-contiguous view stays in logical order
        // (copy-out makes its own contiguous copy at the point of use).
        let v = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let tr = v.transpose(0, 1).unwrap();
        assert!(!tr.is_contiguous());
        assert_eq!(tr.to_f32_vec().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_permute_expand() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], test_device()).unwrap();
        let p = t.permute(&[1, 0]).unwrap();
        assert_eq!(p.shape(), vec![3, 2]);

        let s = Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], test_device()).unwrap();
        let e = s.expand(&[4, 3]).unwrap();
        assert_eq!(e.shape(), vec![4, 3]);
        let data = e.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cat_many() {
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0, 5.0], &[3], test_device()).unwrap();
        let c = Tensor::from_f32(&[6.0], &[1], test_device()).unwrap();

        // Concatenate 3 tensors along dim 0
        let result = Tensor::cat_many(&[&a, &b, &c], 0).unwrap();
        assert_eq!(result.shape(), vec![6]);
        assert_eq!(result.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // 2D: concat along dim 1
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let y = Tensor::from_f32(&[5.0, 6.0], &[2, 1], test_device()).unwrap();
        let z = Tensor::from_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3], test_device()).unwrap();
        let result2 = Tensor::cat_many(&[&x, &y, &z], 1).unwrap();
        assert_eq!(result2.shape(), vec![2, 6]);
        assert_eq!(
            result2.to_f32_vec().unwrap(),
            vec![1.0, 2.0, 5.0, 7.0, 8.0, 9.0, 3.0, 4.0, 6.0, 10.0, 11.0, 12.0]
        );

        // Single tensor -- should just return a copy
        let single = Tensor::cat_many(&[&a], 0).unwrap();
        assert_eq!(single.to_f32_vec().unwrap(), vec![1.0, 2.0]);

        // Empty list -- should error
        let empty: Vec<&Tensor> = vec![];
        assert!(Tensor::cat_many(&empty, 0).is_err());
    }

    #[test]
    fn test_cat_index_select_index_add() {
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0, 5.0], &[3], test_device()).unwrap();
        let c = a.cat(&b, 0).unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let t = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5], test_device()).unwrap();
        let idx = Tensor::from_i64(&[0, 2, 4], &[3], test_device()).unwrap();
        let sel = t.index_select(0, &idx).unwrap();
        assert_eq!(sel.to_f32_vec().unwrap(), vec![10.0, 30.0, 50.0]);

        let base = Tensor::zeros(&[5], test_opts()).unwrap();
        let src = Tensor::from_f32(&[1.0, 1.0, 1.0], &[3], test_device()).unwrap();
        let r = base.index_add(0, &idx, &src).unwrap();
        let data = r.to_f32_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
        assert!((data[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_narrow_scatter_select_scatter() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], test_device()).unwrap();
        let src = Tensor::from_f32(&[10.0, 20.0], &[2], test_device()).unwrap();
        let ns = t.narrow_scatter(&src, 0, 1).unwrap();
        assert_eq!(ns.to_f32_vec().unwrap(), vec![1.0, 10.0, 20.0, 4.0]);

        let t2 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], test_device()).unwrap();
        let row = Tensor::from_f32(&[10.0, 20.0, 30.0], &[3], test_device()).unwrap();
        let ss = t2.select_scatter(&row, 0, 0).unwrap();
        assert_eq!(ss.to_f32_vec().unwrap(), vec![10.0, 20.0, 30.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_chunk_repeat_pad() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], test_device()).unwrap();
        let chunks = t.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].to_f32_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(chunks[1].to_f32_vec().unwrap(), vec![3.0, 4.0]);
        assert_eq!(chunks[2].to_f32_vec().unwrap(), vec![5.0, 6.0]);

        let s = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let rep = s.repeat(&[3]).unwrap();
        assert_eq!(rep.to_f32_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        let pad = s.pad(&[1, 2], 0.0).unwrap();
        assert_eq!(pad.shape(), vec![5]);
        assert_eq!(pad.to_f32_vec().unwrap(), vec![0.0, 1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_unsqueeze_many() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let u = t.unsqueeze_many(&[1, 2]).unwrap();
        assert_eq!(u.shape(), vec![3, 1, 1]);
        // Should match sequential unsqueeze
        let u2 = t.unsqueeze(1).unwrap().unsqueeze(2).unwrap();
        assert_eq!(u.shape(), u2.shape());
        assert_eq!(u.to_f32_vec().unwrap(), u2.to_f32_vec().unwrap());
    }

    #[test]
    fn test_flip() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let f = t.flip(&[0]).unwrap().to_f32_vec().unwrap();
        assert_eq!(f, vec![3.0, 4.0, 1.0, 2.0]);
        let f1 = t.flip(&[1]).unwrap().to_f32_vec().unwrap();
        assert_eq!(f1, vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn test_roll() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], test_device()).unwrap();
        let r = t.roll(1, 0).unwrap().to_f32_vec().unwrap();
        assert_eq!(r, vec![4.0, 1.0, 2.0, 3.0]);
        let r2 = t.roll(-1, 0).unwrap().to_f32_vec().unwrap();
        assert_eq!(r2, vec![2.0, 3.0, 4.0, 1.0]);
    }

    #[test]
    fn test_split() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], test_device()).unwrap();
        let parts = t.split(2, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].to_f32_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(parts[1].to_f32_vec().unwrap(), vec![3.0, 4.0]);
        assert_eq!(parts[2].to_f32_vec().unwrap(), vec![5.0]);
    }

    #[test]
    fn test_unbind() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], test_device()).unwrap();
        let rows = t.unbind(0).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].to_f32_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(rows[1].to_f32_vec().unwrap(), vec![3.0, 4.0]);
        assert_eq!(rows[2].to_f32_vec().unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_contiguous() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let c = t.contiguous().unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), t.to_f32_vec().unwrap());
    }

    #[test]
    fn test_meshgrid() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0], &[2], test_device()).unwrap();
        let grids = Tensor::meshgrid(&[&a, &b]).unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), vec![3, 2]);
        assert_eq!(grids[1].shape(), vec![3, 2]);
        // Grid 0: rows repeat a values
        assert_eq!(grids[0].to_f32_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        // Grid 1: cols repeat b values
        assert_eq!(grids[1].to_f32_vec().unwrap(), vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
    }

    #[test]
    fn test_diagonal() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3], test_device()).unwrap();
        let d = t.diagonal(0, 0, 1).unwrap().to_f32_vec().unwrap();
        assert_eq!(d, vec![1.0, 5.0, 9.0]);
        // Super-diagonal
        let d1 = t.diagonal(1, 0, 1).unwrap().to_f32_vec().unwrap();
        assert_eq!(d1, vec![2.0, 6.0]);
    }

    #[test]
    fn test_movedim() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], test_device()).unwrap();
        let m = t.movedim(0, 1).unwrap();
        assert_eq!(m.shape(), vec![3, 2]);
    }

    #[test]
    fn test_tile() {
        let t = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let r = t.tile(&[3]).unwrap();
        assert_eq!(r.to_f32_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        let t2 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let r2 = t2.tile(&[2, 3]).unwrap();
        assert_eq!(r2.shape(), vec![4, 6]);
    }
}
