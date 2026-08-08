//! Fused neural network operations: layer norm, convolution, linear, RNN cells,
//! pooling, grid sampling, losses, batch norm, and dropout.

use super::{Result, Tensor, check_err, ffi_call};
use flodl_sys::{self as ffi, FlodlTensor};
use std::ptr;

/// Persistent cache of RNN parameter tensors on the C++ side.
///
/// After creation (with optional cuDNN weight flattening), the
/// `std::vector<at::Tensor>` lives in C++ — forward calls pass just
/// this handle, matching PyTorch's single-call `at::lstm()`/`at::gru()`
/// pattern with zero per-forward param collection or FFI overhead.
pub struct RnnParams {
    handle: *mut std::os::raw::c_void,
}

impl RnnParams {
    /// Create a cached parameter set.
    ///
    /// - `mode`: 2 = LSTM, 3 = GRU (cuDNN convention)
    /// - When `flatten` is true, calls `_cudnn_rnn_flatten_weight` to pack
    ///   params into cuDNN's contiguous layout (in-place via `set_()`).
    pub fn new(
        params: &[Tensor],
        mode: i64,
        num_layers: i64,
        batch_first: bool,
        flatten: bool,
    ) -> Result<Self> {
        let handles: Vec<FlodlTensor> = params.iter().map(|t| t.handle).collect();
        let mut out: *mut std::os::raw::c_void = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_rnn_params_create(
                handles.as_ptr(),
                handles.len() as i64,
                mode,
                num_layers,
                batch_first,
                flatten,
                &mut out,
            )
        };
        check_err(err)?;
        Ok(RnnParams { handle: out })
    }
}

impl Drop for RnnParams {
    fn drop(&mut self) {
        unsafe { ffi::flodl_rnn_params_free(self.handle) }
    }
}

impl Tensor {
    /// Native layer normalization. Returns (output, mean, rstd).
    pub fn native_layer_norm(
        &self,
        weight: &Tensor,
        bias: &Tensor,
        normalized_size: i64,
        eps: f64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut out: FlodlTensor = ptr::null_mut();
        let mut mean: FlodlTensor = ptr::null_mut();
        let mut rstd: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_native_layer_norm(
                self.handle,
                weight.handle,
                bias.handle,
                normalized_size,
                eps,
                &mut out,
                &mut mean,
                &mut rstd,
            )
        };
        check_err(err)?;
        Ok((
            Tensor::from_raw(out),
            Tensor::from_raw(mean),
            Tensor::from_raw(rstd),
        ))
    }

    /// 2D convolution. bias may be a null-handle tensor for no bias.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut stride = stride;
        let mut padding = padding;
        let mut dilation = dilation;
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv2d(
                self.handle,
                weight.handle,
                bias_handle,
                stride.as_mut_ptr(),
                padding.as_mut_ptr(),
                dilation.as_mut_ptr(),
                groups,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Transposed 2D convolution.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: [i64; 2],
        padding: [i64; 2],
        output_padding: [i64; 2],
        dilation: [i64; 2],
        groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut stride = stride;
        let mut padding = padding;
        let mut output_padding = output_padding;
        let mut dilation = dilation;
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv_transpose2d(
                self.handle,
                weight.handle,
                bias_handle,
                stride.as_mut_ptr(),
                padding.as_mut_ptr(),
                output_padding.as_mut_ptr(),
                dilation.as_mut_ptr(),
                groups,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// 1D convolution. bias may be None for no bias.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv1d(
                self.handle,
                weight.handle,
                bias_handle,
                stride,
                padding,
                dilation,
                groups,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Transposed 1D convolution.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: i64,
        padding: i64,
        output_padding: i64,
        dilation: i64,
        groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv_transpose1d(
                self.handle,
                weight.handle,
                bias_handle,
                stride,
                padding,
                output_padding,
                dilation,
                groups,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Group normalization. weight and bias are optional (shape `[num_channels]`).
    pub fn group_norm(
        &self,
        num_groups: i64,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let w = weight.map_or(ptr::null_mut(), |t| t.handle);
        let b = bias.map_or(ptr::null_mut(), |t| t.handle);
        let err = unsafe { ffi::flodl_group_norm(self.handle, num_groups, w, b, eps, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused linear: `y = input @ weight^T + bias` (single ATen kernel).
    pub fn linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        ffi_call!(flodl_linear, self.handle, weight.handle, bias_handle)
    }

    /// Fused GRU cell: single ATen `gru_cell` kernel.
    /// Returns new hidden state h'.
    #[allow(clippy::too_many_arguments)]
    pub fn gru_cell(
        &self,
        hx: &Tensor,
        w_ih: &Tensor,
        w_hh: &Tensor,
        b_ih: &Tensor,
        b_hh: &Tensor,
    ) -> Result<Tensor> {
        ffi_call!(
            flodl_gru_cell,
            self.handle,
            hx.handle,
            w_ih.handle,
            w_hh.handle,
            b_ih.handle,
            b_hh.handle,
        )
    }

    /// Fused LSTM cell: single ATen `lstm_cell` kernel.
    /// Returns `(h', c')`.
    #[allow(clippy::too_many_arguments)]
    pub fn lstm_cell(
        &self,
        hx: &Tensor,
        cx: &Tensor,
        w_ih: &Tensor,
        w_hh: &Tensor,
        b_ih: &Tensor,
        b_hh: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut h_out: FlodlTensor = ptr::null_mut();
        let mut c_out: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_lstm_cell(
                self.handle,
                hx.handle,
                cx.handle,
                w_ih.handle,
                w_hh.handle,
                b_ih.handle,
                b_hh.handle,
                &mut h_out,
                &mut c_out,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(h_out), Tensor::from_raw(c_out)))
    }

    /// Fused LSTM sequence: processes all timesteps in a single cuDNN kernel call.
    ///
    /// `params` must be the flat weight list: `[w_ih, w_hh, b_ih, b_hh]` per layer.
    /// When `flatten` is true, calls `_cudnn_rnn_flatten_weight` to pack params into
    /// a contiguous cuDNN-aligned buffer (modifies TensorImpl in-place via `set_()`).
    /// Pass `false` on subsequent calls if params are already flattened.
    /// Returns `(output, h_n, c_n)`.
    pub fn lstm_seq(
        &self,
        h_0: &Tensor,
        c_0: &Tensor,
        params: &[Tensor],
        num_layers: i64,
        batch_first: bool,
        flatten: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let handles: Vec<FlodlTensor> = params.iter().map(|t| t.handle).collect();
        let mut output: FlodlTensor = ptr::null_mut();
        let mut h_n: FlodlTensor = ptr::null_mut();
        let mut c_n: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_lstm(
                self.handle,
                h_0.handle,
                c_0.handle,
                handles.as_ptr(),
                handles.len() as i64,
                num_layers,
                batch_first,
                flatten,
                &mut output,
                &mut h_n,
                &mut c_n,
            )
        };
        check_err(err)?;
        Ok((
            Tensor::from_raw(output),
            Tensor::from_raw(h_n),
            Tensor::from_raw(c_n),
        ))
    }

    /// Fused GRU sequence: processes all timesteps in a single cuDNN kernel call.
    ///
    /// `params` must be the flat weight list: `[w_ih, w_hh, b_ih, b_hh]` per layer.
    /// When `flatten` is true, calls `_cudnn_rnn_flatten_weight` to pack params into
    /// a contiguous cuDNN-aligned buffer. Pass `false` after the first call.
    /// Returns `(output, h_n)`.
    pub fn gru_seq(
        &self,
        h_0: &Tensor,
        params: &[Tensor],
        num_layers: i64,
        batch_first: bool,
        flatten: bool,
    ) -> Result<(Tensor, Tensor)> {
        let handles: Vec<FlodlTensor> = params.iter().map(|t| t.handle).collect();
        let mut output: FlodlTensor = ptr::null_mut();
        let mut h_n: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_gru(
                self.handle,
                h_0.handle,
                handles.as_ptr(),
                handles.len() as i64,
                num_layers,
                batch_first,
                flatten,
                &mut output,
                &mut h_n,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(output), Tensor::from_raw(h_n)))
    }

    /// Fused LSTM using cached C++ params — zero per-forward overhead.
    pub fn lstm_seq_cached(
        &self,
        h_0: &Tensor,
        c_0: &Tensor,
        params: &RnnParams,
        num_layers: i64,
        batch_first: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut output: FlodlTensor = ptr::null_mut();
        let mut h_n: FlodlTensor = ptr::null_mut();
        let mut c_n: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_lstm_cached(
                self.handle,
                h_0.handle,
                c_0.handle,
                params.handle,
                num_layers,
                batch_first,
                &mut output,
                &mut h_n,
                &mut c_n,
            )
        };
        check_err(err)?;
        Ok((
            Tensor::from_raw(output),
            Tensor::from_raw(h_n),
            Tensor::from_raw(c_n),
        ))
    }

    /// Fused GRU using cached C++ params — zero per-forward overhead.
    pub fn gru_seq_cached(
        &self,
        h_0: &Tensor,
        params: &RnnParams,
        num_layers: i64,
        batch_first: bool,
    ) -> Result<(Tensor, Tensor)> {
        let mut output: FlodlTensor = ptr::null_mut();
        let mut h_n: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_gru_cached(
                self.handle,
                h_0.handle,
                params.handle,
                num_layers,
                batch_first,
                &mut output,
                &mut h_n,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(output), Tensor::from_raw(h_n)))
    }

    /// Max pooling over a 2D input (`[B, C, H, W]`).
    ///
    /// Equivalent to `torch.nn.functional.max_pool2d`.
    pub fn max_pool2d(
        &self,
        kernel_size: [i64; 2],
        stride: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        ceil_mode: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut ks = kernel_size;
        let mut st = stride;
        let mut pd = padding;
        let mut dl = dilation;
        let err = unsafe {
            ffi::flodl_max_pool2d(
                self.handle,
                ks.as_mut_ptr(),
                st.as_mut_ptr(),
                pd.as_mut_ptr(),
                dl.as_mut_ptr(),
                ceil_mode as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Average pooling over spatial dimensions.
    pub fn avg_pool2d(
        &self,
        kernel_size: [i64; 2],
        stride: [i64; 2],
        padding: [i64; 2],
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut ks = kernel_size;
        let mut st = stride;
        let mut pd = padding;
        let err = unsafe {
            ffi::flodl_avg_pool2d(
                self.handle,
                ks.as_mut_ptr(),
                st.as_mut_ptr(),
                pd.as_mut_ptr(),
                ceil_mode as i32,
                count_include_pad as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Adaptive average pooling to target spatial size.
    pub fn adaptive_avg_pool2d(&self, output_size: [i64; 2]) -> Result<Tensor> {
        let mut os = output_size;
        ffi_call!(flodl_adaptive_avg_pool2d, self.handle, os.as_mut_ptr())
    }

    /// Grid sampling (bilinear/nearest interpolation).
    pub fn grid_sample(
        &self,
        grid: &Tensor,
        mode: i32,
        padding_mode: i32,
        align_corners: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_grid_sample(
                self.handle,
                grid.handle,
                mode,
                padding_mode,
                align_corners as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Fused loss functions ---

    /// Fused MSE loss: single libtorch kernel.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    pub fn mse_loss(&self, target: &Tensor, reduction: i64) -> Result<Tensor> {
        ffi_call!(flodl_mse_loss, self.handle, target.handle, reduction)
    }

    /// Fused cross-entropy loss: single libtorch kernel.
    /// pred: \[N,C\] logits. target: \[N\] Int64 indices or \[N,C\] Float probs.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_entropy_loss(
        &self,
        target: &Tensor,
        reduction: i64,
        ignore_index: i64,
        label_smoothing: f64,
    ) -> Result<Tensor> {
        ffi_call!(
            flodl_cross_entropy_loss,
            self.handle,
            target.handle,
            reduction,
            ignore_index,
            label_smoothing,
        )
    }

    /// Binary cross-entropy loss from probabilities (NOT logits).
    /// Input must be in \[0, 1\] (e.g. after sigmoid).
    /// reduction: 0=None, 1=Mean, 2=Sum.
    pub fn bce_loss(&self, target: &Tensor, reduction: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err =
            unsafe { ffi::flodl_bce_loss(self.handle, target.handle, reduction, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused BCE with logits loss: single libtorch kernel.
    /// Numerically stable binary cross-entropy from raw logits.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    pub fn bce_with_logits_loss(&self, target: &Tensor, reduction: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_bce_with_logits_loss(self.handle, target.handle, reduction, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused L1 loss: single libtorch kernel.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    pub fn l1_loss(&self, target: &Tensor, reduction: i64) -> Result<Tensor> {
        ffi_call!(flodl_l1_loss, self.handle, target.handle, reduction)
    }

    /// Fused Smooth L1 (Huber) loss: single libtorch kernel.
    /// reduction: 0=None, 1=Mean, 2=Sum. beta: transition point.
    pub fn smooth_l1_loss(&self, target: &Tensor, reduction: i64, beta: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_smooth_l1_loss(self.handle, target.handle, reduction, beta, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused KL divergence loss: single libtorch kernel.
    /// input: log-probabilities. target: probabilities.
    /// reduction: 0=None, 1=Mean, 2=Sum, 5=BatchMean.
    pub fn kl_div_loss(&self, target: &Tensor, reduction: i64, log_target: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_kl_div_loss(
                self.handle,
                target.handle,
                reduction,
                log_target as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Negative log likelihood loss.
    /// `input`: log-probabilities `[N, C]` (output of log_softmax).
    /// `target`: class indices `[N]` (Int64).
    pub fn nll_loss(&self, target: &Tensor, reduction: i64, ignore_index: i64) -> Result<Tensor> {
        ffi_call!(
            flodl_nll_loss,
            self.handle,
            target.handle,
            reduction,
            ignore_index
        )
    }

    /// CTC (Connectionist Temporal Classification) loss for sequence-to-sequence.
    /// `log_probs`: `[T, N, C]` (log-probabilities).
    /// `targets`: `[N, S]` or concatenated 1D.
    /// `input_lengths`/`target_lengths`: `[N]` (Int64).
    pub fn ctc_loss(
        &self,
        targets: &Tensor,
        input_lengths: &Tensor,
        target_lengths: &Tensor,
        blank: i64,
        reduction: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_ctc_loss(
                self.handle,
                targets.handle,
                input_lengths.handle,
                target_lengths.handle,
                blank,
                reduction,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Fused batch normalization ---

    /// Fused batch normalization: single libtorch kernel.
    /// When training=true, updates running_mean/running_var in-place.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm(
        &self,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        running_mean: Option<&Tensor>,
        running_var: Option<&Tensor>,
        training: bool,
        momentum: f64,
        eps: f64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let w = weight.map_or(ptr::null_mut(), |t| t.handle);
        let b = bias.map_or(ptr::null_mut(), |t| t.handle);
        let rm = running_mean.map_or(ptr::null_mut(), |t| t.handle);
        let rv = running_var.map_or(ptr::null_mut(), |t| t.handle);
        let err = unsafe {
            ffi::flodl_batch_norm(
                self.handle,
                w,
                b,
                rm,
                rv,
                training as i32,
                momentum,
                eps,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Fused dropout ---

    /// Fused dropout: single libtorch kernel with inverted scaling.
    pub fn dropout(&self, p: f64, training: bool) -> Result<Tensor> {
        ffi_call!(flodl_dropout, self.handle, p, training as i32)
    }

    /// Fused 2D feature dropout: drops entire channels.
    pub fn feature_dropout(&self, p: f64, training: bool) -> Result<Tensor> {
        ffi_call!(flodl_feature_dropout, self.handle, p, training as i32)
    }

    /// Fused scaled dot-product attention via libtorch's
    /// `at::scaled_dot_product_attention`. Picks flash / mem-efficient / math
    /// implementation at runtime based on inputs, hardware, and dtype.
    ///
    /// Shapes:
    /// - `query`: `[*, Lq, E]`
    /// - `key`: `[*, Lk, E]`
    /// - `value`: `[*, Lk, Ev]`
    /// - `attn_mask`: optional `[*, Lq, Lk]` (broadcastable); additive float
    ///   (added to the scores pre-softmax: 0 = attend, `-inf` = skip) or
    ///   boolean, where **`true` = participate** (libtorch's convention —
    ///   the INVERSE of `masked_fill`-style "true = masked" masks such as
    ///   `nn::MultiheadAttention`'s, which converts to an additive mask at
    ///   the boundary for exactly this reason).
    /// - Output: `[*, Lq, Ev]`.
    ///
    /// `dropout_p`: applied to attention probs (0.0 = disabled).
    /// `is_causal`: apply a causal triangular mask (ignored if `attn_mask`
    /// is also given — libtorch rejects both).
    /// `scale`: `None` uses the default `1/sqrt(E)`.
    pub fn scaled_dot_product_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        dropout_p: f64,
        is_causal: bool,
        scale: Option<f64>,
    ) -> Result<Tensor> {
        let mask_handle = attn_mask.map_or(ptr::null_mut(), |m| m.handle);
        // Sentinel: non-positive scale means "use default".
        let scale_arg = scale.filter(|&s| s > 0.0).unwrap_or(-1.0);
        ffi_call!(
            flodl_scaled_dot_product_attention,
            query.handle,
            key.handle,
            value.handle,
            mask_handle,
            dropout_p,
            if is_causal { 1 } else { 0 },
            scale_arg,
        )
    }

    /// Plain embedding lookup using libtorch's native `at::embedding`.
    ///
    /// `weight`: `[num_embeddings, embedding_dim]` embedding table.
    /// `indices`: i64 tensor of any shape containing token indices.
    /// `padding_idx`: row whose gradient is masked to zero during backward,
    /// or `-1` to disable padding entirely.
    ///
    /// Output shape: `[*indices.shape, embedding_dim]`.
    pub fn embedding(weight: &Tensor, indices: &Tensor, padding_idx: i64) -> Result<Tensor> {
        ffi_call!(
            flodl_embedding,
            weight.handle,
            indices.handle,
            padding_idx,
            /*scale_grad_by_freq=*/ 0,
            /*sparse=*/ 0,
        )
    }

    /// Fused embedding lookup + reduction (sum / mean / max).
    ///
    /// `weight`: `[num_embeddings, embedding_dim]` embedding table.
    /// `indices`: 1-D i64 tensor of token indices.
    /// `offsets`: 1-D i64 tensor marking the start of each bag.
    /// `mode`: 0 = sum, 1 = mean, 2 = max.
    ///
    /// Returns one row per bag with shape `[num_bags, embedding_dim]`.
    pub fn embedding_bag(
        weight: &Tensor,
        indices: &Tensor,
        offsets: &Tensor,
        mode: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_embedding_bag(
                weight.handle,
                indices.handle,
                offsets.handle,
                mode,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Interpolate (resize) a tensor using nearest, bilinear, bicubic, or trilinear mode.
    ///
    /// `output_size`: target spatial dimensions (1D, 2D, or 3D depending on input).
    /// `mode`: 0=nearest, 1=bilinear, 2=bicubic, 3=trilinear.
    /// `align_corners`: whether to align corner pixels (ignored for nearest).
    pub fn interpolate(
        &self,
        output_size: &[i64],
        mode: i32,
        align_corners: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut os = output_size.to_vec();
        let err = unsafe {
            ffi::flodl_interpolate(
                self.handle,
                os.as_mut_ptr(),
                os.len() as i32,
                mode,
                align_corners as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Unfold (im2col): extract sliding local blocks from a 4D input.
    ///
    /// Input: `[N, C, H, W]`.
    /// Output: `[N, C * kH * kW, L]` where L is the number of valid blocks.
    pub fn im2col(
        &self,
        kernel_size: [i64; 2],
        dilation: [i64; 2],
        padding: [i64; 2],
        stride: [i64; 2],
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut ks = kernel_size;
        let mut dl = dilation;
        let mut pd = padding;
        let mut st = stride;
        let err = unsafe {
            ffi::flodl_im2col(
                self.handle,
                ks.as_mut_ptr(),
                dl.as_mut_ptr(),
                pd.as_mut_ptr(),
                st.as_mut_ptr(),
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fold (col2im): reassemble columns back into a 4D image.
    ///
    /// Input: `[N, C * kH * kW, L]`.
    /// Output: `[N, C, output_H, output_W]`.
    pub fn col2im(
        &self,
        output_size: [i64; 2],
        kernel_size: [i64; 2],
        dilation: [i64; 2],
        padding: [i64; 2],
        stride: [i64; 2],
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut os = output_size;
        let mut ks = kernel_size;
        let mut dl = dilation;
        let mut pd = padding;
        let mut st = stride;
        let err = unsafe {
            ffi::flodl_col2im(
                self.handle,
                os.as_mut_ptr(),
                ks.as_mut_ptr(),
                dl.as_mut_ptr(),
                pd.as_mut_ptr(),
                st.as_mut_ptr(),
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// 3D convolution. Input: `[N, C, D, H, W]`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: [i64; 3],
        padding: [i64; 3],
        dilation: [i64; 3],
        groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut stride = stride;
        let mut padding = padding;
        let mut dilation = dilation;
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv3d(
                self.handle,
                weight.handle,
                bias_handle,
                stride.as_mut_ptr(),
                padding.as_mut_ptr(),
                dilation.as_mut_ptr(),
                groups,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Transposed 3D convolution.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose3d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: [i64; 3],
        padding: [i64; 3],
        output_padding: [i64; 3],
        dilation: [i64; 3],
        groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut stride = stride;
        let mut padding = padding;
        let mut output_padding = output_padding;
        let mut dilation = dilation;
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv_transpose3d(
                self.handle,
                weight.handle,
                bias_handle,
                stride.as_mut_ptr(),
                padding.as_mut_ptr(),
                output_padding.as_mut_ptr(),
                dilation.as_mut_ptr(),
                groups,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// 1D max pooling.
    pub fn max_pool1d(
        &self,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        ceil_mode: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_max_pool1d(
                self.handle,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// 1D average pooling.
    pub fn avg_pool1d(
        &self,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_avg_pool1d(
                self.handle,
                kernel_size,
                stride,
                padding,
                ceil_mode as i32,
                count_include_pad as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Adaptive max pooling (returns values only, not indices).
    pub fn adaptive_max_pool2d(&self, output_size: [i64; 2]) -> Result<Tensor> {
        let mut os = output_size;
        ffi_call!(flodl_adaptive_max_pool2d, self.handle, os.as_mut_ptr())
    }

    /// Instance normalization.
    #[allow(clippy::too_many_arguments)]
    pub fn instance_norm(
        &self,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        running_mean: Option<&Tensor>,
        running_var: Option<&Tensor>,
        use_input_stats: bool,
        momentum: f64,
        eps: f64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let w = weight.map_or(ptr::null_mut(), |t| t.handle);
        let b = bias.map_or(ptr::null_mut(), |t| t.handle);
        let rm = running_mean.map_or(ptr::null_mut(), |t| t.handle);
        let rv = running_var.map_or(ptr::null_mut(), |t| t.handle);
        let err = unsafe {
            ffi::flodl_instance_norm(
                self.handle,
                w,
                b,
                rm,
                rv,
                use_input_stats as i32,
                momentum,
                eps,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Pixel shuffle: rearranges `[N, C*r^2, H, W]` to `[N, C, H*r, W*r]`.
    pub fn pixel_shuffle(&self, upscale_factor: i64) -> Result<Tensor> {
        ffi_call!(flodl_pixel_shuffle, self.handle, upscale_factor)
    }

    /// Pixel unshuffle: inverse of pixel_shuffle.
    pub fn pixel_unshuffle(&self, downscale_factor: i64) -> Result<Tensor> {
        ffi_call!(flodl_pixel_unshuffle, self.handle, downscale_factor)
    }

    /// Bilinear transformation: `x1^T A x2 + b`.
    pub fn bilinear(
        input1: &Tensor,
        input2: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_bilinear(
                input1.handle,
                input2.handle,
                weight.handle,
                bias_handle,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }
}
