// shim.h — C interface to libtorch's C++ API.
//
// Rust calls these via extern "C" FFI. The implementations live in
// shim.cpp (compiled as C++) and export C linkage.
//
// Design: every function that can fail returns an error string (caller
// must free it with flodl_free_string). NULL means success.

#ifndef FLODL_SHIM_H
#define FLODL_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a torch::Tensor.
typedef void* FlodlTensor;

// --- Tensor creation ---

char* flodl_zeros(int64_t* shape, int ndim, int dtype, int device_type,
                int device_index, FlodlTensor* result);
char* flodl_ones(int64_t* shape, int ndim, int dtype, int device_type,
               int device_index, FlodlTensor* result);
char* flodl_rand(int64_t* shape, int ndim, int dtype, int device_type,
               int device_index, FlodlTensor* result);
char* flodl_randn(int64_t* shape, int ndim, int dtype, int device_type,
                int device_index, FlodlTensor* result);
char* flodl_from_blob(void* data, int64_t* shape, int ndim, int dtype,
                    int device_type, int device_index, FlodlTensor* result);
char* flodl_linspace(double start, double end, int64_t steps, int dtype,
                   int device_type, int device_index, FlodlTensor* result);
char* flodl_arange(double start, double end, double step, int dtype,
                 int device_type, int device_index, FlodlTensor* result);
char* flodl_expand(FlodlTensor t, int64_t* new_shape, int ndim,
                 FlodlTensor* result);

// --- Tensor lifecycle ---

void flodl_free_tensor(FlodlTensor t);
char* flodl_shallow_clone(FlodlTensor t, FlodlTensor* result);

// --- Tensor metadata ---

int flodl_ndim(FlodlTensor t);
int64_t flodl_shape(FlodlTensor t, int dim);
int flodl_dtype(FlodlTensor t);
int flodl_device_type(FlodlTensor t);
int flodl_device_index(FlodlTensor t);
int64_t flodl_numel(FlodlTensor t);

// --- Data access ---

char* flodl_copy_data(FlodlTensor t, void* buffer, int64_t buffer_bytes);

// --- Arithmetic ---

char* flodl_add(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_sub(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_mul(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_div(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_matmul(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_add_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_mul_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_div_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_neg(FlodlTensor t, FlodlTensor* result);

// --- Activations ---

char* flodl_relu(FlodlTensor t, FlodlTensor* result);
char* flodl_sigmoid(FlodlTensor t, FlodlTensor* result);
char* flodl_tanh_op(FlodlTensor t, FlodlTensor* result);
char* flodl_softmax(FlodlTensor t, int dim, FlodlTensor* result);
char* flodl_log_softmax(FlodlTensor t, int dim, FlodlTensor* result);
char* flodl_gelu(FlodlTensor t, FlodlTensor* result);
char* flodl_gelu_tanh(FlodlTensor t, FlodlTensor* result);
char* flodl_silu(FlodlTensor t, FlodlTensor* result);
char* flodl_leaky_relu(FlodlTensor t, double negative_slope, FlodlTensor* result);
char* flodl_elu(FlodlTensor t, double alpha, FlodlTensor* result);
char* flodl_softplus(FlodlTensor t, double beta, double threshold,
                    FlodlTensor* result);
char* flodl_mish(FlodlTensor t, FlodlTensor* result);
char* flodl_selu(FlodlTensor t, FlodlTensor* result);
char* flodl_hardswish(FlodlTensor t, FlodlTensor* result);
char* flodl_hardsigmoid(FlodlTensor t, FlodlTensor* result);
char* flodl_prelu(FlodlTensor t, FlodlTensor weight, FlodlTensor* result);

// --- Layer normalization ---

char* flodl_native_layer_norm(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                             int64_t normalized_size, double eps,
                             FlodlTensor* output, FlodlTensor* mean, FlodlTensor* rstd);
// --- Group normalization ---

char* flodl_group_norm(FlodlTensor input, int64_t num_groups,
                      FlodlTensor weight, FlodlTensor bias,
                      double eps, FlodlTensor* result);

// --- Element-wise math ---

char* flodl_exp(FlodlTensor t, FlodlTensor* result);
char* flodl_log(FlodlTensor t, FlodlTensor* result);
char* flodl_sqrt(FlodlTensor t, FlodlTensor* result);
char* flodl_abs(FlodlTensor t, FlodlTensor* result);
char* flodl_pow_scalar(FlodlTensor t, double exponent, FlodlTensor* result);
char* flodl_triu(FlodlTensor t, int64_t diagonal, FlodlTensor* result);
char* flodl_tril(FlodlTensor t, int64_t diagonal, FlodlTensor* result);
char* flodl_clamp(FlodlTensor t, double min_val, double max_val,
                FlodlTensor* result);
char* flodl_clamp_min(FlodlTensor t, double min_val, FlodlTensor* result);
char* flodl_clamp_max(FlodlTensor t, double max_val, FlodlTensor* result);
char* flodl_log1p(FlodlTensor t, FlodlTensor* result);
char* flodl_expm1(FlodlTensor t, FlodlTensor* result);
char* flodl_log2(FlodlTensor t, FlodlTensor* result);
char* flodl_log10(FlodlTensor t, FlodlTensor* result);

// --- Reductions ---

char* flodl_sum(FlodlTensor t, FlodlTensor* result);
char* flodl_mean(FlodlTensor t, FlodlTensor* result);
char* flodl_sum_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_mean_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_prod(FlodlTensor t, FlodlTensor* result);
char* flodl_prod_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_cumsum(FlodlTensor t, int dim, FlodlTensor* result);
char* flodl_logsumexp(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_min(FlodlTensor t, FlodlTensor* result);
char* flodl_max(FlodlTensor t, FlodlTensor* result);
char* flodl_norm(FlodlTensor t, FlodlTensor* result);
char* flodl_min_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_max_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_argmax(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_cumprod(FlodlTensor t, int dim, FlodlTensor* result);
char* flodl_norm_p_dim(FlodlTensor t, double p, int dim, int keepdim,
                      FlodlTensor* result);
char* flodl_sum_dims(FlodlTensor t, int64_t* dims, int ndims, int keepdim,
                    FlodlTensor* result);
char* flodl_median(FlodlTensor t, FlodlTensor* result);
char* flodl_median_dim(FlodlTensor t, int dim, int keepdim,
                      FlodlTensor* values, FlodlTensor* indices);
char* flodl_count_nonzero(FlodlTensor t, FlodlTensor* result);
char* flodl_count_nonzero_dim(FlodlTensor t, int dim, FlodlTensor* result);

// --- Query ops ---

char* flodl_nonzero(FlodlTensor t, FlodlTensor* result);
char* flodl_unique(FlodlTensor t, int sorted, int return_inverse,
                  FlodlTensor* output, FlodlTensor* inverse_indices);
char* flodl_searchsorted(FlodlTensor sorted_seq, FlodlTensor values,
                        FlodlTensor* result);

// --- Shape ops (advanced) ---

char* flodl_diagonal(FlodlTensor t, int64_t offset, int dim1, int dim2,
                    FlodlTensor* result);
char* flodl_movedim(FlodlTensor t, int64_t src, int64_t dst,
                   FlodlTensor* result);
char* flodl_tile(FlodlTensor t, int64_t* reps, int ndim, FlodlTensor* result);

// --- Comparison (return float masks: 0.0 or 1.0) ---

char* flodl_gt_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_ge_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_le_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_lt_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_eq_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_ne_scalar(FlodlTensor t, double scalar, FlodlTensor* result);

// --- Boolean / detection (return float masks: 0.0 or 1.0) ---

char* flodl_isnan(FlodlTensor t, FlodlTensor* result);
char* flodl_isinf(FlodlTensor t, FlodlTensor* result);
char* flodl_logical_and(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_logical_or(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_logical_not(FlodlTensor t, FlodlTensor* result);
char* flodl_any(FlodlTensor t, FlodlTensor* result);
char* flodl_all(FlodlTensor t, FlodlTensor* result);

// --- Comparison (tensor-tensor, return float masks: 0.0 or 1.0) ---

char* flodl_gt_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_lt_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_ge_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_le_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_eq_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_ne_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);

// --- Element-wise binary (differentiable) ---

char* flodl_atan2(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_maximum(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_minimum(FlodlTensor a, FlodlTensor b, FlodlTensor* result);

// --- Additional reductions ---

char* flodl_argmin(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_var(FlodlTensor t, FlodlTensor* result);
char* flodl_std_op(FlodlTensor t, FlodlTensor* result);
char* flodl_var_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_std_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);

// --- Element-wise math (trig, rounding, sign) ---

char* flodl_sin(FlodlTensor t, FlodlTensor* result);
char* flodl_cos(FlodlTensor t, FlodlTensor* result);
char* flodl_tan(FlodlTensor t, FlodlTensor* result);
char* flodl_asin(FlodlTensor t, FlodlTensor* result);
char* flodl_acos(FlodlTensor t, FlodlTensor* result);
char* flodl_atan(FlodlTensor t, FlodlTensor* result);
char* flodl_sign(FlodlTensor t, FlodlTensor* result);
char* flodl_floor(FlodlTensor t, FlodlTensor* result);
char* flodl_ceil(FlodlTensor t, FlodlTensor* result);
char* flodl_round(FlodlTensor t, FlodlTensor* result);
char* flodl_reciprocal(FlodlTensor t, FlodlTensor* result);
char* flodl_erf(FlodlTensor t, FlodlTensor* result);
char* flodl_erfc(FlodlTensor t, FlodlTensor* result);
char* flodl_trunc(FlodlTensor t, FlodlTensor* result);
char* flodl_frac(FlodlTensor t, FlodlTensor* result);
char* flodl_fmod_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_fmod_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_remainder_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_remainder_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_lerp(FlodlTensor a, FlodlTensor b, double weight, FlodlTensor* result);
char* flodl_lerp_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor weight,
                       FlodlTensor* result);
char* flodl_isclose(FlodlTensor a, FlodlTensor b, double rtol, double atol,
                   FlodlTensor* result);

// --- Fused mul-add ---

char* flodl_addmm(FlodlTensor bias, FlodlTensor mat1, FlodlTensor mat2,
                  double beta, double alpha, FlodlTensor* result);
char* flodl_addcmul(FlodlTensor self, FlodlTensor t1, FlodlTensor t2,
                   double value, FlodlTensor* result);
char* flodl_addcdiv(FlodlTensor self, FlodlTensor t1, FlodlTensor t2,
                   double value, FlodlTensor* result);

// --- Advanced indexing ---

char* flodl_gather(FlodlTensor t, int dim, FlodlTensor index,
                  FlodlTensor* result);
char* flodl_scatter_add(FlodlTensor t, int dim, FlodlTensor index,
                       FlodlTensor src, FlodlTensor* result);

// --- Sorting ---

char* flodl_topk(FlodlTensor t, int64_t k, int dim, int largest, int sorted,
                FlodlTensor* values, FlodlTensor* indices);
char* flodl_sort(FlodlTensor t, int dim, int descending,
                FlodlTensor* values, FlodlTensor* indices);

// --- Tensor creation ---

char* flodl_eye(int64_t n, int dtype, int device_type, int device_index,
               FlodlTensor* result);
char* flodl_randperm(int64_t n, int dtype, int device_type, int device_index,
                    FlodlTensor* result);
char* flodl_multinomial(FlodlTensor probs, int64_t num_samples,
                       int replacement, FlodlTensor* result);
char* flodl_full(int64_t* shape, int ndim, double value, int dtype,
                int device_type, int device_index, FlodlTensor* result);

// --- Shape operations (additional) ---

char* flodl_chunk(FlodlTensor t, int chunks, int dim,
                 FlodlTensor** results, int* count);
char* flodl_repeat(FlodlTensor t, int64_t* repeats, int ndim,
                  FlodlTensor* result);
char* flodl_pad(FlodlTensor t, int64_t* padding, int pad_len, double value,
               FlodlTensor* result);
// mode: 0=constant, 1=reflect, 2=replicate, 3=circular
char* flodl_pad_mode(FlodlTensor t, int64_t* padding, int pad_len,
                    int mode, double value, FlodlTensor* result);

// --- Interpolation ---
// mode: 0=nearest, 1=bilinear, 2=bicubic, 3=trilinear
char* flodl_interpolate(FlodlTensor input, int64_t* output_size, int ndim,
                       int mode, int align_corners, FlodlTensor* result);

char* flodl_flip(FlodlTensor t, int64_t* dims, int ndim, FlodlTensor* result);
char* flodl_roll(FlodlTensor t, int64_t shift, int dim, FlodlTensor* result);
char* flodl_split(FlodlTensor t, int64_t split_size, int dim,
                 FlodlTensor** results, int* count);
char* flodl_unbind(FlodlTensor t, int dim,
                  FlodlTensor** results, int* count);
char* flodl_contiguous(FlodlTensor t, FlodlTensor* result);
int flodl_is_contiguous(FlodlTensor t);
char* flodl_argsort(FlodlTensor t, int dim, int descending, FlodlTensor* result);
char* flodl_scatter(FlodlTensor t, int dim, FlodlTensor index,
                   FlodlTensor src, FlodlTensor* result);

// --- Shape operations ---

char* flodl_reshape(FlodlTensor t, int64_t* shape, int ndim, FlodlTensor* result);
char* flodl_transpose(FlodlTensor t, int dim0, int dim1, FlodlTensor* result);
char* flodl_permute(FlodlTensor t, int64_t* dims, int ndim, FlodlTensor* result);
char* flodl_select(FlodlTensor t, int dim, int64_t index, FlodlTensor* result);
char* flodl_narrow(FlodlTensor t, int dim, int64_t start, int64_t length,
                 FlodlTensor* result);
char* flodl_squeeze(FlodlTensor t, int dim, FlodlTensor* result);
char* flodl_unsqueeze(FlodlTensor t, int dim, FlodlTensor* result);
char* flodl_flatten(FlodlTensor t, int start_dim, int end_dim, FlodlTensor* result);

// --- Scatter ---

char* flodl_select_scatter(FlodlTensor input, FlodlTensor src, int dim,
                         int64_t index, FlodlTensor* result);
char* flodl_narrow_scatter(FlodlTensor input, FlodlTensor src, int dim,
                         int64_t start, FlodlTensor* result);

// --- Indexing ---

char* flodl_index_select(FlodlTensor t, int dim, FlodlTensor index,
                       FlodlTensor* result);
char* flodl_index_add(FlodlTensor t, int dim, FlodlTensor index,
                    FlodlTensor src, FlodlTensor* result);

// --- Concatenation ---

char* flodl_cat2(FlodlTensor a, FlodlTensor b, int dim, FlodlTensor* result);
char* flodl_cat(FlodlTensor* tensors, int count, int dim, FlodlTensor* result);
char* flodl_stack(FlodlTensor* tensors, int count, int dim, FlodlTensor* result);

// --- Masking ---

char* flodl_masked_fill(FlodlTensor t, FlodlTensor mask, double value,
                       FlodlTensor* result);

// --- Conditional ---

char* flodl_where(FlodlTensor condition, FlodlTensor x, FlodlTensor y,
                FlodlTensor* result);

// --- Like constructors ---

char* flodl_zeros_like(FlodlTensor t, FlodlTensor* result);
char* flodl_ones_like(FlodlTensor t, FlodlTensor* result);
char* flodl_full_like(FlodlTensor t, double value, FlodlTensor* result);
char* flodl_rand_like(FlodlTensor t, FlodlTensor* result);
char* flodl_randn_like(FlodlTensor t, FlodlTensor* result);

// --- Tensor creation (additional) ---

char* flodl_randint(int64_t low, int64_t high, int64_t* shape, int ndim,
                   int dtype, int device_type, int device_index,
                   FlodlTensor* result);
char* flodl_empty(int64_t* shape, int ndim, int dtype, int device_type,
                 int device_index, FlodlTensor* result);
char* flodl_one_hot(FlodlTensor t, int64_t num_classes, FlodlTensor* result);
char* flodl_bernoulli(FlodlTensor t, FlodlTensor* result);

// --- Convolution ---

char* flodl_conv2d(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                 int64_t* stride, int64_t* padding, int64_t* dilation,
                 int64_t groups, FlodlTensor* result);
// --- 1D convolution ---

char* flodl_conv1d(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                 int64_t stride, int64_t padding, int64_t dilation,
                 int64_t groups, FlodlTensor* result);
// --- Transposed convolution ---

char* flodl_conv_transpose2d(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                           int64_t* stride, int64_t* padding,
                           int64_t* output_padding, int64_t* dilation,
                           int64_t groups, FlodlTensor* result);
// --- Transposed 1D convolution ---

char* flodl_conv_transpose1d(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                            int64_t stride, int64_t padding,
                            int64_t output_padding, int64_t dilation,
                            int64_t groups, FlodlTensor* result);
// --- Pooling ---

char* flodl_max_pool2d(FlodlTensor input, int64_t* kernel_size,
                      int64_t* stride, int64_t* padding, int64_t* dilation,
                      int ceil_mode, FlodlTensor* result);
char* flodl_avg_pool2d(FlodlTensor input, int64_t* kernel_size,
                      int64_t* stride, int64_t* padding,
                      int ceil_mode, int count_include_pad,
                      FlodlTensor* result);
char* flodl_adaptive_avg_pool2d(FlodlTensor input, int64_t* output_size,
                              FlodlTensor* result);
char* flodl_adaptive_max_pool2d(FlodlTensor input, int64_t* output_size,
                               FlodlTensor* result);

// --- Unfold / Fold (im2col / col2im) ---

char* flodl_im2col(FlodlTensor input, int64_t* kernel_size, int64_t* dilation,
                  int64_t* padding, int64_t* stride, FlodlTensor* result);
char* flodl_col2im(FlodlTensor input, int64_t* output_size,
                  int64_t* kernel_size, int64_t* dilation,
                  int64_t* padding, int64_t* stride, FlodlTensor* result);

// --- 3D convolution ---

char* flodl_conv3d(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                 int64_t* stride, int64_t* padding, int64_t* dilation,
                 int64_t groups, FlodlTensor* result);
char* flodl_conv_transpose3d(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                            int64_t* stride, int64_t* padding,
                            int64_t* output_padding, int64_t* dilation,
                            int64_t groups, FlodlTensor* result);

// --- 1D pooling ---

char* flodl_max_pool1d(FlodlTensor input, int64_t kernel_size,
                      int64_t stride, int64_t padding, int64_t dilation,
                      int ceil_mode, FlodlTensor* result);
char* flodl_avg_pool1d(FlodlTensor input, int64_t kernel_size,
                      int64_t stride, int64_t padding,
                      int ceil_mode, int count_include_pad,
                      FlodlTensor* result);

// --- Instance normalization ---

char* flodl_instance_norm(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                         FlodlTensor running_mean, FlodlTensor running_var,
                         int use_input_stats, double momentum, double eps,
                         FlodlTensor* result);

// --- PixelShuffle ---

char* flodl_pixel_shuffle(FlodlTensor input, int64_t upscale_factor,
                         FlodlTensor* result);
char* flodl_pixel_unshuffle(FlodlTensor input, int64_t downscale_factor,
                           FlodlTensor* result);

// --- Bilinear ---

char* flodl_bilinear(FlodlTensor input1, FlodlTensor input2,
                    FlodlTensor weight, FlodlTensor bias,
                    FlodlTensor* result);

// --- Grid sampling ---

char* flodl_grid_sample(FlodlTensor input, FlodlTensor grid,
                      int mode, int padding_mode, int align_corners,
                      FlodlTensor* result);

// --- Scaled dot-product attention ---
// Fused attention via at::scaled_dot_product_attention. Pass `attn_mask`
// as nullptr for no mask. `scale <= 0.0` selects the default 1/sqrt(d).
char* flodl_scaled_dot_product_attention(
    FlodlTensor query, FlodlTensor key, FlodlTensor value,
    FlodlTensor attn_mask,
    double dropout_p, int is_causal, double scale,
    FlodlTensor* result);

// --- Fused ops ---

char* flodl_linear(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                  FlodlTensor* result);
char* flodl_gru_cell(FlodlTensor input, FlodlTensor hx,
                    FlodlTensor w_ih, FlodlTensor w_hh,
                    FlodlTensor b_ih, FlodlTensor b_hh,
                    FlodlTensor* result);
char* flodl_lstm_cell(FlodlTensor input, FlodlTensor hx, FlodlTensor cx,
                     FlodlTensor w_ih, FlodlTensor w_hh,
                     FlodlTensor b_ih, FlodlTensor b_hh,
                     FlodlTensor* h_out, FlodlTensor* c_out);

// Fused sequence ops (cuDNN-accelerated full-sequence LSTM/GRU)
char* flodl_lstm(FlodlTensor input, FlodlTensor h_0, FlodlTensor c_0,
                 const FlodlTensor* params, int64_t num_params,
                 int64_t num_layers, bool batch_first, bool flatten,
                 FlodlTensor* output, FlodlTensor* h_n, FlodlTensor* c_n);
char* flodl_gru(FlodlTensor input, FlodlTensor h_0,
                const FlodlTensor* params, int64_t num_params,
                int64_t num_layers, bool batch_first, bool flatten,
                FlodlTensor* output, FlodlTensor* h_n);

// Cached RNN params — zero per-forward overhead (matches PyTorch's single-call pattern)
char* flodl_rnn_params_create(const FlodlTensor* params, int64_t num_params,
                              int64_t mode, int64_t num_layers, bool batch_first,
                              bool flatten, void** out);
void  flodl_rnn_params_free(void* rp);
char* flodl_lstm_cached(FlodlTensor input, FlodlTensor h_0, FlodlTensor c_0,
                        void* rp, int64_t num_layers, bool batch_first,
                        FlodlTensor* output, FlodlTensor* h_n, FlodlTensor* c_n);
char* flodl_gru_cached(FlodlTensor input, FlodlTensor h_0,
                       void* rp, int64_t num_layers, bool batch_first,
                       FlodlTensor* output, FlodlTensor* h_n);

// --- Device ---

char* flodl_to_device(FlodlTensor t, int device_type, int device_index,
                    FlodlTensor* result);

// Non-blocking device transfer. If non_blocking=1 and src is pinned CPU
// memory, the transfer overlaps with host computation.
char* flodl_to_device_async(FlodlTensor t, int device_type, int device_index,
                           FlodlTensor* result);

int flodl_cuda_is_available(void);
int flodl_cuda_device_count(void);
int flodl_force_cuda_link(void);
void flodl_set_cudnn_benchmark(int enable);
void flodl_manual_seed(uint64_t seed);
void flodl_cuda_manual_seed_all(uint64_t seed);
void flodl_set_current_device(int device_index);
int flodl_get_current_device(void);
void flodl_cuda_synchronize(int device_index);

// --- CUDA memory/utilization (monitor support) ---

// Query CUDA memory: writes used and total bytes for a specific device.
// Returns error string on failure (caller must free), NULL on success.
char* flodl_cuda_mem_info(int device_index, uint64_t* used_bytes, uint64_t* total_bytes);

// Query bytes currently handed out by libtorch's CUDA caching allocator.
// This can exceed physical VRAM when unified memory spills to host RAM.
// spill = max(0, allocated - vram_total).
char* flodl_cuda_alloc_bytes(int device_index, uint64_t* allocated_bytes);
char* flodl_cuda_active_bytes(int device_index, uint64_t* active_bytes);

// Peak active allocator bytes (max since last reset).
// Matches torch.cuda.max_memory_allocated() semantics.
char* flodl_cuda_peak_active_bytes(int device_index, uint64_t* peak_bytes);

// Peak reserved allocator bytes (max since last reset).
// Matches torch.cuda.max_memory_reserved() semantics.
char* flodl_cuda_peak_reserved_bytes(int device_index, uint64_t* peak_bytes);

// Reset peak allocator statistics.
// Equivalent to torch.cuda.reset_peak_memory_stats().
void flodl_cuda_reset_peak_stats(int device_index);

// Release all unused cached memory from the CUDA caching allocator.
// Equivalent to torch.cuda.empty_cache().
void flodl_cuda_empty_cache(void);

// Query GPU utilization percentage (0-100) via NVML.
// Returns -1 if NVML is not available or query fails.
int flodl_cuda_utilization(int device_index);

// Query GPU device name (e.g. "NVIDIA GeForce GTX 1060 6GB").
// Writes into caller-provided buffer. Returns error string on failure.
char* flodl_cuda_device_name(int device_index, char* buf, int buf_len);

// Query GPU compute capability (e.g. major=6, minor=1 for sm_61).
// Returns error string on failure, NULL on success.
char* flodl_cuda_compute_capability(int device_index, int* major, int* minor);

// --- Dtype casting ---

char* flodl_to_dtype(FlodlTensor t, int dtype, FlodlTensor* result);
char* flodl_all_finite(FlodlTensor t, int* result);

// --- Autograd ---

char* flodl_set_requires_grad(FlodlTensor t, int requires_grad, FlodlTensor* result);
int flodl_requires_grad(FlodlTensor t);
char* flodl_ensure_grad_accumulator(FlodlTensor t, void** handle_out);
void  flodl_grad_accumulator_delete(void* handle);
char* flodl_backward(FlodlTensor t);
char* flodl_grad(FlodlTensor t, FlodlTensor* result);
char* flodl_set_grad(FlodlTensor t, FlodlTensor grad);
char* flodl_zero_grad(FlodlTensor t);
char* flodl_detach(FlodlTensor t, FlodlTensor* result);
char* flodl_detach_(FlodlTensor t);
int flodl_is_leaf(FlodlTensor t);

// --- Autograd context ---

void* flodl_no_grad_guard_new(void);
void flodl_no_grad_guard_delete(void* guard);
int flodl_is_grad_enabled(void);

// --- Autocast (automatic mixed precision) ---

// Create an autocast guard. While alive, eligible ops (matmul, conv, linear)
// dispatch to the given dtype. device_type: FLODL_CUDA or FLODL_CPU.
// dtype: FLODL_FLOAT16 or FLODL_BFLOAT16.
void* flodl_autocast_guard_new(int device_type, int dtype);
void flodl_autocast_guard_delete(void* guard);
int flodl_is_autocast_enabled(int device_type);

// --- In-place operations ---

char* flodl_add_(FlodlTensor t, FlodlTensor other);
char* flodl_sub_(FlodlTensor t, FlodlTensor other);
char* flodl_mul_scalar_(FlodlTensor t, double scalar);
char* flodl_add_scalar_(FlodlTensor t, double scalar);
char* flodl_zero_(FlodlTensor t);
char* flodl_mul_(FlodlTensor t, FlodlTensor other);
char* flodl_div_scalar_(FlodlTensor t, double scalar);
char* flodl_div_(FlodlTensor t, FlodlTensor other);
char* flodl_fill_(FlodlTensor t, double value);

// --- Meshgrid ---

char* flodl_meshgrid(FlodlTensor* tensors, int count,
                    FlodlTensor** results, int* result_count);

// --- Normalization ---

char* flodl_normalize(FlodlTensor t, double p, int dim, FlodlTensor* result);

// --- Cosine similarity ---

char* flodl_cosine_similarity(FlodlTensor a, FlodlTensor b,
                             int64_t dim, double eps, FlodlTensor* result);

// --- Pairwise distance ---

char* flodl_cdist(FlodlTensor x, FlodlTensor y, double p,
                 FlodlTensor* result);

// --- Fused Adam step ---

// Perform a full Adam/AdamW update in one fused call.
// All tensors (param, m, v) are modified in-place. grad is read-only.
// weight_decay: 0 for Adam, >0 for AdamW (decoupled).
// step: timestep for bias correction (1-based).
// Returns error string or NULL on success.
char* flodl_adam_step(FlodlTensor param, FlodlTensor grad,
                      FlodlTensor m, FlodlTensor v,
                      double lr, double beta1, double beta2, double eps,
                      double weight_decay, int64_t step);

// --- Batched Adam step ---

// Perform Adam/AdamW update on all params in one C++ loop.
// lrs[i] is the learning rate for param i (supports per-group LR).
char* flodl_adam_step_batched(FlodlTensor* params, FlodlTensor* grads,
                              FlodlTensor* ms, FlodlTensor* vs,
                              double* lrs, int count,
                              double beta1, double beta2, double eps,
                              double weight_decay, int64_t step);

// --- Fused Adam/AdamW (multi-tensor, single kernel on CUDA) ---
// Uses libtorch's at::_fused_adam_ / at::_fused_adamw_ to perform the
// complete Adam update across ALL params in one kernel launch on CUDA.
// grad_scale / found_inf: pass NULL to skip (no mixed precision).

// _fused_adam_: L2 weight decay (adds wd*param to gradient).
char* flodl_fused_adam_(FlodlTensor* params, FlodlTensor* grads,
                         FlodlTensor* exp_avgs, FlodlTensor* exp_avg_sqs,
                         int count, double lr,
                         double beta1, double beta2, double eps,
                         double weight_decay, int64_t step,
                         FlodlTensor grad_scale, FlodlTensor found_inf);

// _fused_adamw_: decoupled weight decay (param *= 1 - lr*wd).
char* flodl_fused_adamw_(FlodlTensor* params, FlodlTensor* grads,
                          FlodlTensor* exp_avgs, FlodlTensor* exp_avg_sqs,
                          int count, double lr,
                          double beta1, double beta2, double eps,
                          double weight_decay, int64_t step,
                          FlodlTensor grad_scale, FlodlTensor found_inf);

// --- Pinned memory ---

// Copy a CPU tensor into page-locked (pinned) memory.
// Returns error string or NULL on success.
char* flodl_pin_memory(FlodlTensor t, FlodlTensor* result);

// Returns 1 if the tensor is in pinned memory, 0 otherwise.
int flodl_is_pinned(FlodlTensor t);

// --- Memory diagnostics ---

// Ask glibc to return free memory to the OS. Returns 1 if memory was
// actually released, 0 otherwise. Useful for distinguishing allocator
// fragmentation from real leaks.
int flodl_malloc_trim(void);

// --- Zero grad (set_to_none) ---

// Null out the gradient pointer instead of zeroing the data.
// No CUDA kernel — just resets the grad tensor to undefined.
void flodl_zero_grad_set_to_none(FlodlTensor t);

// --- Fused clip_grad_norm ---

// Compute global L2 norm across all param grads and scale in-place
// if it exceeds max_norm. Replaces 2N FFI calls with 1.
// Writes the original total norm to *total_norm_out.
char* flodl_clip_grad_norm(FlodlTensor* params, int count,
                           double max_norm, double* total_norm_out);

// --- Multi-tensor foreach operations ---
// Use libtorch's _foreach_* ops: batch multiple tensors into fewer kernel
// launches on CUDA. Falls back to per-tensor loops on CPU.

// In-place add scalar: tensors[i] += scalar
char* flodl_foreach_add_scalar_(FlodlTensor* tensors, int count, double scalar);

// In-place multiply by scalar: tensors[i] *= scalar
char* flodl_foreach_mul_scalar_(FlodlTensor* tensors, int count, double scalar);

// In-place zero: tensors[i] = 0
char* flodl_foreach_zero_(FlodlTensor* tensors, int count);

// In-place list add: tensors1[i] += alpha * tensors2[i]
char* flodl_foreach_add_list_(FlodlTensor* tensors1, FlodlTensor* tensors2,
                               int count, double alpha);

// Compute per-tensor L2 norms. results must be pre-allocated array of count
// FlodlTensors. Each result is a new scalar tensor (caller must free).
char* flodl_foreach_norm(FlodlTensor* tensors, int count, double ord,
                          FlodlTensor* results);

// In-place lerp: tensors1[i] += weight * (tensors2[i] - tensors1[i])
char* flodl_foreach_lerp_scalar_(FlodlTensor* tensors1, FlodlTensor* tensors2,
                                  int count, double weight);

// In-place sqrt: tensors[i] = sqrt(tensors[i])
char* flodl_foreach_sqrt_(FlodlTensor* tensors, int count);

// --- Autograd diagnostics ---

// Count unique autograd nodes reachable from this tensor's grad_fn.
// Returns 0 for leaf tensors or tensors without gradient tracking.
int64_t flodl_autograd_node_count(FlodlTensor t);

// --- Fused loss functions ---
// reduction: 0=None, 1=Mean, 2=Sum (matches at::Reduction)

char* flodl_mse_loss(FlodlTensor pred, FlodlTensor target,
                     int64_t reduction, FlodlTensor* result);
char* flodl_cross_entropy_loss(FlodlTensor pred, FlodlTensor target,
                               int64_t reduction, int64_t ignore_index,
                               double label_smoothing, FlodlTensor* result);
char* flodl_bce_with_logits_loss(FlodlTensor pred, FlodlTensor target,
                                  int64_t reduction, FlodlTensor* result);
char* flodl_bce_loss(FlodlTensor pred, FlodlTensor target,
                     int64_t reduction, FlodlTensor* result);
char* flodl_l1_loss(FlodlTensor pred, FlodlTensor target,
                    int64_t reduction, FlodlTensor* result);
char* flodl_smooth_l1_loss(FlodlTensor pred, FlodlTensor target,
                           int64_t reduction, double beta,
                           FlodlTensor* result);
char* flodl_kl_div_loss(FlodlTensor input, FlodlTensor target,
                        int64_t reduction, int log_target,
                        FlodlTensor* result);
char* flodl_nll_loss(FlodlTensor input, FlodlTensor target,
                    int64_t reduction, int64_t ignore_index,
                    FlodlTensor* result);
char* flodl_ctc_loss(FlodlTensor log_probs, FlodlTensor targets,
                    FlodlTensor input_lengths, FlodlTensor target_lengths,
                    int64_t blank, int64_t reduction, FlodlTensor* result);

// --- Fused batch normalization ---

char* flodl_batch_norm(FlodlTensor input, FlodlTensor weight,
                       FlodlTensor bias, FlodlTensor running_mean,
                       FlodlTensor running_var, int training,
                       double momentum, double eps,
                       FlodlTensor* result);

// --- Fused dropout ---

char* flodl_dropout(FlodlTensor input, double p, int training,
                    FlodlTensor* result);
char* flodl_feature_dropout(FlodlTensor input, double p, int training,
                            FlodlTensor* result);

// --- Embedding lookup ---
// Plain embedding: indexes rows of `weight` by `indices`. `padding_idx = -1`
// disables padding; any other valid index zeros out that row's gradient
// during backward (matches PyTorch `nn.functional.embedding`).
char* flodl_embedding(FlodlTensor weight, FlodlTensor indices,
                      int64_t padding_idx,
                      int scale_grad_by_freq, int sparse,
                      FlodlTensor* result);

// --- Embedding bag ---
// Fused embedding lookup + reduction (sum / mean / max).
// mode: 0=sum, 1=mean, 2=max.
char* flodl_embedding_bag(FlodlTensor weight, FlodlTensor indices,
                          FlodlTensor offsets, int64_t mode,
                          FlodlTensor* result);

// --- In-place copy ---

// Copy src into dst in-place (dst.copy_(src)).
// non_blocking: if 1, may be asynchronous for cross-device copies.
char* flodl_copy_(FlodlTensor dst, FlodlTensor src, int non_blocking);

// --- Memory format ---

// Convert tensor to channels-last (NHWC) memory format.
// Only meaningful for 4D tensors (N, C, H, W).
char* flodl_to_channels_last(FlodlTensor t, FlodlTensor* result);

// Returns 1 if tensor is contiguous in channels-last format, 0 otherwise.
int flodl_is_channels_last(FlodlTensor t);

// --- CUDA Graphs ---
// All CUDA Graph functions require CUDA builds. On CPU builds, they
// return an error string.

char* flodl_cuda_graph_new(void** graph_out);
char* flodl_cuda_graph_capture_begin(void* graph, uint64_t pool_hi,
                                      uint64_t pool_lo, int mode);
char* flodl_cuda_graph_capture_end(void* graph);
char* flodl_cuda_graph_replay(void* graph);
char* flodl_cuda_graph_reset(void* graph);
void  flodl_cuda_graph_delete(void* graph);
void  flodl_cuda_graph_pool(void* graph, uint64_t* pool_hi, uint64_t* pool_lo);
void  flodl_cuda_graph_pool_handle(uint64_t* pool_hi, uint64_t* pool_lo);

// --- CUDA Events ---
// All CUDA Event functions require CUDA builds. On CPU builds, they
// return an error string.

// flags: 0 = Default (timing enabled), 1 = DisableTiming (lower overhead)
char* flodl_cuda_event_new(int flags, void** event_out);
char* flodl_cuda_event_record(void* event);
char* flodl_cuda_event_record_on_stream(void* event, void* stream);
char* flodl_cuda_event_synchronize(void* event);
char* flodl_cuda_event_elapsed_time(void* start, void* end, float* ms_out);
int   flodl_cuda_event_query(void* event);
void  flodl_cuda_event_delete(void* event);

// --- CUDA Streams ---
// All CUDA Stream functions require CUDA builds. On CPU builds, they
// return an error string.

char* flodl_cuda_stream_new(int device_index, int high_priority, void** stream_out);
char* flodl_cuda_stream_synchronize(void* stream);
char* flodl_cuda_stream_wait_event(void* stream, void* event);
int   flodl_cuda_stream_query(void* stream);
void  flodl_cuda_stream_set_current(void* stream);
void* flodl_cuda_stream_get_current(int device_index);
void  flodl_cuda_stream_restore_default(int device_index);
void  flodl_cuda_stream_delete(void* stream);

// --- NCCL Collective Operations ---
// All NCCL functions require CUDA builds with 2+ GPUs at runtime.
// On CPU builds, they return an error string.

// Initialize NCCL communicators for ndev devices.
// devlist: array of CUDA device indices (length ndev).
// Returns an opaque handle to the communicator group.
char* flodl_nccl_init(int ndev, const int* devlist, void** handle_out);

// Destroy NCCL communicators.
void flodl_nccl_destroy(void* handle);

// In-place AllReduce across all devices.
// tensors: array of ndev FlodlTensors, one per device (same shape/dtype).
// streams: array of ndev stream handles, or NULL for default streams.
// op: 0=Sum, 1=Prod, 2=Max, 3=Min, 4=Avg
char* flodl_nccl_all_reduce(void* handle, FlodlTensor* tensors,
                             void** streams, int op);

// Broadcast tensor from root device to all others (in-place).
// tensors: array of ndev FlodlTensors, one per device.
// streams: array of ndev stream handles, or NULL for default streams.
// root: rank index (0-based) within the communicator group.
char* flodl_nccl_broadcast(void* handle, FlodlTensor* tensors,
                            void** streams, int root);

// Number of devices in the communicator group.
int flodl_nccl_size(void* handle);

// --- NCCL Per-Rank Operations (for multi-threaded DDP) ---
//
// IMPORTANT: ncclCommInitRank from worker threads corrupts the CUDA context
// on heterogeneous GPU setups (observed: RTX 5060 Ti + GTX 1060). Subsequent
// CUBLAS kernel launches fail with cudaErrorNoKernelImageForDevice.
//
// The safe pattern for multi-threaded DDP is:
//   1. Main thread: flodl_nccl_init() with ncclCommInitAll (creates all comms)
//   2. Main thread: flodl_nccl_split_rank() to extract individual comms
//   3. Worker threads: use extracted comms for flodl_nccl_all_reduce_rank()
//
// The init_rank / get_unique_id functions below are provided for multi-process
// DDP (one process per GPU) where this issue does not apply.

// NCCL unique ID size in bytes.
#define FLODL_NCCL_UNIQUE_ID_BYTES 128

// Generate a unique ID for NCCL communicator initialization.
// uid_out: pointer to at least FLODL_NCCL_UNIQUE_ID_BYTES bytes.
// Called once on any thread; result shared with all ranks.
char* flodl_nccl_get_unique_id(void* uid_out);

// Initialize a single-rank NCCL communicator.
// Caller must set the CUDA device for this rank before calling.
// All ranks must call this concurrently with the same uid/nranks.
char* flodl_nccl_init_rank(int rank, int nranks, const void* uid,
                            void** handle_out);

// Destroy a single-rank NCCL communicator.
void flodl_nccl_destroy_rank(void* handle);

// In-place AllReduce on a single rank's communicator.
// All ranks must call this concurrently for the collective to complete.
// tensors: array of ntensors FlodlTensors (all on this rank's device).
// ntensors: number of tensors to reduce (batched with ncclGroupStart/End).
// stream: CUDA stream handle, or NULL for default stream.
// op: 0=Sum, 1=Prod, 2=Max, 3=Min, 4=Avg
char* flodl_nccl_all_reduce_rank(void* handle, FlodlTensor* tensors,
                                  int ntensors, void* stream, int op);

// Broadcast on a single-rank communicator (cross-process / cross-host).
// root rank's tensors are sent in-place to all other ranks.
// All ranks must call this concurrently for the collective to complete.
// tensors: array of ntensors FlodlTensors (all on this rank's device).
// ntensors: number of tensors to broadcast (batched with ncclGroupStart/End).
// stream: CUDA stream handle, or NULL for default stream.
// root: source rank (in 0..world_size).
char* flodl_nccl_broadcast_rank(void* handle, FlodlTensor* tensors,
                                 int ntensors, void* stream, int root);

// Abort a single-rank communicator, unblocking any in-progress collective.
// Thread-safe: can be called from any thread to unblock a stuck AllReduce.
// After abort, the communicator is destroyed and must not be used again.
// The corresponding flodl_nccl_destroy_rank() becomes a no-op.
char* flodl_nccl_abort_rank(void* handle);

// Extract a single-rank communicator from a group handle.
// Moves ownership: the group's slot for this rank is nullified.
// The returned handle must be freed with flodl_nccl_destroy_rank().
char* flodl_nccl_split_rank(void* group_handle, int rank,
                             void** rank_handle_out);

// --- Utility ---

void flodl_free_string(char* s);

// --- DType constants ---
#define FLODL_FLOAT16  5
#define FLODL_BFLOAT16 15
#define FLODL_FLOAT32  6
#define FLODL_FLOAT64  7
#define FLODL_INT32    3
#define FLODL_INT64    4

// --- Device constants ---
#define FLODL_CPU  0
#define FLODL_CUDA 1

#ifdef __cplusplus
}
#endif

#endif // FLODL_SHIM_H
