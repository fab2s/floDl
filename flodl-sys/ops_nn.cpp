// ops_nn.cpp — neural-network-centric tensor ops.
//
// Covers: convolution (1D/2D/3D + transposed), pooling (max/avg/adaptive),
// unfold/fold (im2col), instance normalization, pixel_shuffle / pixel_unshuffle,
// bilinear, grid_sample, dtype casting, device transfer (sync/async),
// CUDA memory/utilization/device-info utilities.

#include "helpers.h"

#ifdef FLODL_BUILD_CUDA
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

// --- Convolution ---

extern "C" char* flodl_conv2d(FlodlTensor input, FlodlTensor weight,
                             FlodlTensor bias,
                             int64_t* stride, int64_t* padding,
                             int64_t* dilation,
                             int64_t groups, FlodlTensor* result) {
    try {
        auto in = unwrap(input);
        auto w = unwrap(weight);
        c10::optional<torch::Tensor> b;
        if (bias != nullptr) {
            b = unwrap(bias);
        }
        *result = wrap(torch::conv2d(in, w, b,
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(dilation, 2),
            groups));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- 1D convolution ---

extern "C" char* flodl_conv1d(FlodlTensor input, FlodlTensor weight,
                             FlodlTensor bias,
                             int64_t stride, int64_t padding,
                             int64_t dilation,
                             int64_t groups, FlodlTensor* result) {
    try {
        auto in = unwrap(input);
        auto w = unwrap(weight);
        c10::optional<torch::Tensor> b;
        if (bias != nullptr) {
            b = unwrap(bias);
        }
        *result = wrap(torch::conv1d(in, w, b,
            /*stride=*/{stride},
            /*padding=*/{padding},
            /*dilation=*/{dilation},
            groups));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Transposed convolution ---

extern "C" char* flodl_conv_transpose2d(FlodlTensor input, FlodlTensor weight,
                                       FlodlTensor bias,
                                       int64_t* stride, int64_t* padding,
                                       int64_t* output_padding, int64_t* dilation,
                                       int64_t groups, FlodlTensor* result) {
    try {
        auto in = unwrap(input);
        auto w = unwrap(weight);
        c10::optional<torch::Tensor> b;
        if (bias != nullptr) {
            b = unwrap(bias);
        }
        *result = wrap(torch::conv_transpose2d(in, w, b,
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(output_padding, 2),
            groups,
            torch::IntArrayRef(dilation, 2)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Transposed 1D convolution ---

extern "C" char* flodl_conv_transpose1d(FlodlTensor input, FlodlTensor weight,
                                        FlodlTensor bias,
                                        int64_t stride, int64_t padding,
                                        int64_t output_padding, int64_t dilation,
                                        int64_t groups, FlodlTensor* result) {
    try {
        auto in = unwrap(input);
        auto w = unwrap(weight);
        c10::optional<torch::Tensor> b;
        if (bias != nullptr) {
            b = unwrap(bias);
        }
        *result = wrap(torch::conv_transpose1d(in, w, b,
            /*stride=*/{stride},
            /*padding=*/{padding},
            /*output_padding=*/{output_padding},
            groups,
            /*dilation=*/{dilation}));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Pooling ---

extern "C" char* flodl_max_pool2d(FlodlTensor input, int64_t* kernel_size,
                                 int64_t* stride, int64_t* padding, int64_t* dilation,
                                 int ceil_mode, FlodlTensor* result) {
    try {
        *result = wrap(at::max_pool2d(
            unwrap(input),
            torch::IntArrayRef(kernel_size, 2),
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(dilation, 2),
            ceil_mode != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_avg_pool2d(FlodlTensor input, int64_t* kernel_size,
                                   int64_t* stride, int64_t* padding,
                                   int ceil_mode, int count_include_pad,
                                   FlodlTensor* result) {
    try {
        *result = wrap(at::avg_pool2d(
            unwrap(input),
            torch::IntArrayRef(kernel_size, 2),
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            ceil_mode != 0,
            count_include_pad != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_adaptive_avg_pool2d(FlodlTensor input, int64_t* output_size,
                                          FlodlTensor* result) {
    try {
        *result = wrap(at::adaptive_avg_pool2d(
            unwrap(input), torch::IntArrayRef(output_size, 2)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_adaptive_max_pool2d(FlodlTensor input, int64_t* output_size,
                                           FlodlTensor* result) {
    try {
        auto [out, _indices] = at::adaptive_max_pool2d(
            unwrap(input), torch::IntArrayRef(output_size, 2));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Unfold / Fold (im2col / col2im) ---

extern "C" char* flodl_im2col(FlodlTensor input, int64_t* kernel_size,
                              int64_t* dilation, int64_t* padding,
                              int64_t* stride, FlodlTensor* result) {
    try {
        *result = wrap(at::im2col(unwrap(input),
                                  torch::IntArrayRef(kernel_size, 2),
                                  torch::IntArrayRef(dilation, 2),
                                  torch::IntArrayRef(padding, 2),
                                  torch::IntArrayRef(stride, 2)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_col2im(FlodlTensor input, int64_t* output_size,
                              int64_t* kernel_size, int64_t* dilation,
                              int64_t* padding, int64_t* stride,
                              FlodlTensor* result) {
    try {
        *result = wrap(at::col2im(unwrap(input),
                                  torch::IntArrayRef(output_size, 2),
                                  torch::IntArrayRef(kernel_size, 2),
                                  torch::IntArrayRef(dilation, 2),
                                  torch::IntArrayRef(padding, 2),
                                  torch::IntArrayRef(stride, 2)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- 3D convolution ---

extern "C" char* flodl_conv3d(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                              int64_t* stride, int64_t* padding, int64_t* dilation,
                              int64_t groups, FlodlTensor* result) {
    try {
        auto b = bias ? torch::optional<torch::Tensor>(unwrap(bias))
                      : torch::optional<torch::Tensor>();
        *result = wrap(at::conv3d(unwrap(input), unwrap(weight), b,
                                  torch::IntArrayRef(stride, 3),
                                  torch::IntArrayRef(padding, 3),
                                  torch::IntArrayRef(dilation, 3), groups));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_conv_transpose3d(FlodlTensor input, FlodlTensor weight,
                                        FlodlTensor bias,
                                        int64_t* stride, int64_t* padding,
                                        int64_t* output_padding, int64_t* dilation,
                                        int64_t groups, FlodlTensor* result) {
    try {
        auto b = bias ? torch::optional<torch::Tensor>(unwrap(bias))
                      : torch::optional<torch::Tensor>();
        *result = wrap(at::conv_transpose3d(unwrap(input), unwrap(weight), b,
                                            torch::IntArrayRef(stride, 3),
                                            torch::IntArrayRef(padding, 3),
                                            torch::IntArrayRef(output_padding, 3),
                                            groups,
                                            torch::IntArrayRef(dilation, 3)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- 1D pooling ---

extern "C" char* flodl_max_pool1d(FlodlTensor input, int64_t kernel_size,
                                  int64_t stride, int64_t padding, int64_t dilation,
                                  int ceil_mode, FlodlTensor* result) {
    try {
        *result = wrap(at::max_pool1d(unwrap(input), {kernel_size},
                                      {stride}, {padding}, {dilation},
                                      ceil_mode != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_avg_pool1d(FlodlTensor input, int64_t kernel_size,
                                  int64_t stride, int64_t padding,
                                  int ceil_mode, int count_include_pad,
                                  FlodlTensor* result) {
    try {
        *result = wrap(at::avg_pool1d(unwrap(input), {kernel_size},
                                      {stride}, {padding},
                                      ceil_mode != 0, count_include_pad != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Instance normalization ---

extern "C" char* flodl_instance_norm(FlodlTensor input, FlodlTensor weight,
                                     FlodlTensor bias,
                                     FlodlTensor running_mean, FlodlTensor running_var,
                                     int use_input_stats, double momentum, double eps,
                                     FlodlTensor* result) {
    try {
        auto w = weight ? torch::optional<torch::Tensor>(unwrap(weight))
                        : torch::optional<torch::Tensor>();
        auto b = bias ? torch::optional<torch::Tensor>(unwrap(bias))
                      : torch::optional<torch::Tensor>();
        auto rm = running_mean ? torch::optional<torch::Tensor>(unwrap(running_mean))
                               : torch::optional<torch::Tensor>();
        auto rv = running_var ? torch::optional<torch::Tensor>(unwrap(running_var))
                              : torch::optional<torch::Tensor>();
        *result = wrap(at::instance_norm(unwrap(input), w, b, rm, rv,
                                         use_input_stats != 0, momentum, eps, false));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- PixelShuffle ---

extern "C" char* flodl_pixel_shuffle(FlodlTensor input, int64_t upscale_factor,
                                     FlodlTensor* result) {
    try {
        *result = wrap(at::pixel_shuffle(unwrap(input), upscale_factor));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_pixel_unshuffle(FlodlTensor input, int64_t downscale_factor,
                                       FlodlTensor* result) {
    try {
        *result = wrap(at::pixel_unshuffle(unwrap(input), downscale_factor));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Bilinear ---

extern "C" char* flodl_bilinear(FlodlTensor input1, FlodlTensor input2,
                                FlodlTensor weight, FlodlTensor bias,
                                FlodlTensor* result) {
    try {
        auto b = bias ? torch::optional<torch::Tensor>(unwrap(bias))
                      : torch::optional<torch::Tensor>();
        *result = wrap(at::bilinear(unwrap(input1), unwrap(input2),
                                    unwrap(weight), b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Grid sampling ---

extern "C" char* flodl_grid_sample(FlodlTensor input, FlodlTensor grid,
                                  int mode, int padding_mode,
                                  int align_corners, FlodlTensor* result) {
    try {
        *result = wrap(at::grid_sampler(
            unwrap(input), unwrap(grid), mode, padding_mode, align_corners != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Scaled dot-product attention ---
// Fused self/cross attention — libtorch picks flash / mem-efficient / math
// at runtime based on inputs, hardware, and dtype. One kernel call replaces
// the naive matmul+softmax+matmul chain in most cases.
//
// Shapes:
//   query: [*, Lq, E]   key: [*, Lk, E]   value: [*, Lk, Ev]
//   attn_mask (nullable): [*, Lq, Lk] or broadcastable; additive (float) or
//     boolean. Pass nullptr for no mask.
//   result: [*, Lq, Ev]
//
// `dropout_p` applies to attention probs (post-softmax); 0.0 disables.
// `scale <= 0.0` is a sentinel meaning "use default 1/sqrt(last_dim)".
extern "C" char* flodl_scaled_dot_product_attention(
    FlodlTensor query, FlodlTensor key, FlodlTensor value,
    FlodlTensor attn_mask,
    double dropout_p, int is_causal, double scale,
    FlodlTensor* result) {
    try {
        c10::optional<at::Tensor> mask;
        if (attn_mask != nullptr) {
            mask = unwrap(attn_mask);
        }
        c10::optional<double> scale_opt;
        if (scale > 0.0) {
            scale_opt = scale;
        }
        *result = wrap(at::scaled_dot_product_attention(
            unwrap(query), unwrap(key), unwrap(value),
            mask, dropout_p, is_causal != 0, scale_opt));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Dtype casting ---

extern "C" char* flodl_to_dtype(FlodlTensor t, int dtype, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).to(to_scalar_type(dtype)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_all_finite(FlodlTensor t, int* result) {
    try {
        auto& tensor = unwrap(t);
        *result = torch::isfinite(tensor).all().item<bool>() ? 1 : 0;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Device operations ---

extern "C" char* flodl_to_device(FlodlTensor t, int device_type,
                                int device_index, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).to(to_device(device_type, device_index)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_to_device_async(FlodlTensor t, int device_type,
                                       int device_index, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).to(to_device(device_type, device_index),
                                    /*non_blocking=*/true));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" int flodl_cuda_is_available(void) {
    try {
    return torch::cuda::is_available() ? 1 : 0;
    } catch (const std::exception& e) {
        flodl_fatal("flodl_cuda_is_available", e.what());
    } catch (...) {
        flodl_fatal("flodl_cuda_is_available", nullptr);
    }
}

extern "C" int flodl_cuda_device_count(void) {
    try {
    return (int)torch::cuda::device_count();
    } catch (const std::exception& e) {
        flodl_fatal("flodl_cuda_device_count", e.what());
    } catch (...) {
        flodl_fatal("flodl_cuda_device_count", nullptr);
    }
}

extern "C" void flodl_set_current_device(int device_index) {
    try {
#ifdef FLODL_BUILD_CUDA
    c10::cuda::set_device((c10::DeviceIndex)device_index);
#else
    (void)device_index;
#endif
    } catch (const std::exception& e) {
        flodl_fatal("flodl_set_current_device", e.what());
    } catch (...) {
        flodl_fatal("flodl_set_current_device", nullptr);
    }
}

extern "C" int flodl_get_current_device(void) {
    try {
#ifdef FLODL_BUILD_CUDA
    return (int)c10::cuda::current_device();
#else
    return 0;
#endif
    } catch (const std::exception& e) {
        flodl_fatal("flodl_get_current_device", e.what());
    } catch (...) {
        flodl_fatal("flodl_get_current_device", nullptr);
    }
}

extern "C" void flodl_cuda_synchronize(int device_index) {
    try {
#ifdef FLODL_BUILD_CUDA
    if (torch::cuda::is_available()) {
        c10::cuda::set_device((c10::DeviceIndex)device_index);
        cudaDeviceSynchronize();
    }
#else
    (void)device_index;
#endif
    } catch (const std::exception& e) {
        flodl_fatal("flodl_cuda_synchronize", e.what());
    } catch (...) {
        flodl_fatal("flodl_cuda_synchronize", nullptr);
    }
}

// Force real symbol references to BOTH c10_cuda.so and libtorch_cuda.so
// so that --as-needed doesn't drop them from the link. Without this,
// no Rust code directly references symbols in these libraries, so the
// linker silently drops them and torch::cuda::is_available() returns
// false even with a GPU present.
//
// c10_cuda.so      -> c10::cuda::device_count()
// libtorch_cuda.so -> torch::CudaIPCCollect() (static initializers in
//                     this library register the CUDA backend)
#ifdef FLODL_BUILD_CUDA
#include <cuda_runtime.h>
#include <dlfcn.h>
namespace torch { void CudaIPCCollect(); }
#endif

extern "C" int flodl_force_cuda_link(void) {
    try {
#ifdef FLODL_BUILD_CUDA
    // c10_cuda.so dependency
    volatile int n = (int)c10::cuda::device_count();
    // libtorch_cuda.so dependency -- take address, don't call
    volatile auto p = &torch::CudaIPCCollect;
    (void)p;
    return n;
#else
    return 0;
#endif
    } catch (const std::exception& e) {
        flodl_fatal("flodl_force_cuda_link", e.what());
    } catch (...) {
        flodl_fatal("flodl_force_cuda_link", nullptr);
    }
}

// --- CUDA memory info via cudaMemGetInfo ---

extern "C" char* flodl_cuda_mem_info(int device_index,
                                    uint64_t* used_bytes, uint64_t* total_bytes) {
    try {
#ifdef FLODL_BUILD_CUDA
    if (!torch::cuda::is_available()) {
        return make_error("CUDA not available");
    }
    // Switch to target device, query, then restore
    auto prev = c10::cuda::current_device();
    c10::cuda::set_device((c10::DeviceIndex)device_index);
    size_t free_b = 0, total_b = 0;
    auto err = cudaMemGetInfo(&free_b, &total_b);
    c10::cuda::set_device(prev);
    if (err != cudaSuccess) {
        return make_error(cudaGetErrorString(err));
    }
    *total_bytes = (uint64_t)total_b;
    *used_bytes  = (uint64_t)(total_b - free_b);
    return nullptr;
#else
    (void)device_index; (void)used_bytes; (void)total_bytes;
    return make_error("CUDA not available (built without cuda feature)");
#endif
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- CUDA caching allocator stats ---

extern "C" char* flodl_cuda_alloc_bytes(int device_index,
                                         uint64_t* allocated_bytes) {
#ifdef FLODL_BUILD_CUDA
    if (!torch::cuda::is_available()) {
        return make_error("CUDA not available");
    }
    try {
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
            (c10::DeviceIndex)device_index);
        // reserved_bytes = total memory grabbed from CUDA driver (including
        // unified-memory spill to host RAM).  allocated_bytes only counts
        // actively-used sub-blocks, which never exceeds physical VRAM.
        *allocated_bytes = (uint64_t)stats.reserved_bytes[0].current;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
#else
    (void)device_index; (void)allocated_bytes;
    return make_error("CUDA not available (built without cuda feature)");
#endif
}

// Active allocator bytes (tensors actually in use, not cached free blocks).
// Matches torch.cuda.memory_allocated() semantics (current, not peak).
extern "C" char* flodl_cuda_active_bytes(int device_index,
                                          uint64_t* active_bytes) {
#ifdef FLODL_BUILD_CUDA
    if (!torch::cuda::is_available()) {
        return make_error("CUDA not available");
    }
    try {
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
            (c10::DeviceIndex)device_index);
        *active_bytes = (uint64_t)stats.allocated_bytes[0].current;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
#else
    (void)device_index; (void)active_bytes;
    return make_error("CUDA not available (built without cuda feature)");
#endif
}

// Peak active allocator bytes (max since last reset).
// Matches torch.cuda.max_memory_allocated() semantics.
extern "C" char* flodl_cuda_peak_active_bytes(int device_index,
                                               uint64_t* peak_bytes) {
#ifdef FLODL_BUILD_CUDA
    if (!torch::cuda::is_available()) {
        return make_error("CUDA not available");
    }
    try {
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
            (c10::DeviceIndex)device_index);
        *peak_bytes = (uint64_t)stats.allocated_bytes[0].peak;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
#else
    (void)device_index; (void)peak_bytes;
    return make_error("CUDA not available (built without cuda feature)");
#endif
}

// Peak reserved allocator bytes (max since last reset).
// Matches torch.cuda.max_memory_reserved() semantics.
extern "C" char* flodl_cuda_peak_reserved_bytes(int device_index,
                                                  uint64_t* peak_bytes) {
#ifdef FLODL_BUILD_CUDA
    if (!torch::cuda::is_available()) {
        return make_error("CUDA not available");
    }
    try {
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
            (c10::DeviceIndex)device_index);
        *peak_bytes = (uint64_t)stats.reserved_bytes[0].peak;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
#else
    (void)device_index; (void)peak_bytes;
    return make_error("CUDA not available (built without cuda feature)");
#endif
}

// Reset peak allocator statistics.
// Equivalent to torch.cuda.reset_peak_memory_stats().
extern "C" void flodl_cuda_reset_peak_stats(int device_index) {
    try {
#ifdef FLODL_BUILD_CUDA
    c10::cuda::CUDACachingAllocator::resetPeakStats((c10::DeviceIndex)device_index);
#else
    (void)device_index;
#endif
    } catch (const std::exception& e) {
        flodl_fatal("flodl_cuda_reset_peak_stats", e.what());
    } catch (...) {
        flodl_fatal("flodl_cuda_reset_peak_stats", nullptr);
    }
}

// --- CUDA empty cache ---

extern "C" void flodl_cuda_empty_cache(void) {
    try {
#ifdef FLODL_BUILD_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    } catch (const std::exception& e) {
        flodl_fatal("flodl_cuda_empty_cache", e.what());
    } catch (...) {
        flodl_fatal("flodl_cuda_empty_cache", nullptr);
    }
}

// --- GPU utilization via NVML (dynamically loaded) ---

#ifdef FLODL_BUILD_CUDA
namespace {
    typedef int nvml_ret_t;
    typedef void* nvml_device_t;
    struct NvmlUtil { unsigned int gpu; unsigned int memory; };

    struct NvmlState {
        bool tried = false;
        bool ok = false;
        nvml_ret_t (*init)(void) = nullptr;
        nvml_ret_t (*getHandle)(unsigned int, nvml_device_t*) = nullptr;
        nvml_ret_t (*getUtil)(nvml_device_t, NvmlUtil*) = nullptr;
    };
    static NvmlState nvml;

    static void nvml_try_load() {
        if (nvml.tried) return;
        nvml.tried = true;
        void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!lib) return;
        nvml.init      = (decltype(nvml.init))dlsym(lib, "nvmlInit_v2");
        nvml.getHandle = (decltype(nvml.getHandle))dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
        nvml.getUtil   = (decltype(nvml.getUtil))dlsym(lib, "nvmlDeviceGetUtilizationRates");
        if (!nvml.init || !nvml.getHandle || !nvml.getUtil) return;
        nvml.ok = (nvml.init() == 0);
    }
} // anonymous namespace
#endif

extern "C" int flodl_cuda_utilization(int device_index) {
    try {
#ifdef FLODL_BUILD_CUDA
    nvml_try_load();
    if (!nvml.ok) return -1;
    nvml_device_t dev;
    if (nvml.getHandle((unsigned int)device_index, &dev) != 0) return -1;
    NvmlUtil util;
    if (nvml.getUtil(dev, &util) != 0) return -1;
    return (int)util.gpu;
#else
    (void)device_index;
    return -1;
#endif
    } catch (const std::exception& e) {
        flodl_fatal("flodl_cuda_utilization", e.what());
    } catch (...) {
        flodl_fatal("flodl_cuda_utilization", nullptr);
    }
}

// --- GPU device name ---

extern "C" char* flodl_cuda_device_name(int device_index, char* buf, int buf_len) {
    try {
#ifdef FLODL_BUILD_CUDA
    if (!torch::cuda::is_available()) {
        return make_error("CUDA not available");
    }
    cudaDeviceProp prop;
    auto err = cudaGetDeviceProperties(&prop, device_index);
    if (err != cudaSuccess) {
        return make_error(cudaGetErrorString(err));
    }
    snprintf(buf, buf_len, "%s", prop.name);
    return nullptr;
#else
    (void)device_index; (void)buf; (void)buf_len;
    return make_error("CUDA not available (built without cuda feature)");
#endif
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_cuda_compute_capability(int device_index,
                                                 int* major, int* minor) {
    try {
#ifdef FLODL_BUILD_CUDA
    if (!torch::cuda::is_available()) {
        return make_error("CUDA not available");
    }
    cudaDeviceProp prop;
    auto err = cudaGetDeviceProperties(&prop, device_index);
    if (err != cudaSuccess) {
        return make_error(cudaGetErrorString(err));
    }
    *major = prop.major;
    *minor = prop.minor;
    return nullptr;
#else
    (void)device_index; (void)major; (void)minor;
    return make_error("CUDA not available (built without cuda feature)");
#endif
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}
