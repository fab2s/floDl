// helpers.h — internal helpers shared by the split shim translation units.
//
// All helpers are `static inline` so each TU gets its own copy; the compiler
// typically inlines them away. This keeps the split .cpp files self-contained
// without requiring a separate object file for a handful of tiny utilities.
//
// Do NOT include this from Rust — it is C++ only. The C FFI surface is in
// shim.h.

#pragma once

#include "shim.h"
#include <torch/torch.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Exception firewall for extern "C" functions with no error-return
// channel: a C++ exception unwinding through the C boundary into Rust is
// undefined behavior, so contain it and fail loudly instead. Only
// reachable when an invariant is already broken (bad handle, corrupted
// CUDA context, allocation failure) — a defined, named abort beats UB.
[[noreturn]] static inline void flodl_fatal(const char* fn, const char* what) {
    fprintf(stderr, "flodl: fatal C++ exception in %s: %s\n",
            fn, what ? what : "non-standard C++ exception");
    fflush(stderr);
    abort();
}

// Helper: convert a C++ exception to a malloc'd C string.
static inline char* make_error(const std::string& msg) {
    char* err = (char*)malloc(msg.size() + 1);
    if (err) {
        memcpy(err, msg.c_str(), msg.size() + 1);
    }
    return err;
}

// Helper: convert our dtype constant to torch::ScalarType.
static inline torch::ScalarType to_scalar_type(int dtype) {
    switch (dtype) {
        case FLODL_FLOAT16:  return torch::kFloat16;
        case FLODL_BFLOAT16: return torch::kBFloat16;
        case FLODL_FLOAT32:  return torch::kFloat32;
        case FLODL_FLOAT64:  return torch::kFloat64;
        case FLODL_INT32:    return torch::kInt32;
        case FLODL_INT64:    return torch::kInt64;
        default:           return torch::kFloat32;
    }
}

// Helper: convert our dtype constant back from torch::ScalarType.
static inline int from_scalar_type(torch::ScalarType st) {
    switch (st) {
        case torch::kFloat16:  return FLODL_FLOAT16;
        case torch::kBFloat16: return FLODL_BFLOAT16;
        case torch::kFloat32:  return FLODL_FLOAT32;
        case torch::kFloat64:  return FLODL_FLOAT64;
        case torch::kInt32:    return FLODL_INT32;
        case torch::kInt64:    return FLODL_INT64;
        default:               return FLODL_FLOAT32;
    }
}

// Helper: convert our device (type, index) to torch::Device.
static inline torch::Device to_device(int device_type, int device_index) {
    if (device_type == FLODL_CUDA) {
        return torch::Device(torch::kCUDA, (c10::DeviceIndex)device_index);
    }
    return torch::Device(torch::kCPU);
}

// Helper: convert torch::Device back to our (type, index) pair.
static inline int from_device_type(const torch::Device& dev) {
    if (dev.is_cuda()) return FLODL_CUDA;
    return FLODL_CPU;
}

static inline int from_device_index(const torch::Device& dev) {
    if (dev.is_cuda()) return (int)dev.index();
    return 0;
}

// Helper: wrap a new torch::Tensor into a heap-allocated pointer.
// The caller (Rust) owns this pointer and must call flodl_free_tensor.
static inline FlodlTensor wrap(torch::Tensor t) {
    auto* p = new torch::Tensor(std::move(t));
    return (FlodlTensor)p;
}

// Helper: unwrap an FlodlTensor handle back to a reference.
static inline torch::Tensor& unwrap(FlodlTensor t) {
    return *((torch::Tensor*)t);
}

// Helper: convert FlodlTensor* array to std::vector<at::Tensor>.
// The returned tensors share storage with the originals — in-place ops
// on the vector elements modify the original data.
static inline std::vector<at::Tensor> unwrap_list(FlodlTensor* tensors, int count) {
    std::vector<at::Tensor> result;
    result.reserve(count);
    for (int i = 0; i < count; i++) {
        result.push_back(unwrap(tensors[i]));
    }
    return result;
}

// Helper: build IntArrayRef from C array.
static inline torch::IntArrayRef make_shape(int64_t* shape, int ndim) {
    return torch::IntArrayRef(shape, ndim);
}

// Helper: dtype to use for comparison masks.
// Float inputs: mask uses input's dtype. Non-float (Int64, Bool): mask is Float32.
static inline torch::ScalarType mask_dtype(const torch::Tensor& t) {
    return t.is_floating_point() ? t.scalar_type() : torch::kFloat32;
}
