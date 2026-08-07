// gpu_compat.h — the single place where the shim's CUDA spelling is
// reconciled with the GPU vendor actually being built for.
//
// WHY THIS EXISTS
//
// libtorch is built for exactly ONE GPU backend, and both backends claim
// c10::DeviceType::CUDA: ROCm masquerades as CUDA all the way up to the
// dispatcher, which is why user code keeps saying .cuda() on an AMD box.
// The vendor is therefore a BUILD-TIME property, fixed at one per process
// — libtorch_cuda.so and libtorch_hip.so cannot be loaded together, as
// they would both register kernels against the same dispatch key.
//
// So the rest of the shim keeps CUDA spelling unconditionally and this
// header does all the reconciling. The mapping falls into three tiers,
// and the split is worth knowing before touching any of it:
//
//   1. IDENTICAL — no aliases needed, only the include path differs.
//      torch::cuda::{is_available,device_count,manual_seed_all} live in
//      libtorch_cpu.so and dispatch through the CUDA hooks that the HIP
//      build registers. ATen/hip/HIPGraph.h and ATen/hip/HIPEvent.h
//      declare at::cuda::{CUDAGraph,CUDAEvent,graph_pool_handle,
//      MempoolId_t} under those exact names, with hip types in their
//      signatures. That is deliberate upstream, not an accident.
//
//   2. MECHANICAL — c10::cuda::* -> c10::hip::*, and the raw runtime
//      cudaXxx -> hipXxx. Pure renames, 1:1.
//
//   3. SEMANTIC, and the one item to actually understand: STREAMS.
//      In a HIP build a tensor's device type is kCUDA while the native
//      stream object lives on DeviceType::HIP. c10::hip::HIPStream is
//      therefore NOT interchangeable with what a tensor hands you:
//      passing one where a CUDA-typed stream is expected is a device-type
//      mismatch, not a naming inconvenience. HIPStreamMasqueradingAsCUDA
//      holds a HIPStream but reports device_type() == CUDA, coercing
//      unsafely in both directions. Every stream alias below goes through
//      it, and ATen/hip/HIPEvent.h is itself written against that same
//      type, so events, streams and record_stream all line up.
//
// NOT handled here: the NCCL/RCCL header split, which stays at its own
// include site next to the collectives that need it.

#pragma once

#ifdef FLODL_BUILD_GPU

// The torch library that registers the GPU backend's ATen kernels. It has
// to be force-loaded at process start; see the long note at the dlopen
// site in ops_nn.cpp for why `--as-needed` makes that load-bearing.
#ifdef __HIP_PLATFORM_AMD__
#define FLODL_TORCH_GPU_LIB "libtorch_hip.so"
#else
#define FLODL_TORCH_GPU_LIB "libtorch_cuda.so"
#endif

#ifdef __HIP_PLATFORM_AMD__

// ---------------------------------------------------------------- ROCm

#include <hip/hip_runtime.h>

#include <ATen/hip/HIPGraph.h>
#include <ATen/hip/HIPEvent.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/hip/HIPFunctions.h>
#include <c10/hip/HIPCachingAllocator.h>

// Tier 2: c10::cuda::* -> c10::hip::*.
//
// The HIP build ships c10/cuda/*.h as dead weight (they are the unbuilt
// CUDA headers, and their generated cuda_cmake_macros.h is absent), and
// there is no libc10_cuda.so to link against — libc10_hip.so exports
// c10::hip::* only. So c10::cuda does not exist here and we define it.
namespace c10 {
namespace cuda {

using ::c10::hip::current_device;
using ::c10::hip::device_count;
using ::c10::hip::set_device;

namespace CUDACachingAllocator = ::c10::hip::HIPCachingAllocator;

} // namespace cuda
} // namespace c10

// Tier 3: streams, via the masquerading wrapper. See the note above —
// the wrapper is what keeps a HIP stream usable wherever the dispatcher
// has already decided the device is kCUDA.
//
// at::cuda already exists at this point (HIPGraph.h and HIPEvent.h
// declare into it); these names are additions to it, not a redefinition.
namespace at {
namespace cuda {

using CUDAStream = ::c10::hip::HIPStreamMasqueradingAsCUDA;

inline CUDAStream getCurrentCUDAStream(::c10::DeviceIndex device_index = -1) {
    return ::c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(device_index);
}

inline CUDAStream getDefaultCUDAStream(::c10::DeviceIndex device_index = -1) {
    return ::c10::hip::getDefaultHIPStreamMasqueradingAsCUDA(device_index);
}

inline void setCurrentCUDAStream(CUDAStream stream) {
    ::c10::hip::setCurrentHIPStreamMasqueradingAsCUDA(stream);
}

// Both upstream overloads are mirrored, so a later caller reaching for
// the int-priority form does not rediscover this file the hard way.
inline CUDAStream getStreamFromPool(const bool isHighPriority = false,
                                    ::c10::DeviceIndex device = -1) {
    return ::c10::hip::getStreamFromPoolMasqueradingAsCUDA(isHighPriority, device);
}

inline CUDAStream getStreamFromPool(const int priority,
                                    ::c10::DeviceIndex device = -1) {
    return ::c10::hip::getStreamFromPoolMasqueradingAsCUDA(priority, device);
}

} // namespace cuda
} // namespace at

// Tier 2 continued: the raw runtime API. Every name below was verified
// present 1:1 in ROCm 7.0's hip/hip_runtime_api.h. Function aliases are
// pointers rather than wrappers so the signature is exact by
// construction and cannot drift from the hip declaration.
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaGraph_t = hipGraph_t;
using cudaStreamCaptureStatus = hipStreamCaptureStatus;
using cudaStreamCaptureMode = hipStreamCaptureMode;

constexpr auto cudaSuccess = hipSuccess;
constexpr auto cudaStreamCaptureStatusActive = hipStreamCaptureStatusActive;
constexpr auto cudaEventDisableTiming = hipEventDisableTiming;
constexpr auto cudaEventDefault = hipEventDefault;

constexpr auto cudaDeviceSynchronize = hipDeviceSynchronize;
constexpr auto cudaGetErrorString = hipGetErrorString;
constexpr auto cudaGraphDestroy = hipGraphDestroy;
constexpr auto cudaMemGetInfo = hipMemGetInfo;
constexpr auto cudaSetDevice = hipSetDevice;
constexpr auto cudaStreamEndCapture = hipStreamEndCapture;
constexpr auto cudaStreamIsCapturing = hipStreamIsCapturing;

// ROCm 6+ versions this symbol: hipGetDeviceProperties is a macro onto
// hipGetDevicePropertiesR0600, and hipDeviceProp_t onto the matching
// struct, so the two stay consistent through the alias above.
constexpr auto cudaGetDeviceProperties = hipGetDeviceProperties;

#else

// ---------------------------------------------------------------- CUDA

#include <cuda_runtime.h>

#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDACachingAllocator.h>

#endif // __HIP_PLATFORM_AMD__

#endif // FLODL_BUILD_GPU
