// ops_cuda.cpp — CUDA Graphs, Events, Streams, NCCL.
//
// All CUDA-dependent functionality. Preserves the original
// #ifdef FLODL_BUILD_CUDA / #else (CPU stubs) / #endif structure
// so the file is build-feature-aware in a single translation unit.
//
// Also contains the NCCL collective and per-rank APIs used by DDP,
// and the final flodl_free_string() utility.

#include "helpers.h"

// --- CUDA Graphs ---

#ifdef FLODL_BUILD_CUDA
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>

// Wrapper that owns a CUDAGraph + the side stream used for capture.
// CUDA graphs must be captured on a non-default stream.
struct FlodlCudaGraph {
    at::cuda::CUDAGraph graph;
    c10::optional<at::cuda::CUDAStream> capture_stream;
    bool capturing = false;

    ~FlodlCudaGraph() {
        // If destroyed while still capturing (e.g. panic in Rust),
        // end the capture to avoid leaving the stream in a bad state.
        if (capturing && capture_stream.has_value()) {
            cudaStreamCaptureStatus status;
            cudaStreamIsCapturing(capture_stream.value().stream(), &status);
            if (status == cudaStreamCaptureStatusActive) {
                cudaGraph_t dummy = nullptr;
                cudaStreamEndCapture(capture_stream.value().stream(), &dummy);
                if (dummy) cudaGraphDestroy(dummy);
            }
            // Restore default stream on this thread.
            at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());
        }
    }
};

extern "C" char* flodl_cuda_graph_new(void** graph_out) {
    try {
        auto* g = new FlodlCudaGraph();
        *graph_out = static_cast<void*>(g);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_cuda_graph_capture_begin(void* graph, uint64_t pool_hi,
                                                  uint64_t pool_lo, int mode) {
    auto* g = static_cast<FlodlCudaGraph*>(graph);
    try {
        at::cuda::MempoolId_t pool = {pool_hi, pool_lo};
        auto capture_mode = static_cast<cudaStreamCaptureMode>(mode);

        // Create a side stream for capture (CUDA graphs need non-default stream).
        auto stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
        g->capture_stream = stream;

        // Wait for any pending work on the default stream.
        at::cuda::CUDAEvent event;
        event.record(at::cuda::getCurrentCUDAStream());
        event.block(stream);

        // Switch to the capture stream.
        at::cuda::setCurrentCUDAStream(stream);

        g->capturing = true;
        g->graph.capture_begin(pool, capture_mode);
        return nullptr;
    } catch (const std::exception& e) {
        // If capture_begin fails, restore default stream.
        at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());
        g->capturing = false;
        return make_error(e.what());
    }
}

extern "C" char* flodl_cuda_graph_capture_end(void* graph) {
    try {
        auto* g = static_cast<FlodlCudaGraph*>(graph);
        g->graph.capture_end();
        g->capturing = false;

        // Restore the default stream.
        auto default_stream = at::cuda::getDefaultCUDAStream();
        if (g->capture_stream.has_value()) {
            // Wait for capture stream to finish before handing back to default.
            at::cuda::CUDAEvent event;
            event.record(g->capture_stream.value());
            event.block(default_stream);
        }
        at::cuda::setCurrentCUDAStream(default_stream);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_cuda_graph_replay(void* graph) {
    try {
        auto* g = static_cast<FlodlCudaGraph*>(graph);
        g->graph.replay();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_cuda_graph_reset(void* graph) {
    try {
        auto* g = static_cast<FlodlCudaGraph*>(graph);
        g->graph.reset();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" void flodl_cuda_graph_delete(void* graph) {
    delete static_cast<FlodlCudaGraph*>(graph);
}

extern "C" void flodl_cuda_graph_pool(void* graph, uint64_t* pool_hi, uint64_t* pool_lo) {
    auto* g = static_cast<FlodlCudaGraph*>(graph);
    auto pool = g->graph.pool();
    *pool_hi = pool.first;
    *pool_lo = pool.second;
}

extern "C" void flodl_cuda_graph_pool_handle(uint64_t* pool_hi, uint64_t* pool_lo) {
    auto pool = at::cuda::graph_pool_handle();
    *pool_hi = pool.first;
    *pool_lo = pool.second;
}

// --- CUDA Events ---

extern "C" char* flodl_cuda_event_new(int flags, void** event_out) {
    try {
        unsigned int cuda_flags = (flags == 1)
            ? cudaEventDisableTiming
            : cudaEventDefault;
        auto* event = new at::cuda::CUDAEvent(cuda_flags);
        *event_out = static_cast<void*>(event);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_cuda_event_record(void* event) {
    try {
        static_cast<at::cuda::CUDAEvent*>(event)->record();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_cuda_event_record_on_stream(void* event, void* stream) {
    try {
        auto* e = static_cast<at::cuda::CUDAEvent*>(event);
        auto* s = static_cast<at::cuda::CUDAStream*>(stream);
        e->record(*s);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_cuda_event_synchronize(void* event) {
    try {
        static_cast<at::cuda::CUDAEvent*>(event)->synchronize();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_cuda_event_elapsed_time(void* start, void* end,
                                                 float* ms_out) {
    try {
        *ms_out = static_cast<at::cuda::CUDAEvent*>(start)->elapsed_time(
            *static_cast<at::cuda::CUDAEvent*>(end));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" int flodl_cuda_event_query(void* event) {
    return static_cast<at::cuda::CUDAEvent*>(event)->query() ? 1 : 0;
}

extern "C" void flodl_cuda_event_delete(void* event) {
    delete static_cast<at::cuda::CUDAEvent*>(event);
}

// --- CUDA Streams ---

extern "C" char* flodl_cuda_stream_new(int device_index, int high_priority,
                                        void** stream_out) {
    try {
        auto stream = at::cuda::getStreamFromPool(
            /*isHighPriority=*/high_priority != 0,
            static_cast<c10::DeviceIndex>(device_index));
        *stream_out = new at::cuda::CUDAStream(stream);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_cuda_stream_synchronize(void* stream) {
    try {
        static_cast<at::cuda::CUDAStream*>(stream)->synchronize();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_cuda_stream_wait_event(void* stream, void* event) {
    try {
        auto* s = static_cast<at::cuda::CUDAStream*>(stream);
        auto* e = static_cast<at::cuda::CUDAEvent*>(event);
        e->block(*s);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" int flodl_cuda_stream_query(void* stream) {
    return static_cast<at::cuda::CUDAStream*>(stream)->query() ? 1 : 0;
}

extern "C" void flodl_cuda_stream_set_current(void* stream) {
    at::cuda::setCurrentCUDAStream(
        *static_cast<at::cuda::CUDAStream*>(stream));
}

extern "C" void* flodl_cuda_stream_get_current(int device_index) {
    auto stream = at::cuda::getCurrentCUDAStream(
        static_cast<c10::DeviceIndex>(device_index));
    auto* heap = new at::cuda::CUDAStream(stream);
    return static_cast<void*>(heap);
}

extern "C" void flodl_cuda_stream_restore_default(int device_index) {
    at::cuda::setCurrentCUDAStream(
        at::cuda::getDefaultCUDAStream(
            static_cast<c10::DeviceIndex>(device_index)));
}

extern "C" void flodl_cuda_stream_delete(void* stream) {
    delete static_cast<at::cuda::CUDAStream*>(stream);
}

// --- NCCL Collective Operations ---

#include <nccl.h>
#include <atomic>

static ncclDataType_t to_nccl_dtype(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:    return ncclFloat32;
        case at::kDouble:   return ncclFloat64;
        case at::kHalf:     return ncclFloat16;
        case at::kBFloat16: return ncclBfloat16;
        case at::kInt:      return ncclInt32;
        case at::kLong:     return ncclInt64;
        case at::kByte:     return ncclUint8;
        case at::kChar:     return ncclInt8;
        default:
            throw std::runtime_error(
                std::string("Unsupported dtype for NCCL: ") +
                toString(dtype));
    }
}

struct FlodlNcclComms {
    std::vector<ncclComm_t> comms;
    std::vector<int> devlist;
    int ndev;

    ~FlodlNcclComms() {
        for (int i = 0; i < ndev; i++) {
            if (comms[i]) {
                ncclCommDestroy(comms[i]);
            }
        }
    }
};

extern "C" char* flodl_nccl_init(int ndev, const int* devlist,
                                   void** handle_out) {
    try {
        auto* h = new FlodlNcclComms();
        h->ndev = ndev;
        h->devlist.assign(devlist, devlist + ndev);
        h->comms.resize(ndev);
        ncclResult_t result = ncclCommInitAll(h->comms.data(), ndev, devlist);
        if (result != ncclSuccess) {
            std::string msg = std::string("ncclCommInitAll failed: ") +
                              ncclGetErrorString(result);
            delete h;
            return make_error(msg);
        }
        *handle_out = static_cast<void*>(h);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" void flodl_nccl_destroy(void* handle) {
    delete static_cast<FlodlNcclComms*>(handle);
}

extern "C" char* flodl_nccl_all_reduce(void* handle, FlodlTensor* tensors,
                                         void** streams, int op) {
    auto* h = static_cast<FlodlNcclComms*>(handle);
    try {
        ncclGroupStart();
        for (int i = 0; i < h->ndev; i++) {
            cudaSetDevice(h->devlist[i]);
            auto& t = *reinterpret_cast<torch::Tensor*>(tensors[i]);
            void* data = t.data_ptr();
            size_t count = static_cast<size_t>(t.numel());
            ncclDataType_t dtype = to_nccl_dtype(t.scalar_type());
            auto nccl_op = static_cast<ncclRedOp_t>(op);

            cudaStream_t cuda_stream;
            if (streams && streams[i]) {
                cuda_stream = static_cast<at::cuda::CUDAStream*>(streams[i])
                    ->stream();
            } else {
                cuda_stream = at::cuda::getDefaultCUDAStream(
                    static_cast<c10::DeviceIndex>(h->devlist[i])).stream();
            }

            ncclResult_t result = ncclAllReduce(
                data, data, count, dtype, nccl_op,
                h->comms[i], cuda_stream);
            if (result != ncclSuccess) {
                ncclGroupEnd();
                return make_error(
                    std::string("ncclAllReduce failed on device ") +
                    std::to_string(h->devlist[i]) + ": " +
                    ncclGetErrorString(result));
            }
        }
        ncclResult_t result = ncclGroupEnd();
        if (result != ncclSuccess) {
            return make_error(
                std::string("ncclGroupEnd failed: ") +
                ncclGetErrorString(result));
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_nccl_broadcast(void* handle, FlodlTensor* tensors,
                                        void** streams, int root) {
    auto* h = static_cast<FlodlNcclComms*>(handle);
    try {
        ncclGroupStart();
        for (int i = 0; i < h->ndev; i++) {
            cudaSetDevice(h->devlist[i]);
            auto& t = *reinterpret_cast<torch::Tensor*>(tensors[i]);
            void* data = t.data_ptr();
            size_t count = static_cast<size_t>(t.numel());
            ncclDataType_t dtype = to_nccl_dtype(t.scalar_type());

            cudaStream_t cuda_stream;
            if (streams && streams[i]) {
                cuda_stream = static_cast<at::cuda::CUDAStream*>(streams[i])
                    ->stream();
            } else {
                cuda_stream = at::cuda::getDefaultCUDAStream(
                    static_cast<c10::DeviceIndex>(h->devlist[i])).stream();
            }

            ncclResult_t result = ncclBroadcast(
                data, data, count, dtype, root,
                h->comms[i], cuda_stream);
            if (result != ncclSuccess) {
                ncclGroupEnd();
                return make_error(
                    std::string("ncclBroadcast failed on device ") +
                    std::to_string(h->devlist[i]) + ": " +
                    ncclGetErrorString(result));
            }
        }
        ncclResult_t result = ncclGroupEnd();
        if (result != ncclSuccess) {
            return make_error(
                std::string("ncclGroupEnd failed: ") +
                ncclGetErrorString(result));
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" int flodl_nccl_size(void* handle) {
    return static_cast<FlodlNcclComms*>(handle)->ndev;
}

// --- NCCL Per-Rank Operations (for multi-threaded DDP) ---

struct FlodlNcclRankComm {
    ncclComm_t comm;
    std::atomic<bool> aborted{false};

    ~FlodlNcclRankComm() {
        // ncclCommAbort already frees the comm; skip destroy if aborted.
        if (comm && !aborted.load(std::memory_order_acquire)) {
            ncclCommDestroy(comm);
        }
    }
};

extern "C" char* flodl_nccl_get_unique_id(void* uid_out) {
    try {
        ncclUniqueId id;
        ncclResult_t result = ncclGetUniqueId(&id);
        if (result != ncclSuccess) {
            return make_error(
                std::string("ncclGetUniqueId failed: ") +
                ncclGetErrorString(result));
        }
        static_assert(sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES,
                      "ncclUniqueId size mismatch");
        memcpy(uid_out, &id, sizeof(ncclUniqueId));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_nccl_init_rank(int rank, int nranks, const void* uid,
                                        void** handle_out) {
    try {
        ncclUniqueId id;
        memcpy(&id, uid, sizeof(ncclUniqueId));

        auto* h = new FlodlNcclRankComm();
        h->comm = nullptr;
        ncclResult_t result = ncclCommInitRank(&h->comm, nranks, id, rank);
        if (result != ncclSuccess) {
            std::string msg = std::string("ncclCommInitRank failed (rank ") +
                              std::to_string(rank) + "): " +
                              ncclGetErrorString(result);
            delete h;
            return make_error(msg);
        }
        *handle_out = static_cast<void*>(h);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" void flodl_nccl_destroy_rank(void* handle) {
    delete static_cast<FlodlNcclRankComm*>(handle);
}

extern "C" char* flodl_nccl_abort_rank(void* handle) {
    auto* h = static_cast<FlodlNcclRankComm*>(handle);
    if (!h || !h->comm) return nullptr;
    // Idempotent: only abort once.
    bool expected = false;
    if (!h->aborted.compare_exchange_strong(expected, true,
            std::memory_order_acq_rel)) {
        return nullptr; // already aborted
    }
    ncclResult_t result = ncclCommAbort(h->comm);
    h->comm = nullptr; // prevent double-free in destructor
    if (result != ncclSuccess) {
        return make_error(std::string("ncclCommAbort failed: ") +
                          ncclGetErrorString(result));
    }
    return nullptr;
}

extern "C" char* flodl_nccl_all_reduce_rank(void* handle, FlodlTensor* tensors,
                                              int ntensors, void* stream,
                                              int op) {
    auto* h = static_cast<FlodlNcclRankComm*>(handle);
    try {
        auto nccl_op = static_cast<ncclRedOp_t>(op);
        cudaStream_t cuda_stream;
        if (stream) {
            cuda_stream = static_cast<at::cuda::CUDAStream*>(stream)->stream();
        } else {
            cuda_stream = at::cuda::getCurrentCUDAStream().stream();
        }

        ncclGroupStart();
        for (int i = 0; i < ntensors; i++) {
            auto& t = *reinterpret_cast<torch::Tensor*>(tensors[i]);
            void* data = t.data_ptr();
            size_t count = static_cast<size_t>(t.numel());
            ncclDataType_t dtype = to_nccl_dtype(t.scalar_type());

            ncclResult_t result = ncclAllReduce(
                data, data, count, dtype, nccl_op,
                h->comm, cuda_stream);
            if (result != ncclSuccess) {
                ncclGroupEnd();
                return make_error(
                    std::string("ncclAllReduce (rank) failed on tensor ") +
                    std::to_string(i) + ": " +
                    ncclGetErrorString(result));
            }
        }
        ncclResult_t result = ncclGroupEnd();
        if (result != ncclSuccess) {
            return make_error(
                std::string("ncclGroupEnd (rank) failed: ") +
                ncclGetErrorString(result));
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_nccl_broadcast_rank(void* handle, FlodlTensor* tensors,
                                             int ntensors, void* stream,
                                             int root) {
    auto* h = static_cast<FlodlNcclRankComm*>(handle);
    try {
        cudaStream_t cuda_stream;
        if (stream) {
            cuda_stream = static_cast<at::cuda::CUDAStream*>(stream)->stream();
        } else {
            cuda_stream = at::cuda::getCurrentCUDAStream().stream();
        }

        ncclGroupStart();
        for (int i = 0; i < ntensors; i++) {
            auto& t = *reinterpret_cast<torch::Tensor*>(tensors[i]);
            void* data = t.data_ptr();
            size_t count = static_cast<size_t>(t.numel());
            ncclDataType_t dtype = to_nccl_dtype(t.scalar_type());

            ncclResult_t result = ncclBroadcast(
                data, data, count, dtype, root,
                h->comm, cuda_stream);
            if (result != ncclSuccess) {
                ncclGroupEnd();
                return make_error(
                    std::string("ncclBroadcast (rank) failed on tensor ") +
                    std::to_string(i) + ": " +
                    ncclGetErrorString(result));
            }
        }
        ncclResult_t result = ncclGroupEnd();
        if (result != ncclSuccess) {
            return make_error(
                std::string("ncclGroupEnd (rank broadcast) failed: ") +
                ncclGetErrorString(result));
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_nccl_split_rank(void* group_handle, int rank,
                                         void** rank_handle_out) {
    auto* g = static_cast<FlodlNcclComms*>(group_handle);
    try {
        if (rank < 0 || rank >= g->ndev) {
            return make_error(
                std::string("flodl_nccl_split_rank: rank ") +
                std::to_string(rank) + " out of range (ndev=" +
                std::to_string(g->ndev) + ")");
        }
        if (!g->comms[rank]) {
            return make_error(
                std::string("flodl_nccl_split_rank: rank ") +
                std::to_string(rank) + " already extracted");
        }
        auto* h = new FlodlNcclRankComm();
        h->comm = g->comms[rank];
        g->comms[rank] = nullptr;  // transfer ownership
        *rank_handle_out = static_cast<void*>(h);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

#else // CPU-only stubs

extern "C" char* flodl_cuda_graph_new(void** graph_out) {
    (void)graph_out;
    return make_error("CUDA Graphs require a CUDA build");
}

extern "C" char* flodl_cuda_graph_capture_begin(void* graph, uint64_t pool_hi,
                                                  uint64_t pool_lo, int mode) {
    (void)graph; (void)pool_hi; (void)pool_lo; (void)mode;
    return make_error("CUDA Graphs require a CUDA build");
}

extern "C" char* flodl_cuda_graph_capture_end(void* graph) {
    (void)graph;
    return make_error("CUDA Graphs require a CUDA build");
}

extern "C" char* flodl_cuda_graph_replay(void* graph) {
    (void)graph;
    return make_error("CUDA Graphs require a CUDA build");
}

extern "C" char* flodl_cuda_graph_reset(void* graph) {
    (void)graph;
    return make_error("CUDA Graphs require a CUDA build");
}

extern "C" void flodl_cuda_graph_delete(void* graph) { (void)graph; }

extern "C" void flodl_cuda_graph_pool(void* graph, uint64_t* pool_hi, uint64_t* pool_lo) {
    (void)graph; *pool_hi = 0; *pool_lo = 0;
}

extern "C" void flodl_cuda_graph_pool_handle(uint64_t* pool_hi, uint64_t* pool_lo) {
    *pool_hi = 0; *pool_lo = 0;
}

// --- CUDA Events (CPU stubs) ---

extern "C" char* flodl_cuda_event_new(int flags, void** event_out) {
    (void)flags; (void)event_out;
    return make_error("CUDA Events require a CUDA build");
}
extern "C" char* flodl_cuda_event_record(void* event) {
    (void)event;
    return make_error("CUDA Events require a CUDA build");
}
extern "C" char* flodl_cuda_event_record_on_stream(void* event, void* stream) {
    (void)event; (void)stream;
    return make_error("CUDA Events require a CUDA build");
}
extern "C" char* flodl_cuda_event_synchronize(void* event) {
    (void)event;
    return make_error("CUDA Events require a CUDA build");
}
extern "C" char* flodl_cuda_event_elapsed_time(void* start, void* end,
                                                 float* ms_out) {
    (void)start; (void)end; (void)ms_out;
    return make_error("CUDA Events require a CUDA build");
}
extern "C" int flodl_cuda_event_query(void* event) {
    (void)event; return 1;
}
extern "C" void flodl_cuda_event_delete(void* event) { (void)event; }

// --- CUDA Streams (CPU stubs) ---

extern "C" char* flodl_cuda_stream_new(int device_index, int high_priority,
                                        void** stream_out) {
    (void)device_index; (void)high_priority; (void)stream_out;
    return make_error("CUDA Streams require a CUDA build");
}
extern "C" char* flodl_cuda_stream_synchronize(void* stream) {
    (void)stream;
    return make_error("CUDA Streams require a CUDA build");
}
extern "C" char* flodl_cuda_stream_wait_event(void* stream, void* event) {
    (void)stream; (void)event;
    return make_error("CUDA Streams require a CUDA build");
}
extern "C" int flodl_cuda_stream_query(void* stream) {
    (void)stream; return 1;
}
extern "C" void flodl_cuda_stream_set_current(void* stream) { (void)stream; }
extern "C" void* flodl_cuda_stream_get_current(int device_index) {
    (void)device_index; return nullptr;
}
extern "C" void flodl_cuda_stream_restore_default(int device_index) {
    (void)device_index;
}
extern "C" void flodl_cuda_stream_delete(void* stream) { (void)stream; }

// --- NCCL (CPU stubs) ---

extern "C" char* flodl_nccl_init(int ndev, const int* devlist,
                                   void** handle_out) {
    (void)ndev; (void)devlist; (void)handle_out;
    return make_error("NCCL requires a CUDA build");
}
extern "C" void flodl_nccl_destroy(void* handle) { (void)handle; }
extern "C" char* flodl_nccl_all_reduce(void* handle, FlodlTensor* tensors,
                                         void** streams, int op) {
    (void)handle; (void)tensors; (void)streams; (void)op;
    return make_error("NCCL requires a CUDA build");
}
extern "C" char* flodl_nccl_broadcast(void* handle, FlodlTensor* tensors,
                                        void** streams, int root) {
    (void)handle; (void)tensors; (void)streams; (void)root;
    return make_error("NCCL requires a CUDA build");
}
extern "C" int flodl_nccl_size(void* handle) { (void)handle; return 0; }

// --- NCCL Per-Rank (CPU stubs) ---

extern "C" char* flodl_nccl_get_unique_id(void* uid_out) {
    (void)uid_out;
    return make_error("NCCL requires a CUDA build");
}
extern "C" char* flodl_nccl_init_rank(int rank, int nranks, const void* uid,
                                        void** handle_out) {
    (void)rank; (void)nranks; (void)uid; (void)handle_out;
    return make_error("NCCL requires a CUDA build");
}
extern "C" void flodl_nccl_destroy_rank(void* handle) { (void)handle; }
extern "C" char* flodl_nccl_abort_rank(void* handle) {
    (void)handle;
    return nullptr; // no-op on CPU
}
extern "C" char* flodl_nccl_all_reduce_rank(void* handle, FlodlTensor* tensors,
                                              int ntensors, void* stream,
                                              int op) {
    (void)handle; (void)tensors; (void)ntensors; (void)stream; (void)op;
    return make_error("NCCL requires a CUDA build");
}
extern "C" char* flodl_nccl_broadcast_rank(void* handle, FlodlTensor* tensors,
                                             int ntensors, void* stream,
                                             int root) {
    (void)handle; (void)tensors; (void)ntensors; (void)stream; (void)root;
    return make_error("NCCL requires a CUDA build");
}
extern "C" char* flodl_nccl_split_rank(void* group_handle, int rank,
                                         void** rank_handle_out) {
    (void)group_handle; (void)rank; (void)rank_handle_out;
    return make_error("NCCL requires a CUDA build");
}

#endif // FLODL_BUILD_CUDA

// --- Utility ---

extern "C" void flodl_free_string(char* s) {
    free(s);
}
