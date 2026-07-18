// ops_training.cpp — training-time ops.
//
// Covers: autograd (requires_grad, backward, grad, detach, no_grad guard,
// autograd_node_count), autocast (AMP), in-place arithmetic, linear,
// RNN (GRU/LSTM cell + sequence + cached-params variants), misc
// (cudnn_benchmark, manual_seed, meshgrid, cdist, cosine_similarity),
// optimizer kernels (adam_step, fused_adam/adamw, batched),
// pin_memory / channels_last / malloc_trim, fused clip_grad_norm,
// foreach multi-tensor ops, losses (mse/ce/bce/kl/nll/ctc),
// batch_norm, dropout, embedding_bag, copy_.

#include "helpers.h"
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <queue>
#include <unordered_set>
#ifdef __linux__
#include <malloc.h>
#endif

// --- Autograd ---

extern "C" char* flodl_set_requires_grad(FlodlTensor t, int requires_grad,
                                          FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).set_requires_grad(requires_grad != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" int flodl_requires_grad(FlodlTensor t) {
    try {
    return unwrap(t).requires_grad() ? 1 : 0;
    } catch (const std::exception& e) {
        flodl_fatal("flodl_requires_grad", e.what());
    } catch (...) {
        flodl_fatal("flodl_requires_grad", nullptr);
    }
}

// Force creation of the AccumulateGrad node for a leaf tensor with
// requires_grad=true and return a handle that keeps the node alive.
//
// Normally the node is created lazily on first backward() — inside the
// autograd engine's worker thread, whose current stream is the device
// default. The node's input_metadata captures getCurrentCUDAStream(device)
// at construction time, so the node is pinned to the default stream and
// libtorch fires the "AccumulateGrad node's stream does not match"
// warning on every run that uses a non-default training stream.
//
// The tensor's AutogradMeta stores a weak_ptr to the AccumulateGrad.
// To keep the node alive across backward calls we must hold a strong
// shared_ptr somewhere — returning an opaque handle lets the caller
// own the lifetime.
//
// Usage:
//   1. Under a StreamGuard on the training stream, call this for each
//      leaf parameter.
//   2. Store the returned handle(s) for the lifetime of the worker.
//   3. Call flodl_grad_accumulator_delete to free the handle (and the
//      AccumulateGrad node, if no backward is in flight).
//
// No-op (returns nullptr handle, nullptr error) for non-leaf or
// non-requires-grad tensors.
extern "C" char* flodl_ensure_grad_accumulator(FlodlTensor t, void** handle_out) {
    try {
        *handle_out = nullptr;
        auto& tensor = unwrap(t);
        if (tensor.is_leaf() && tensor.requires_grad()) {
            auto acc = torch::autograd::impl::grad_accumulator(tensor);
            // Heap-allocate the shared_ptr so the strong ref survives
            // after this function returns. Rust owns the deletion.
            auto* boxed = new std::shared_ptr<torch::autograd::Node>(std::move(acc));
            *handle_out = static_cast<void*>(boxed);
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// Delete a handle returned by flodl_ensure_grad_accumulator.
// Safe to call with nullptr (no-op).
extern "C" void flodl_grad_accumulator_delete(void* handle) {
    try {
    if (handle != nullptr) {
        delete static_cast<std::shared_ptr<torch::autograd::Node>*>(handle);
    }
    } catch (const std::exception& e) {
        flodl_fatal("flodl_grad_accumulator_delete", e.what());
    } catch (...) {
        flodl_fatal("flodl_grad_accumulator_delete", nullptr);
    }
}

extern "C" char* flodl_backward(FlodlTensor t) {
    try {
        unwrap(t).backward();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_grad(FlodlTensor t, FlodlTensor* result) {
    try {
        auto g = unwrap(t).grad();
        if (g.defined()) {
            *result = wrap(g);
        } else {
            *result = nullptr;
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_set_grad(FlodlTensor t, FlodlTensor grad) {
    try {
        unwrap(t).mutable_grad() = unwrap(grad);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_zero_grad(FlodlTensor t) {
    try {
        auto& tensor = unwrap(t);
        if (tensor.grad().defined()) {
            tensor.mutable_grad().zero_();
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" void flodl_zero_grad_set_to_none(FlodlTensor t) {
    try {
    auto& tensor = unwrap(t);
    if (tensor.grad().defined()) {
        tensor.mutable_grad().reset();
    }
    } catch (const std::exception& e) {
        flodl_fatal("flodl_zero_grad_set_to_none", e.what());
    } catch (...) {
        flodl_fatal("flodl_zero_grad_set_to_none", nullptr);
    }
}

extern "C" char* flodl_detach(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).detach());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_detach_(FlodlTensor t) {
    try {
        unwrap(t).detach_();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" int flodl_is_leaf(FlodlTensor t) {
    try {
    return unwrap(t).is_leaf() ? 1 : 0;
    } catch (const std::exception& e) {
        flodl_fatal("flodl_is_leaf", e.what());
    } catch (...) {
        flodl_fatal("flodl_is_leaf", nullptr);
    }
}

// --- Autograd context ---

extern "C" void* flodl_no_grad_guard_new() {
    try {
    return new torch::NoGradGuard();
    } catch (const std::exception& e) {
        flodl_fatal("flodl_no_grad_guard_new", e.what());
    } catch (...) {
        flodl_fatal("flodl_no_grad_guard_new", nullptr);
    }
}

extern "C" void flodl_no_grad_guard_delete(void* guard) {
    try {
    delete static_cast<torch::NoGradGuard*>(guard);
    } catch (const std::exception& e) {
        flodl_fatal("flodl_no_grad_guard_delete", e.what());
    } catch (...) {
        flodl_fatal("flodl_no_grad_guard_delete", nullptr);
    }
}

extern "C" int flodl_is_grad_enabled() {
    try {
    return torch::GradMode::is_enabled() ? 1 : 0;
    } catch (const std::exception& e) {
        flodl_fatal("flodl_is_grad_enabled", e.what());
    } catch (...) {
        flodl_fatal("flodl_is_grad_enabled", nullptr);
    }
}

// --- Autocast (automatic mixed precision) ---

// RAII struct: saves and restores autocast state on construction/destruction.
struct FlodlAutocastGuard {
    c10::DeviceType device;
    bool was_enabled;
    at::ScalarType old_dtype;

    FlodlAutocastGuard(c10::DeviceType dev, at::ScalarType dtype)
        : device(dev)
        , was_enabled(at::autocast::is_autocast_enabled(dev))
        , old_dtype(at::autocast::get_autocast_dtype(dev))
    {
        at::autocast::set_autocast_enabled(dev, true);
        at::autocast::set_autocast_dtype(dev, dtype);
        at::autocast::increment_nesting();
    }

    ~FlodlAutocastGuard() {
        if (at::autocast::decrement_nesting() == 0) {
            at::autocast::clear_cache();
        }
        at::autocast::set_autocast_dtype(device, old_dtype);
        at::autocast::set_autocast_enabled(device, was_enabled);
    }
};

static c10::DeviceType to_device_type_enum(int device_type) {
    if (device_type == FLODL_CUDA) return c10::DeviceType::CUDA;
    return c10::DeviceType::CPU;
}

extern "C" void* flodl_autocast_guard_new(int device_type, int dtype) {
    try {
    auto dev = to_device_type_enum(device_type);
    auto st = to_scalar_type(dtype);
    return new FlodlAutocastGuard(dev, st);
    } catch (const std::exception& e) {
        flodl_fatal("flodl_autocast_guard_new", e.what());
    } catch (...) {
        flodl_fatal("flodl_autocast_guard_new", nullptr);
    }
}

extern "C" void flodl_autocast_guard_delete(void* guard) {
    try {
    delete static_cast<FlodlAutocastGuard*>(guard);
    } catch (const std::exception& e) {
        flodl_fatal("flodl_autocast_guard_delete", e.what());
    } catch (...) {
        flodl_fatal("flodl_autocast_guard_delete", nullptr);
    }
}

extern "C" int flodl_is_autocast_enabled(int device_type) {
    try {
    auto dev = to_device_type_enum(device_type);
    return at::autocast::is_autocast_enabled(dev) ? 1 : 0;
    } catch (const std::exception& e) {
        flodl_fatal("flodl_is_autocast_enabled", e.what());
    } catch (...) {
        flodl_fatal("flodl_is_autocast_enabled", nullptr);
    }
}

// --- In-place operations ---

extern "C" char* flodl_add_(FlodlTensor t, FlodlTensor other) {
    try {
        unwrap(t).add_(unwrap(other));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_sub_(FlodlTensor t, FlodlTensor other) {
    try {
        unwrap(t).sub_(unwrap(other));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_mul_scalar_(FlodlTensor t, double scalar) {
    try {
        unwrap(t).mul_(scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_add_scalar_(FlodlTensor t, double scalar) {
    try {
        unwrap(t).add_(scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_zero_(FlodlTensor t) {
    try {
        unwrap(t).zero_();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_mul_(FlodlTensor t, FlodlTensor other) {
    try {
        unwrap(t).mul_(unwrap(other));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_div_scalar_(FlodlTensor t, double scalar) {
    try {
        unwrap(t).div_(scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_div_(FlodlTensor t, FlodlTensor other) {
    try {
        unwrap(t).div_(unwrap(other));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_fill_(FlodlTensor t, double value) {
    try {
        unwrap(t).fill_(value);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Fused ops ---

extern "C" char* flodl_linear(FlodlTensor input, FlodlTensor weight,
                              FlodlTensor bias, FlodlTensor* result) {
    try {
        c10::optional<torch::Tensor> b;
        if (bias != nullptr) {
            b = unwrap(bias);
        }
        *result = wrap(torch::linear(unwrap(input), unwrap(weight), b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_gru_cell(FlodlTensor input, FlodlTensor hx,
                                FlodlTensor w_ih, FlodlTensor w_hh,
                                FlodlTensor b_ih, FlodlTensor b_hh,
                                FlodlTensor* result) {
    try {
        *result = wrap(torch::gru_cell(
            unwrap(input), unwrap(hx),
            unwrap(w_ih), unwrap(w_hh),
            unwrap(b_ih), unwrap(b_hh)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_lstm_cell(FlodlTensor input, FlodlTensor hx,
                                 FlodlTensor cx,
                                 FlodlTensor w_ih, FlodlTensor w_hh,
                                 FlodlTensor b_ih, FlodlTensor b_hh,
                                 FlodlTensor* h_out, FlodlTensor* c_out) {
    try {
        auto result = torch::lstm_cell(
            unwrap(input), {unwrap(hx), unwrap(cx)},
            unwrap(w_ih), unwrap(w_hh),
            unwrap(b_ih), unwrap(b_hh));
        *h_out = wrap(std::get<0>(result));
        *c_out = wrap(std::get<1>(result));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Fused sequence ops (cuDNN-accelerated) ---

// Persistent cache for RNN parameter vectors. After creation (with optional
// cuDNN flatten), the std::vector<at::Tensor> lives here — forward calls
// just dereference the pointer, matching PyTorch's single-call pattern.
struct FlodlRnnParamsImpl {
    std::vector<at::Tensor> params;
};

// Flatten RNN params into cuDNN's expected weight layout using
// at::_cudnn_rnn_flatten_weight — the same function PyTorch's
// nn.LSTM/GRU.flatten_parameters() calls internally. Handles
// cuDNN-specific alignment/padding that simple contiguous packing
// misses. Modifies the shared TensorImpl in-place via set_(), so
// the original Parameter tensors in Rust also see the flat layout.
// Persists across training steps (in-place optimizers keep the storage).
// Self-corrects if the layout is broken (checkpoint load, cast, etc.).
// mode: 2 = LSTM, 3 = GRU
static void flatten_rnn_params(std::vector<at::Tensor>& params,
                                int64_t mode, int64_t num_layers,
                                bool batch_first) {
    if (params.empty() || !params[0].is_cuda()) return;

    int64_t weight_stride0 = 4; // w_ih, w_hh, b_ih, b_hh per layer
    int64_t input_size = params[0].size(1);  // w_ih: [gates*hs, input_size]
    int64_t hidden_size = params[1].size(1); // w_hh: [gates*hs, hidden_size]

    at::NoGradGuard no_grad;
    at::_cudnn_rnn_flatten_weight(
        params, weight_stride0,
        input_size, mode, hidden_size,
        /*proj_size=*/0, num_layers,
        batch_first, /*bidirectional=*/false);
}

extern "C" char* flodl_lstm(FlodlTensor input, FlodlTensor h_0, FlodlTensor c_0,
                             const FlodlTensor* params, int64_t num_params,
                             int64_t num_layers, bool batch_first, bool flatten,
                             FlodlTensor* output, FlodlTensor* h_n, FlodlTensor* c_n) {
    try {
        std::vector<at::Tensor> params_vec;
        params_vec.reserve(num_params);
        for (int64_t i = 0; i < num_params; i++) {
            params_vec.push_back(unwrap(params[i]));
        }
        if (flatten) {
            flatten_rnn_params(params_vec, /*LSTM=*/2, num_layers, batch_first);
        }
        auto result = at::lstm(
            unwrap(input), {unwrap(h_0), unwrap(c_0)}, params_vec,
            /*has_biases=*/true, num_layers, /*dropout=*/0.0,
            /*train=*/true, /*bidirectional=*/false, batch_first);
        *output = wrap(std::get<0>(result));
        *h_n = wrap(std::get<1>(result));
        *c_n = wrap(std::get<2>(result));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_gru(FlodlTensor input, FlodlTensor h_0,
                            const FlodlTensor* params, int64_t num_params,
                            int64_t num_layers, bool batch_first, bool flatten,
                            FlodlTensor* output, FlodlTensor* h_n) {
    try {
        std::vector<at::Tensor> params_vec;
        params_vec.reserve(num_params);
        for (int64_t i = 0; i < num_params; i++) {
            params_vec.push_back(unwrap(params[i]));
        }
        if (flatten) {
            flatten_rnn_params(params_vec, /*GRU=*/3, num_layers, batch_first);
        }
        auto result = at::gru(
            unwrap(input), unwrap(h_0), params_vec,
            /*has_biases=*/true, num_layers, /*dropout=*/0.0,
            /*train=*/true, /*bidirectional=*/false, batch_first);
        *output = wrap(std::get<0>(result));
        *h_n = wrap(std::get<1>(result));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Cached RNN params (zero per-forward overhead) ---

extern "C" char* flodl_rnn_params_create(
        const FlodlTensor* params, int64_t num_params,
        int64_t mode, int64_t num_layers, bool batch_first,
        bool flatten, void** out) {
    auto rp = std::make_unique<FlodlRnnParamsImpl>();
    try {
        rp->params.reserve(num_params);
        for (int64_t i = 0; i < num_params; i++) {
            rp->params.push_back(unwrap(params[i]));
        }
        if (flatten) {
            flatten_rnn_params(rp->params, mode, num_layers, batch_first);
        }
        *out = rp.release();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" void flodl_rnn_params_free(void* rp) {
    try {
    delete static_cast<FlodlRnnParamsImpl*>(rp);
    } catch (const std::exception& e) {
        flodl_fatal("flodl_rnn_params_free", e.what());
    } catch (...) {
        flodl_fatal("flodl_rnn_params_free", nullptr);
    }
}

extern "C" char* flodl_lstm_cached(
        FlodlTensor input, FlodlTensor h_0, FlodlTensor c_0,
        void* rp, int64_t num_layers, bool batch_first,
        FlodlTensor* output, FlodlTensor* h_n, FlodlTensor* c_n) {
    try {
        auto& params = static_cast<FlodlRnnParamsImpl*>(rp)->params;
        auto result = at::lstm(
            unwrap(input), {unwrap(h_0), unwrap(c_0)}, params,
            /*has_biases=*/true, num_layers, /*dropout=*/0.0,
            /*train=*/true, /*bidirectional=*/false, batch_first);
        *output = wrap(std::get<0>(result));
        *h_n = wrap(std::get<1>(result));
        *c_n = wrap(std::get<2>(result));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_gru_cached(
        FlodlTensor input, FlodlTensor h_0,
        void* rp, int64_t num_layers, bool batch_first,
        FlodlTensor* output, FlodlTensor* h_n) {
    try {
        auto& params = static_cast<FlodlRnnParamsImpl*>(rp)->params;
        auto result = at::gru(
            unwrap(input), unwrap(h_0), params,
            /*has_biases=*/true, num_layers, /*dropout=*/0.0,
            /*train=*/true, /*bidirectional=*/false, batch_first);
        *output = wrap(std::get<0>(result));
        *h_n = wrap(std::get<1>(result));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- cuDNN benchmark ---

extern "C" void flodl_set_cudnn_benchmark(int enable) {
    try {
    at::globalContext().setBenchmarkCuDNN(enable != 0);
    } catch (const std::exception& e) {
        flodl_fatal("flodl_set_cudnn_benchmark", e.what());
    } catch (...) {
        flodl_fatal("flodl_set_cudnn_benchmark", nullptr);
    }
}

// --- RNG seed ---

extern "C" void flodl_manual_seed(uint64_t seed) {
    try {
    torch::manual_seed(static_cast<int64_t>(seed));
    } catch (const std::exception& e) {
        flodl_fatal("flodl_manual_seed", e.what());
    } catch (...) {
        flodl_fatal("flodl_manual_seed", nullptr);
    }
}

extern "C" void flodl_cuda_manual_seed_all(uint64_t seed) {
    try {
#ifdef FLODL_BUILD_CUDA
    torch::cuda::manual_seed_all(static_cast<int64_t>(seed));
#else
    (void)seed;
#endif
    } catch (const std::exception& e) {
        flodl_fatal("flodl_cuda_manual_seed_all", e.what());
    } catch (...) {
        flodl_fatal("flodl_cuda_manual_seed_all", nullptr);
    }
}

// --- Meshgrid ---

extern "C" char* flodl_meshgrid(FlodlTensor* tensors, int count,
                                FlodlTensor** results, int* result_count) {
    try {
        std::vector<torch::Tensor> vec;
        vec.reserve(count);
        for (int i = 0; i < count; i++) {
            vec.push_back(unwrap(tensors[i]));
        }
        auto grids = torch::meshgrid(vec, "ij");
        int n = (int)grids.size();
        *result_count = n;
        FlodlTensor* arr = (FlodlTensor*)malloc(n * sizeof(FlodlTensor));
        for (int i = 0; i < n; i++) {
            arr[i] = new torch::Tensor(grids[i]);
        }
        *results = arr;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Pairwise distance ---

extern "C" char* flodl_cdist(FlodlTensor x, FlodlTensor y, double p,
                             FlodlTensor* result) {
    try {
        auto out = torch::cdist(unwrap(x), unwrap(y), p);
        *result = new torch::Tensor(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Cosine similarity ---

extern "C" char* flodl_cosine_similarity(FlodlTensor a, FlodlTensor b,
                                          int64_t dim, double eps,
                                          FlodlTensor* result) {
    try {
        namespace F = torch::nn::functional;
        auto opts = F::CosineSimilarityFuncOptions().dim(dim).eps(eps);
        *result = wrap(F::cosine_similarity(unwrap(a), unwrap(b), opts));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Fused Adam step ---

extern "C" char* flodl_adam_step(FlodlTensor param, FlodlTensor grad,
                                 FlodlTensor m, FlodlTensor v,
                                 double lr, double beta1, double beta2, double eps,
                                 double weight_decay, int64_t step) {
    try {
        // Get autograd-free view of param data
        auto p = unwrap(param).data();
        auto& g = unwrap(grad);
        auto& m_ref = unwrap(m);
        auto& v_ref = unwrap(v);

        // Decoupled weight decay (AdamW): p *= (1 - lr * wd)
        if (weight_decay > 0.0) {
            p.mul_(1.0 - lr * weight_decay);
        }

        // First moment: m = beta1 * m + (1 - beta1) * g  [1 fused kernel]
        m_ref.mul_(beta1).add_(g, 1.0 - beta1);

        // Second moment: v = beta2 * v + (1 - beta2) * g^2  [1 fused kernel]
        v_ref.mul_(beta2).addcmul_(g, g, 1.0 - beta2);

        // Bias correction
        double bc1 = 1.0 - std::pow(beta1, (double)step);
        double bc2 = 1.0 - std::pow(beta2, (double)step);
        double step_size = lr / bc1;

        // denom = sqrt(v / bc2) + eps
        auto denom = (v_ref / bc2).sqrt_().add_(eps);

        // p -= step_size * m / denom  [1 fused kernel]
        p.addcdiv_(m_ref, denom, -step_size);

        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Batched Adam step ---

extern "C" char* flodl_adam_step_batched(
        FlodlTensor* params, FlodlTensor* grads,
        FlodlTensor* ms, FlodlTensor* vs,
        double* lrs, int count,
        double beta1, double beta2, double eps,
        double weight_decay, int64_t step) {
    try {
        double bc1 = 1.0 - std::pow(beta1, (double)step);
        double bc2 = 1.0 - std::pow(beta2, (double)step);

        for (int i = 0; i < count; i++) {
            auto p = unwrap(params[i]).data();
            auto& g = unwrap(grads[i]);
            auto& m = unwrap(ms[i]);
            auto& v = unwrap(vs[i]);
            double lr = lrs[i];

            if (weight_decay > 0.0) {
                p.mul_(1.0 - lr * weight_decay);
            }
            m.mul_(beta1).add_(g, 1.0 - beta1);
            v.mul_(beta2).addcmul_(g, g, 1.0 - beta2);

            double step_size = lr / bc1;
            auto denom = (v / bc2).sqrt_().add_(eps);
            p.addcdiv_(m, denom, -step_size);
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Fused Adam/AdamW (multi-tensor kernel) ---

// Shared implementation for _fused_adam_ and _fused_adamw_.
static char* fused_adam_impl(FlodlTensor* params, FlodlTensor* grads,
                              FlodlTensor* exp_avgs, FlodlTensor* exp_avg_sqs,
                              int count, double lr,
                              double beta1, double beta2, double eps,
                              double weight_decay, const int64_t* steps_in,
                              FlodlTensor grad_scale, FlodlTensor found_inf,
                              bool adamw) {
    try {
        if (count == 0) return nullptr;

        auto p_list = unwrap_list(params, count);
        auto g_list = unwrap_list(grads, count);
        auto m_list = unwrap_list(exp_avgs, count);
        auto v_list = unwrap_list(exp_avg_sqs, count);

        // No amsgrad support — empty list
        std::vector<at::Tensor> max_v_list;

        // state_steps: per-param float32 scalar tensors on the param's
        // device — bias correction is computed per param from these.
        std::vector<at::Tensor> steps;
        steps.reserve(count);
        for (int i = 0; i < count; i++) {
            steps.push_back(torch::tensor((float)steps_in[i],
                torch::dtype(torch::kFloat32).device(p_list[i].device())));
        }

        auto gs = grad_scale
            ? c10::optional<at::Tensor>(unwrap(grad_scale))
            : c10::nullopt;
        auto fi = found_inf
            ? c10::optional<at::Tensor>(unwrap(found_inf))
            : c10::nullopt;

        if (adamw) {
            at::_fused_adamw_(
                p_list, g_list, m_list, v_list,
                max_v_list, steps,
                lr, beta1, beta2, weight_decay, eps,
                /*amsgrad=*/false, /*maximize=*/false, gs, fi);
        } else {
            at::_fused_adam_(
                p_list, g_list, m_list, v_list,
                max_v_list, steps,
                lr, beta1, beta2, weight_decay, eps,
                /*amsgrad=*/false, /*maximize=*/false, gs, fi);
        }

        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_fused_adam_(
        FlodlTensor* params, FlodlTensor* grads,
        FlodlTensor* exp_avgs, FlodlTensor* exp_avg_sqs,
        int count, double lr,
        double beta1, double beta2, double eps,
        double weight_decay, const int64_t* steps,
        FlodlTensor grad_scale, FlodlTensor found_inf) {
    try {
    return fused_adam_impl(params, grads, exp_avgs, exp_avg_sqs,
        count, lr, beta1, beta2, eps, weight_decay, steps,
        grad_scale, found_inf, /*adamw=*/false);
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_fused_adamw_(
        FlodlTensor* params, FlodlTensor* grads,
        FlodlTensor* exp_avgs, FlodlTensor* exp_avg_sqs,
        int count, double lr,
        double beta1, double beta2, double eps,
        double weight_decay, const int64_t* steps,
        FlodlTensor grad_scale, FlodlTensor found_inf) {
    try {
    return fused_adam_impl(params, grads, exp_avgs, exp_avg_sqs,
        count, lr, beta1, beta2, eps, weight_decay, steps,
        grad_scale, found_inf, /*adamw=*/true);
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Pinned memory ---

extern "C" char* flodl_pin_memory(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).pin_memory());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" int flodl_is_pinned(FlodlTensor t) {
    try {
    return unwrap(t).is_pinned() ? 1 : 0;
    } catch (const std::exception& e) {
        flodl_fatal("flodl_is_pinned", e.what());
    } catch (...) {
        flodl_fatal("flodl_is_pinned", nullptr);
    }
}

// --- Memory format ---

extern "C" char* flodl_to_channels_last(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).to(torch::MemoryFormat::ChannelsLast));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" int flodl_is_channels_last(FlodlTensor t) {
    try {
    return unwrap(t).is_contiguous(at::MemoryFormat::ChannelsLast) ? 1 : 0;
    } catch (const std::exception& e) {
        flodl_fatal("flodl_is_channels_last", e.what());
    } catch (...) {
        flodl_fatal("flodl_is_channels_last", nullptr);
    }
}

// --- Memory diagnostics ---

extern "C" int flodl_malloc_trim() {
    try {
#ifdef __linux__
    return malloc_trim(0);
#else
    return 0;
#endif
    } catch (const std::exception& e) {
        flodl_fatal("flodl_malloc_trim", e.what());
    } catch (...) {
        flodl_fatal("flodl_malloc_trim", nullptr);
    }
}

// --- Fused clip_grad_norm (uses _foreach_* for batched kernels) ---

extern "C" char* flodl_clip_grad_norm(FlodlTensor* params, int count,
                                       double max_norm, double* total_norm_out) {
    try {
        // Collect mutable grad tensors
        std::vector<at::Tensor> grads;
        grads.reserve(count);
        for (int i = 0; i < count; i++) {
            auto& p = unwrap(params[i]);
            if (p.grad().defined()) {
                grads.push_back(p.mutable_grad());
            }
        }
        if (grads.empty()) {
            *total_norm_out = 0.0;
            return nullptr;
        }

        // Batched norm: 1 kernel for all grads instead of N
        auto norms = at::_foreach_norm(grads, 2.0);
        double total = at::stack(norms).norm().item<double>();
        *total_norm_out = total;

        // Batched scale: 1 kernel for all grads instead of N
        if (total > max_norm) {
            double scale = max_norm / (total + 1e-6);
            at::_foreach_mul_(grads, scale);
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Multi-tensor foreach operations ---

extern "C" char* flodl_foreach_add_scalar_(FlodlTensor* tensors, int count, double scalar) {
    try {
        if (count == 0) return nullptr;
        auto list = unwrap_list(tensors, count);
        at::_foreach_add_(list, scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_foreach_mul_scalar_(FlodlTensor* tensors, int count, double scalar) {
    try {
        if (count == 0) return nullptr;
        auto list = unwrap_list(tensors, count);
        at::_foreach_mul_(list, scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_foreach_zero_(FlodlTensor* tensors, int count) {
    try {
        if (count == 0) return nullptr;
        auto list = unwrap_list(tensors, count);
        at::_foreach_zero_(list);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_foreach_add_list_(FlodlTensor* tensors1, FlodlTensor* tensors2,
                                          int count, double alpha) {
    try {
        if (count == 0) return nullptr;
        auto list1 = unwrap_list(tensors1, count);
        auto list2 = unwrap_list(tensors2, count);
        at::_foreach_add_(list1, list2, alpha);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_foreach_norm(FlodlTensor* tensors, int count, double ord,
                                     FlodlTensor* results) {
    try {
        if (count == 0) return nullptr;
        auto list = unwrap_list(tensors, count);
        auto norms = at::_foreach_norm(list, ord);
        for (size_t i = 0; i < norms.size(); i++) {
            results[i] = wrap(norms[i]);
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_foreach_lerp_scalar_(FlodlTensor* tensors1, FlodlTensor* tensors2,
                                             int count, double weight) {
    try {
        if (count == 0) return nullptr;
        auto list1 = unwrap_list(tensors1, count);
        auto list2 = unwrap_list(tensors2, count);
        at::_foreach_lerp_(list1, list2, weight);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_foreach_sqrt_(FlodlTensor* tensors, int count) {
    try {
        if (count == 0) return nullptr;
        auto list = unwrap_list(tensors, count);
        at::_foreach_sqrt_(list);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Autograd diagnostics ---

extern "C" int64_t flodl_autograd_node_count(FlodlTensor t) {
    try {
    auto& tensor = unwrap(t);
    auto fn = tensor.grad_fn();
    if (!fn) return 0;

    std::unordered_set<torch::autograd::Node*> visited;
    std::queue<torch::autograd::Node*> q;
    q.push(fn.get());
    visited.insert(fn.get());

    while (!q.empty()) {
        auto* node = q.front();
        q.pop();
        for (auto& edge : node->next_edges()) {
            auto* next = edge.function.get();
            if (next && visited.insert(next).second) {
                q.push(next);
            }
        }
    }
    return static_cast<int64_t>(visited.size());
    } catch (const std::exception& e) {
        flodl_fatal("flodl_autograd_node_count", e.what());
    } catch (...) {
        flodl_fatal("flodl_autograd_node_count", nullptr);
    }
}

// --- Fused loss functions ---

extern "C" char* flodl_mse_loss(FlodlTensor pred, FlodlTensor target,
                                 int64_t reduction, FlodlTensor* result) {
    try {
        *result = wrap(torch::mse_loss(unwrap(pred), unwrap(target), reduction));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_cross_entropy_loss(FlodlTensor pred, FlodlTensor target,
                                           int64_t reduction, int64_t ignore_index,
                                           double label_smoothing, FlodlTensor* result) {
    try {
        *result = wrap(at::cross_entropy_loss(
            unwrap(pred), unwrap(target),
            /*weight=*/{}, reduction, ignore_index, label_smoothing));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_bce_with_logits_loss(FlodlTensor pred, FlodlTensor target,
                                              int64_t reduction, FlodlTensor* result) {
    try {
        *result = wrap(torch::binary_cross_entropy_with_logits(
            unwrap(pred), unwrap(target),
            /*weight=*/{}, /*pos_weight=*/{}, reduction));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_bce_loss(FlodlTensor pred, FlodlTensor target,
                                 int64_t reduction, FlodlTensor* result) {
    try {
        *result = wrap(at::binary_cross_entropy(
            unwrap(pred), unwrap(target),
            /*weight=*/{}, reduction));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_l1_loss(FlodlTensor pred, FlodlTensor target,
                                int64_t reduction, FlodlTensor* result) {
    try {
        *result = wrap(torch::l1_loss(unwrap(pred), unwrap(target), reduction));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_smooth_l1_loss(FlodlTensor pred, FlodlTensor target,
                                       int64_t reduction, double beta,
                                       FlodlTensor* result) {
    try {
        *result = wrap(torch::smooth_l1_loss(unwrap(pred), unwrap(target),
                                              reduction, beta));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_kl_div_loss(FlodlTensor input, FlodlTensor target,
                                    int64_t reduction, int log_target,
                                    FlodlTensor* result) {
    try {
        *result = wrap(torch::kl_div(unwrap(input), unwrap(target),
                                      reduction, log_target != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_nll_loss(FlodlTensor input, FlodlTensor target,
                                int64_t reduction, int64_t ignore_index,
                                FlodlTensor* result) {
    try {
        *result = wrap(at::nll_loss(unwrap(input), unwrap(target),
                                    {}, reduction, ignore_index));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_ctc_loss(FlodlTensor log_probs, FlodlTensor targets,
                                FlodlTensor input_lengths, FlodlTensor target_lengths,
                                int64_t blank, int64_t reduction,
                                FlodlTensor* result) {
    try {
        *result = wrap(at::ctc_loss(unwrap(log_probs), unwrap(targets),
                                    unwrap(input_lengths), unwrap(target_lengths),
                                    blank, reduction, false));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Fused batch normalization ---

extern "C" char* flodl_batch_norm(FlodlTensor input, FlodlTensor weight,
                                   FlodlTensor bias, FlodlTensor running_mean,
                                   FlodlTensor running_var, int training,
                                   double momentum, double eps,
                                   FlodlTensor* result) {
    try {
        c10::optional<torch::Tensor> w = weight ? c10::make_optional(unwrap(weight))
                                                : c10::nullopt;
        c10::optional<torch::Tensor> b = bias ? c10::make_optional(unwrap(bias))
                                              : c10::nullopt;
        c10::optional<torch::Tensor> rm = running_mean
            ? c10::make_optional(unwrap(running_mean)) : c10::nullopt;
        c10::optional<torch::Tensor> rv = running_var
            ? c10::make_optional(unwrap(running_var)) : c10::nullopt;

        *result = wrap(torch::batch_norm(unwrap(input), w, b, rm, rv,
                       training != 0, momentum, eps, /*cudnn_enabled=*/true));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Fused dropout ---

extern "C" char* flodl_dropout(FlodlTensor input, double p, int training,
                                FlodlTensor* result) {
    try {
        *result = wrap(torch::dropout(unwrap(input), p, training != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_feature_dropout(FlodlTensor input, double p, int training,
                                        FlodlTensor* result) {
    try {
        *result = wrap(torch::feature_dropout(unwrap(input), p, training != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Embedding lookup ---
// padding_idx = -1 disables padding. When set to a valid index, the row at
// that index contributes zero gradient during backward (native libtorch
// behavior via at::embedding).

extern "C" char* flodl_embedding(FlodlTensor weight, FlodlTensor indices,
                                  int64_t padding_idx,
                                  int scale_grad_by_freq, int sparse,
                                  FlodlTensor* result) {
    try {
        *result = wrap(at::embedding(
            unwrap(weight), unwrap(indices),
            /*padding_idx=*/padding_idx,
            /*scale_grad_by_freq=*/scale_grad_by_freq != 0,
            /*sparse=*/sparse != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Embedding bag ---

extern "C" char* flodl_embedding_bag(FlodlTensor weight, FlodlTensor indices,
                                      FlodlTensor offsets, int64_t mode,
                                      FlodlTensor* result) {
    try {
        auto out = std::get<0>(at::embedding_bag(
            unwrap(weight), unwrap(indices), unwrap(offsets),
            /*scale_grad_by_freq=*/false, /*mode=*/mode,
            /*sparse=*/false, /*per_sample_weights=*/{},
            /*include_last_offset=*/false));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- In-place copy ---

extern "C" char* flodl_copy_(FlodlTensor dst, FlodlTensor src, int non_blocking) {
    try {
        unwrap(dst).copy_(unwrap(src), non_blocking != 0);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}
