// ops_tensor.cpp — core tensor operations.
//
// Covers: creation, lifecycle, metadata, data access, arithmetic,
// activations, norms (layer/group), element-wise math, reductions,
// scalar/boolean comparisons, shape operations, scatter, indexing,
// concatenation, masking, like-constructors.

#include "helpers.h"
#include <torch/csrc/autograd/function.h>

// --- Tensor creation ---

extern "C" char* flodl_zeros(int64_t* shape, int ndim, int dtype,
                            int device_type, int device_index,
                            FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::zeros(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_ones(int64_t* shape, int ndim, int dtype,
                           int device_type, int device_index,
                           FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::ones(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_rand(int64_t* shape, int ndim, int dtype,
                           int device_type, int device_index,
                           FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::rand(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_randn(int64_t* shape, int ndim, int dtype,
                             int device_type, int device_index,
                             FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::randn(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_from_blob(void* data, int64_t* shape, int ndim,
                                int dtype, int device_type, int device_index,
                                FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions().dtype(to_scalar_type(dtype));
        // from_blob does not take ownership — clone to get an independent copy.
        auto t = torch::from_blob(data, make_shape(shape, ndim), options).clone();
        if (device_type == FLODL_CUDA) {
            t = t.to(to_device(device_type, device_index));
        }
        *result = wrap(std::move(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_linspace(double start, double end, int64_t steps,
                               int dtype, int device_type, int device_index,
                               FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions().dtype(to_scalar_type(dtype));
        auto t = torch::linspace(start, end, steps, options);
        if (device_type == FLODL_CUDA) {
            t = t.to(to_device(device_type, device_index));
        }
        *result = wrap(std::move(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_arange(double start, double end, double step,
                              int dtype, int device_type, int device_index,
                              FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::arange(start, end, step, options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_expand(FlodlTensor t, int64_t* new_shape, int ndim,
                             FlodlTensor* result) {
    try {
        // expand returns a view; contiguous() makes an owned copy.
        *result = wrap(unwrap(t).expand(make_shape(new_shape, ndim)).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Tensor lifecycle ---

extern "C" void flodl_free_tensor(FlodlTensor t) {
    try {
    if (t) {
        delete (torch::Tensor*)t;
    }
    } catch (const std::exception& e) {
        flodl_fatal("flodl_free_tensor", e.what());
    } catch (...) {
        flodl_fatal("flodl_free_tensor", nullptr);
    }
}

extern "C" char* flodl_shallow_clone(FlodlTensor t, FlodlTensor* result) {
    try {
        auto* src = reinterpret_cast<torch::Tensor*>(t);
        *result = new torch::Tensor(*src);  // Copy ctor: shares TensorImpl, bumps refcount.
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Tensor metadata ---

extern "C" int flodl_ndim(FlodlTensor t) {
    try {
    return (int)unwrap(t).dim();
    } catch (const std::exception& e) {
        flodl_fatal("flodl_ndim", e.what());
    } catch (...) {
        flodl_fatal("flodl_ndim", nullptr);
    }
}

extern "C" int64_t flodl_shape(FlodlTensor t, int dim) {
    try {
    return unwrap(t).size(dim);
    } catch (const std::exception& e) {
        flodl_fatal("flodl_shape", e.what());
    } catch (...) {
        flodl_fatal("flodl_shape", nullptr);
    }
}

extern "C" int flodl_dtype(FlodlTensor t) {
    try {
    return from_scalar_type(unwrap(t).scalar_type());
    } catch (const std::exception& e) {
        flodl_fatal("flodl_dtype", e.what());
    } catch (...) {
        flodl_fatal("flodl_dtype", nullptr);
    }
}

extern "C" int flodl_device_type(FlodlTensor t) {
    try {
    return from_device_type(unwrap(t).device());
    } catch (const std::exception& e) {
        flodl_fatal("flodl_device_type", e.what());
    } catch (...) {
        flodl_fatal("flodl_device_type", nullptr);
    }
}

extern "C" int flodl_device_index(FlodlTensor t) {
    try {
    return from_device_index(unwrap(t).device());
    } catch (const std::exception& e) {
        flodl_fatal("flodl_device_index", e.what());
    } catch (...) {
        flodl_fatal("flodl_device_index", nullptr);
    }
}

extern "C" int64_t flodl_numel(FlodlTensor t) {
    try {
    return unwrap(t).numel();
    } catch (const std::exception& e) {
        flodl_fatal("flodl_numel", e.what());
    } catch (...) {
        flodl_fatal("flodl_numel", nullptr);
    }
}

extern "C" int64_t flodl_storage_nbytes(FlodlTensor t) {
    try {
    const auto& tensor = unwrap(t);
    if (!tensor.has_storage()) {
        return tensor.numel() * tensor.element_size();
    }
    return static_cast<int64_t>(tensor.storage().nbytes());
    } catch (const std::exception& e) {
        flodl_fatal("flodl_storage_nbytes", e.what());
    } catch (...) {
        flodl_fatal("flodl_storage_nbytes", nullptr);
    }
}

// --- Data access ---

extern "C" char* flodl_copy_data(FlodlTensor t, void* buffer,
                                int64_t buffer_bytes) {
    try {
        auto tensor = unwrap(t);
        // Move to CPU if on another device
        if (!tensor.is_cpu()) {
            tensor = tensor.to(torch::kCPU);
        }
        // Ensure contiguous layout
        tensor = tensor.contiguous();
        int64_t data_bytes = tensor.numel() * tensor.element_size();
        if (buffer_bytes < data_bytes) {
            return make_error("buffer too small: need " +
                              std::to_string(data_bytes) + " bytes, got " +
                              std::to_string(buffer_bytes));
        }
        memcpy(buffer, tensor.data_ptr(), data_bytes);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Arithmetic ---

extern "C" char* flodl_add(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(a) + unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_sub(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(a) - unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_mul(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(a) * unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_div(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(a) / unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_matmul(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(torch::matmul(unwrap(a), unwrap(b)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_add_scalar(FlodlTensor t, double scalar,
                                 FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t) + scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_mul_scalar(FlodlTensor t, double scalar,
                                 FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t) * scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_div_scalar(FlodlTensor t, double scalar,
                                 FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t) / scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_neg(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(-unwrap(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Activations ---

extern "C" char* flodl_relu(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::relu(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_sigmoid(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::sigmoid(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_tanh_op(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::tanh(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_softmax(FlodlTensor t, int dim, FlodlTensor* result) {
    try {
        *result = wrap(torch::softmax(unwrap(t), dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_log_softmax(FlodlTensor t, int dim, FlodlTensor* result) {
    try {
        *result = wrap(torch::log_softmax(unwrap(t), dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_gelu(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::gelu(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// Tanh-approximation GELU — matches HuggingFace `hidden_act="gelu_new"`
// (and PyTorch's `F.gelu(x, approximate="tanh")`):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// The plain `flodl_gelu` above is the erf form; both share `Variable::gelu_*`
// surfaces in flodl/src/autograd/ops.rs.
extern "C" char* flodl_gelu_tanh(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::gelu(unwrap(t), "tanh"));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_silu(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::silu(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_leaky_relu(FlodlTensor t, double negative_slope,
                                   FlodlTensor* result) {
    try {
        *result = wrap(torch::nn::functional::leaky_relu(
            unwrap(t),
            torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_elu(FlodlTensor t, double alpha, FlodlTensor* result) {
    try {
        *result = wrap(torch::elu(unwrap(t), alpha));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_softplus(FlodlTensor t, double beta, double threshold,
                                 FlodlTensor* result) {
    try {
        *result = wrap(torch::nn::functional::softplus(
            unwrap(t),
            torch::nn::functional::SoftplusFuncOptions().beta(beta).threshold(threshold)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_mish(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::mish(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_selu(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::selu(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_hardswish(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::hardswish(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_hardsigmoid(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::hardsigmoid(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_prelu(FlodlTensor t, FlodlTensor weight,
                              FlodlTensor* result) {
    try {
        *result = wrap(torch::prelu(unwrap(t), unwrap(weight)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Layer normalization ---

extern "C" char* flodl_native_layer_norm(FlodlTensor input, FlodlTensor weight,
                                         FlodlTensor bias, int64_t normalized_size,
                                         double eps,
                                         FlodlTensor* output, FlodlTensor* mean,
                                         FlodlTensor* rstd) {
    try {
        auto result = at::native_layer_norm(
            unwrap(input), {normalized_size},
            weight ? c10::optional<torch::Tensor>(unwrap(weight)) : c10::nullopt,
            bias ? c10::optional<torch::Tensor>(unwrap(bias)) : c10::nullopt,
            eps);
        *output = wrap(std::get<0>(result));
        *mean = wrap(std::get<1>(result));
        *rstd = wrap(std::get<2>(result));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Group normalization ---

extern "C" char* flodl_group_norm(FlodlTensor input, int64_t num_groups,
                                   FlodlTensor weight, FlodlTensor bias,
                                   double eps, FlodlTensor* result) {
    try {
        c10::optional<torch::Tensor> w = weight
            ? c10::make_optional(unwrap(weight)) : c10::nullopt;
        c10::optional<torch::Tensor> b = bias
            ? c10::make_optional(unwrap(bias)) : c10::nullopt;
        *result = wrap(at::group_norm(unwrap(input), num_groups, w, b, eps,
                       /*cudnn_enabled=*/true));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Element-wise math ---

extern "C" char* flodl_exp(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::exp(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_log(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::log(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_sqrt(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::sqrt(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_abs(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).abs());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_triu(FlodlTensor t, int64_t diagonal,
                            FlodlTensor* result) {
    try {
        *result = wrap(torch::triu(unwrap(t), diagonal));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_tril(FlodlTensor t, int64_t diagonal,
                            FlodlTensor* result) {
    try {
        *result = wrap(torch::tril(unwrap(t), diagonal));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_pow_scalar(FlodlTensor t, double exponent,
                                 FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).pow(exponent));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_clamp(FlodlTensor t, double min_val, double max_val,
                             FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).clamp(min_val, max_val));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_clamp_min(FlodlTensor t, double min_val,
                                  FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).clamp_min(min_val));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_clamp_max(FlodlTensor t, double max_val,
                                  FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).clamp_max(max_val));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_log1p(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::log1p(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_expm1(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::expm1(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_log2(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::log2(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_log10(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::log10(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Reductions ---

extern "C" char* flodl_sum(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).sum());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_mean(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).mean());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_sum_dim(FlodlTensor t, int dim, int keepdim,
                              FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).sum(dim, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_mean_dim(FlodlTensor t, int dim, int keepdim,
                               FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).mean(dim, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_prod(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).prod());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_prod_dim(FlodlTensor t, int dim, int keepdim,
                                FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).prod(dim, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_cumsum(FlodlTensor t, int dim, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).cumsum(dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_logsumexp(FlodlTensor t, int dim, int keepdim,
                                 FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).logsumexp(dim, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_min(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).min());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_max(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).max());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_norm(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).norm());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_min_dim(FlodlTensor t, int dim, int keepdim,
                              FlodlTensor* result) {
    try {
        auto [values, indices] = unwrap(t).min(dim, (bool)keepdim);
        *result = wrap(values);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_max_dim(FlodlTensor t, int dim, int keepdim,
                              FlodlTensor* result) {
    try {
        // std::get<0> gets the values (not the indices)
        *result = wrap(std::get<0>(unwrap(t).max(dim, keepdim != 0)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_argmax(FlodlTensor t, int dim, int keepdim,
                              FlodlTensor* result) {
    try {
        *result = wrap(torch::argmax(unwrap(t), dim, (bool)keepdim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Comparison (return float masks: 0.0 or 1.0) ---
// (mask_dtype lives in helpers.h — shared with ops_math_ext.cpp.)

extern "C" char* flodl_gt_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::gt(unwrap(t), scalar);
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_ge_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::ge(unwrap(t), scalar);
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_le_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::le(unwrap(t), scalar);
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_lt_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::lt(unwrap(t), scalar);
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_eq_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::eq(unwrap(t), scalar);
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_ne_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::ne(unwrap(t), scalar);
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Boolean / detection (return float masks) ---

extern "C" char* flodl_isnan(FlodlTensor t, FlodlTensor* result) {
    try {
        auto mask = torch::isnan(unwrap(t));
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_isinf(FlodlTensor t, FlodlTensor* result) {
    try {
        auto mask = torch::isinf(unwrap(t));
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_logical_and(FlodlTensor a, FlodlTensor b,
                                    FlodlTensor* result) {
    try {
        auto mask = torch::logical_and(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_logical_or(FlodlTensor a, FlodlTensor b,
                                   FlodlTensor* result) {
    try {
        auto mask = torch::logical_or(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_logical_not(FlodlTensor t, FlodlTensor* result) {
    try {
        auto mask = torch::logical_not(unwrap(t));
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_any(FlodlTensor t, FlodlTensor* result) {
    try {
        auto mask = unwrap(t).any();
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_all(FlodlTensor t, FlodlTensor* result) {
    try {
        auto mask = unwrap(t).all();
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Shape operations ---
// These return views sharing the source storage whenever libtorch does
// (PyTorch parity). Consumers needing owned contiguous bytes make their own
// copy at the point of use (see flodl_copy_data).

extern "C" char* flodl_reshape(FlodlTensor t, int64_t* shape, int ndim,
                              FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).reshape(make_shape(shape, ndim)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_transpose(FlodlTensor t, int dim0, int dim1,
                                FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).transpose(dim0, dim1));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_permute(FlodlTensor t, int64_t* dims, int ndim,
                              FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).permute(torch::IntArrayRef(dims, ndim)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_select(FlodlTensor t, int dim, int64_t index,
                             FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).select(dim, index));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_narrow(FlodlTensor t, int dim, int64_t start,
                             int64_t length, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).narrow(dim, start, length));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_squeeze(FlodlTensor t, int dim, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).squeeze(dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_unsqueeze(FlodlTensor t, int dim, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).unsqueeze(dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_flatten(FlodlTensor t, int start_dim, int end_dim,
                              FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).flatten(start_dim, end_dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Scatter ---

extern "C" char* flodl_select_scatter(FlodlTensor input, FlodlTensor src,
                                     int dim, int64_t index,
                                     FlodlTensor* result) {
    try {
        auto out = unwrap(input).clone();
        out.select(dim, index).copy_(unwrap(src));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_narrow_scatter(FlodlTensor input, FlodlTensor src,
                                     int dim, int64_t start,
                                     FlodlTensor* result) {
    try {
        auto out = unwrap(input).clone();
        out.narrow(dim, start, unwrap(src).size(dim)).copy_(unwrap(src));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Indexing ---

extern "C" char* flodl_index_select(FlodlTensor t, int dim, FlodlTensor index,
                                   FlodlTensor* result) {
    try {
        *result = wrap(torch::index_select(unwrap(t), dim, unwrap(index)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_index_add(FlodlTensor t, int dim, FlodlTensor index,
                                FlodlTensor src, FlodlTensor* result) {
    try {
        // Out-of-place: returns t with src scattered at index positions.
        *result = wrap(unwrap(t).index_add(dim, unwrap(index), unwrap(src)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Concatenation ---

extern "C" char* flodl_cat2(FlodlTensor a, FlodlTensor b, int dim,
                           FlodlTensor* result) {
    try {
        *result = wrap(torch::cat({unwrap(a), unwrap(b)}, dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_cat(FlodlTensor* tensors, int count, int dim,
                           FlodlTensor* result) {
    try {
        std::vector<at::Tensor> vec;
        vec.reserve(count);
        for (int i = 0; i < count; i++) {
            vec.push_back(unwrap(tensors[i]));
        }
        *result = wrap(torch::cat(vec, dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_stack(FlodlTensor* tensors, int count, int dim,
                            FlodlTensor* result) {
    try {
        std::vector<at::Tensor> vec;
        vec.reserve(count);
        for (int i = 0; i < count; i++) {
            vec.push_back(unwrap(tensors[i]));
        }
        *result = wrap(torch::stack(vec, dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Masking ---

extern "C" char* flodl_masked_fill(FlodlTensor t, FlodlTensor mask,
                                    double value, FlodlTensor* result) {
    try {
        auto bool_mask = unwrap(mask).to(torch::kBool);
        *result = wrap(unwrap(t).masked_fill(bool_mask, value));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Conditional ---

extern "C" char* flodl_where(FlodlTensor condition, FlodlTensor x,
                             FlodlTensor y, FlodlTensor* result) {
    try {
        auto cond = unwrap(condition).to(torch::kBool);
        *result = wrap(torch::where(cond, unwrap(x), unwrap(y)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Like constructors ---

extern "C" char* flodl_zeros_like(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::zeros_like(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_ones_like(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::ones_like(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_full_like(FlodlTensor t, double value, FlodlTensor* result) {
    try {
        *result = wrap(torch::full_like(unwrap(t), value));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_rand_like(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::rand_like(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_randn_like(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::randn_like(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_randint(int64_t low, int64_t high,
                                int64_t* shape, int ndim,
                                int dtype, int device_type, int device_index,
                                FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::randint(low, high, make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_empty(int64_t* shape, int ndim, int dtype,
                              int device_type, int device_index,
                              FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::empty(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_one_hot(FlodlTensor t, int64_t num_classes,
                                FlodlTensor* result) {
    try {
        auto oh = torch::one_hot(unwrap(t), num_classes);
        // Convert to float for consistency (one_hot returns Int64)
        *result = wrap(oh.to(torch::kFloat32));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_bernoulli(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::bernoulli(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}
