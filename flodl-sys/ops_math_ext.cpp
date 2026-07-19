// ops_math_ext.cpp — extended math and tensor-manipulation ops.
//
// Covers: tensor-tensor comparisons, element-wise binary (max/min),
// additional reductions (var/std/argmin), trig/rounding/sign ops,
// fused mul-add (addmm/addcmul/addcdiv), Tier-3 reductions,
// query ops (nonzero, unique, searchsorted), advanced shape
// (diagonal, movedim, tile), gather/scatter_add, sorting, additional
// tensor creation (eye/full/randperm/multinomial), normalize,
// chunk/split/pad/interpolate/flip/roll/unbind/contiguous/argsort.

#include "helpers.h"

// --- Comparison (tensor-tensor, return float masks: 0.0 or 1.0) ---

extern "C" char* flodl_gt_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::gt(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_lt_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::lt(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_ge_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::ge(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_le_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::le(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_eq_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::eq(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_ne_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::ne(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Element-wise binary (differentiable) ---

extern "C" char* flodl_atan2(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(torch::atan2(unwrap(a), unwrap(b)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_maximum(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(torch::maximum(unwrap(a), unwrap(b)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_minimum(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(torch::minimum(unwrap(a), unwrap(b)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Additional reductions ---

extern "C" char* flodl_argmin(FlodlTensor t, int dim, int keepdim, FlodlTensor* result) {
    try {
        *result = wrap(torch::argmin(unwrap(t), dim, (bool)keepdim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_var(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).var());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_std_op(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).std());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_var_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).var({(int64_t)dim}, 1, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_std_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).std({(int64_t)dim}, 1, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Element-wise math (trig, rounding, sign) ---

extern "C" char* flodl_sin(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::sin(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_cos(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::cos(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_sign(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::sign(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_floor(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::floor(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_ceil(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::ceil(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_round(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::round(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_reciprocal(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::reciprocal(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_tan(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::tan(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_asin(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::asin(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_acos(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::acos(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_atan(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::atan(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_erf(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::erf(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_erfc(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::erfc(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_trunc(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::trunc(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_frac(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::frac(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_fmod_scalar(FlodlTensor t, double scalar,
                                   FlodlTensor* result) {
    try {
        *result = wrap(torch::fmod(unwrap(t), scalar));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_fmod_tensor(FlodlTensor a, FlodlTensor b,
                                   FlodlTensor* result) {
    try {
        *result = wrap(torch::fmod(unwrap(a), unwrap(b)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_remainder_scalar(FlodlTensor t, double scalar,
                                        FlodlTensor* result) {
    try {
        *result = wrap(torch::remainder(unwrap(t), scalar));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_remainder_tensor(FlodlTensor a, FlodlTensor b,
                                        FlodlTensor* result) {
    try {
        *result = wrap(torch::remainder(unwrap(a), unwrap(b)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_lerp(FlodlTensor a, FlodlTensor b, double weight,
                            FlodlTensor* result) {
    try {
        *result = wrap(torch::lerp(unwrap(a), unwrap(b), weight));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_lerp_tensor(FlodlTensor a, FlodlTensor b,
                                   FlodlTensor weight, FlodlTensor* result) {
    try {
        *result = wrap(torch::lerp(unwrap(a), unwrap(b), unwrap(weight)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_isclose(FlodlTensor a, FlodlTensor b,
                               double rtol, double atol,
                               FlodlTensor* result) {
    try {
        auto out = torch::isclose(unwrap(a), unwrap(b), rtol, atol);
        *result = wrap(out.to(unwrap(a).dtype()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Fused mul-add ---

extern "C" char* flodl_addmm(FlodlTensor bias, FlodlTensor mat1,
                              FlodlTensor mat2, double beta, double alpha,
                              FlodlTensor* result) {
    try {
        *result = wrap(torch::addmm(unwrap(bias), unwrap(mat1), unwrap(mat2),
                                    beta, alpha));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_addcmul(FlodlTensor self, FlodlTensor t1,
                                FlodlTensor t2, double value,
                                FlodlTensor* result) {
    try {
        *result = wrap(torch::addcmul(unwrap(self), unwrap(t1), unwrap(t2),
                                      value));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_addcdiv(FlodlTensor self, FlodlTensor t1,
                                FlodlTensor t2, double value,
                                FlodlTensor* result) {
    try {
        *result = wrap(torch::addcdiv(unwrap(self), unwrap(t1), unwrap(t2),
                                      value));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Additional reductions (Tier 3) ---

extern "C" char* flodl_cumprod(FlodlTensor t, int dim, FlodlTensor* result) {
    try {
        *result = wrap(torch::cumprod(unwrap(t), dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_norm_p_dim(FlodlTensor t, double p, int dim,
                                  int keepdim, FlodlTensor* result) {
    try {
        *result = wrap(torch::norm(unwrap(t), p, dim, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_sum_dims(FlodlTensor t, int64_t* dims, int ndims,
                                int keepdim, FlodlTensor* result) {
    try {
        std::vector<int64_t> dim_vec(dims, dims + ndims);
        *result = wrap(torch::sum(unwrap(t), dim_vec, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_median(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::median(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_median_dim(FlodlTensor t, int dim, int keepdim,
                                  FlodlTensor* values, FlodlTensor* indices) {
    try {
        auto [v, i] = torch::median(unwrap(t), dim, keepdim != 0);
        *values = wrap(v);
        *indices = wrap(i);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_count_nonzero(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::count_nonzero(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_count_nonzero_dim(FlodlTensor t, int dim,
                                         FlodlTensor* result) {
    try {
        std::optional<int64_t> d(static_cast<int64_t>(dim));
        *result = wrap(torch::count_nonzero(unwrap(t), d));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Query ops ---

extern "C" char* flodl_nonzero(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::nonzero(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_unique(FlodlTensor t, int sorted, int return_inverse,
                              FlodlTensor* output,
                              FlodlTensor* inverse_indices) {
    try {
        // Global dedup for both sorted values; `sorted` only controls output
        // ordering (unsorted order is kernel-defined, as in PyTorch).
        auto [out, inv, _counts] = at::_unique2(unwrap(t), sorted != 0,
                                                return_inverse != 0, false);
        *output = wrap(out);
        if (return_inverse != 0)
            *inverse_indices = wrap(inv);
        else
            *inverse_indices = nullptr;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_unique_consecutive(FlodlTensor t, int return_inverse,
                                          FlodlTensor* output,
                                          FlodlTensor* inverse_indices) {
    try {
        auto [out, inv, _counts] = torch::unique_consecutive(
            unwrap(t), return_inverse != 0, false);
        *output = wrap(out);
        if (return_inverse != 0)
            *inverse_indices = wrap(inv);
        else
            *inverse_indices = nullptr;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_searchsorted(FlodlTensor sorted_seq, FlodlTensor values,
                                    FlodlTensor* result) {
    try {
        *result = wrap(torch::searchsorted(unwrap(sorted_seq), unwrap(values)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Shape ops (advanced) ---

extern "C" char* flodl_diagonal(FlodlTensor t, int64_t offset, int dim1,
                                int dim2, FlodlTensor* result) {
    try {
        *result = wrap(torch::diagonal(unwrap(t), offset, dim1, dim2));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_movedim(FlodlTensor t, int64_t src, int64_t dst,
                               FlodlTensor* result) {
    try {
        *result = wrap(torch::movedim(unwrap(t), src, dst));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_tile(FlodlTensor t, int64_t* reps, int ndim,
                            FlodlTensor* result) {
    try {
        std::vector<int64_t> r(reps, reps + ndim);
        *result = wrap(torch::tile(unwrap(t), r));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Advanced indexing ---

extern "C" char* flodl_gather(FlodlTensor t, int dim, FlodlTensor index,
                              FlodlTensor* result) {
    try {
        *result = wrap(torch::gather(unwrap(t), dim, unwrap(index)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_scatter_add(FlodlTensor t, int dim, FlodlTensor index,
                                   FlodlTensor src, FlodlTensor* result) {
    try {
        auto out = unwrap(t).clone();
        out.scatter_add_(dim, unwrap(index), unwrap(src));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Sorting ---

extern "C" char* flodl_topk(FlodlTensor t, int64_t k, int dim, int largest, int sorted,
                            FlodlTensor* values, FlodlTensor* indices) {
    try {
        auto [v, i] = torch::topk(unwrap(t), k, dim, largest != 0, sorted != 0);
        *values = wrap(v);
        *indices = wrap(i);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_sort(FlodlTensor t, int dim, int descending,
                            FlodlTensor* values, FlodlTensor* indices) {
    try {
        auto [v, i] = torch::sort(unwrap(t), dim, descending != 0);
        *values = wrap(v);
        *indices = wrap(i);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Tensor creation (additional) ---

extern "C" char* flodl_eye(int64_t n, int dtype, int device_type,
                          int device_index, FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::eye(n, options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_full(int64_t* shape, int ndim, double value, int dtype,
                            int device_type, int device_index,
                            FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::full(make_shape(shape, ndim), value, options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_randperm(int64_t n, int dtype, int device_type,
                                int device_index, FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::randperm(n, options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_multinomial(FlodlTensor probs, int64_t num_samples,
                                    int replacement, FlodlTensor* result) {
    try {
        *result = wrap(torch::multinomial(unwrap(probs), num_samples,
                                           replacement != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Normalization ---

extern "C" char* flodl_normalize(FlodlTensor t, double p, int dim,
                                  FlodlTensor* result) {
    try {
        *result = wrap(torch::nn::functional::normalize(
            unwrap(t),
            torch::nn::functional::NormalizeFuncOptions().p(p).dim(dim)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Shape operations (additional) ---

extern "C" char* flodl_chunk(FlodlTensor t, int chunks, int dim,
                             FlodlTensor** results, int* count) {
    try {
        auto chunks_vec = torch::chunk(unwrap(t), chunks, dim);
        std::vector<torch::Tensor> contig;
        contig.reserve(chunks_vec.size());
        for (auto& c : chunks_vec) { contig.push_back(c.contiguous()); }
        FlodlTensor* arr = wrap_list(contig);
        if (!arr) {
            return make_error("flodl_chunk: malloc failed");
        }
        *results = arr;
        *count = (int)contig.size();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_repeat(FlodlTensor t, int64_t* repeats, int ndim,
                              FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).repeat(make_shape(repeats, ndim)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_pad(FlodlTensor t, int64_t* padding, int pad_len, double value,
                           FlodlTensor* result) {
    try {
        *result = wrap(at::constant_pad_nd(unwrap(t),
                       torch::IntArrayRef(padding, pad_len), value));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// mode: 0=constant, 1=reflect, 2=replicate, 3=circular
extern "C" char* flodl_pad_mode(FlodlTensor t, int64_t* padding, int pad_len,
                                 int mode, double value, FlodlTensor* result) {
    try {
        namespace F = torch::nn::functional;
        auto pad_vec = std::vector<int64_t>(padding, padding + pad_len);
        auto opts = F::PadFuncOptions(pad_vec);
        if (mode == 1) {
            opts = opts.mode(torch::kReflect);
        } else if (mode == 2) {
            opts = opts.mode(torch::kReplicate);
        } else if (mode == 3) {
            opts = opts.mode(torch::kCircular);
        } else {
            opts = opts.mode(torch::kConstant).value(value);
        }
        *result = wrap(F::pad(unwrap(t), opts));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

// --- Interpolation ---
// mode: 0=nearest, 1=bilinear, 2=bicubic, 3=trilinear
extern "C" char* flodl_interpolate(FlodlTensor input, int64_t* output_size, int ndim,
                                    int mode, int align_corners,
                                    FlodlTensor* result) {
    try {
        namespace F = torch::nn::functional;
        auto opts = F::InterpolateFuncOptions()
            .size(std::vector<int64_t>(output_size, output_size + ndim));
        switch (mode) {
            case 0: opts = opts.mode(torch::kNearest); break;
            case 1: opts = opts.mode(torch::kBilinear).align_corners(align_corners != 0); break;
            case 2: opts = opts.mode(torch::kBicubic).align_corners(align_corners != 0); break;
            case 3: opts = opts.mode(torch::kTrilinear).align_corners(align_corners != 0); break;
            default: return make_error("flodl_interpolate: invalid mode");
        }
        *result = wrap(F::interpolate(unwrap(input), opts));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_flip(FlodlTensor t, int64_t* dims, int ndim,
                             FlodlTensor* result) {
    try {
        *result = wrap(torch::flip(unwrap(t), torch::IntArrayRef(dims, ndim)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_roll(FlodlTensor t, int64_t shift, int dim,
                             FlodlTensor* result) {
    try {
        *result = wrap(torch::roll(unwrap(t), shift, dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_split(FlodlTensor t, int64_t split_size, int dim,
                              FlodlTensor** results, int* count) {
    try {
        auto splits = torch::split(unwrap(t), split_size, dim);
        std::vector<torch::Tensor> contig;
        contig.reserve(splits.size());
        for (auto& s : splits) { contig.push_back(s.contiguous()); }
        FlodlTensor* arr = wrap_list(contig);
        if (!arr) {
            return make_error("flodl_split: malloc failed");
        }
        *results = arr;
        *count = (int)contig.size();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_unbind(FlodlTensor t, int dim,
                               FlodlTensor** results, int* count) {
    try {
        auto slices = torch::unbind(unwrap(t), dim);
        std::vector<torch::Tensor> contig;
        contig.reserve(slices.size());
        for (auto& s : slices) { contig.push_back(s.contiguous()); }
        FlodlTensor* arr = wrap_list(contig);
        if (!arr) {
            return make_error("flodl_unbind: malloc failed");
        }
        *results = arr;
        *count = (int)contig.size();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_contiguous(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" int flodl_is_contiguous(FlodlTensor t) {
    try {
    return unwrap(t).is_contiguous() ? 1 : 0;
    } catch (const std::exception& e) {
        flodl_fatal("flodl_is_contiguous", e.what());
    } catch (...) {
        flodl_fatal("flodl_is_contiguous", nullptr);
    }
}

extern "C" char* flodl_argsort(FlodlTensor t, int dim, int descending,
                                FlodlTensor* result) {
    try {
        *result = wrap(torch::argsort(unwrap(t), dim, descending != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}

extern "C" char* flodl_scatter(FlodlTensor t, int dim, FlodlTensor index,
                                FlodlTensor src, FlodlTensor* result) {
    try {
        auto out = unwrap(t).clone();
        out.scatter_(dim, unwrap(index), unwrap(src));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("flodl: non-standard C++ exception");
    }
}
