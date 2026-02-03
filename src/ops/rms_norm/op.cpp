#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
template <typename T>
void rms_norm_kernel(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    size_t d = in->shape().back();
    size_t rows = in->numel() / d;
    for (size_t i = 0; i < rows; ++i) {
        T *p_in = (T*)in->data() + i*d, *p_out = (T*)out->data() + i*d;
        float ss = 0;
        for (size_t j = 0; j < d; ++j) { float v = utils::cast<float>(p_in[j]); ss += v*v; }
        float inv_rms = 1.0f / sqrtf(ss / d + eps);
        for (size_t j = 0; j < d; ++j) 
            p_out[j] = utils::cast<T>(utils::cast<float>(p_in[j]) * inv_rms * utils::cast<float>(((T*)weight->data())[j]));
    }
}
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:  rms_norm_kernel<float>(out, in, weight, eps); break;
        case LLAISYS_DTYPE_F16:  rms_norm_kernel<fp16_t>(out, in, weight, eps); break;
        case LLAISYS_DTYPE_BF16: rms_norm_kernel<bf16_t>(out, in, weight, eps); break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops
