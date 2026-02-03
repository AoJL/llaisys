#include "op.hpp"

namespace llaisys::ops {
template <typename T>
void linear_cpu_kernel(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    size_t M = in->shape()[0], K = in->shape()[1], N = weight->shape()[0];
    const T *p_in = (T*)in->data(), *p_w = (T*)weight->data(), *p_b = bias ? (T*)bias->data() : nullptr;
    T* p_out = (T*)out->data();
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = p_b ? utils::cast<float>(p_b[j]) : 0.0f;
            for (size_t k = 0; k < K; ++k) 
                sum += utils::cast<float>(p_in[i*K+k]) * utils::cast<float>(p_w[j*K+k]);
            p_out[i*N+j] = utils::cast<T>(sum);
        }
    }
} 
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:  linear_cpu_kernel<float>(out, in, weight, bias); break;
        case LLAISYS_DTYPE_F16:  linear_cpu_kernel<fp16_t>(out, in, weight, bias); break;
        case LLAISYS_DTYPE_BF16: linear_cpu_kernel<bf16_t>(out, in, weight, bias); break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops
