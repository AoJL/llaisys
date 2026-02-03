#include "op.hpp"
#include <cmath>  

namespace llaisys::ops {
template <typename T>
void swiglu_kernel(tensor_t out, tensor_t gate, tensor_t up) {
    size_t n = out->numel();
    T *p_out = (T*)out->data(), *p_g = (T*)gate->data(), *p_u = (T*)up->data();
    for (size_t i = 0; i < n; ++i) {
        float g = utils::cast<float>(p_g[i]);
        // SiLU 公式: g * sigmoid(g)
        float silu = g / (1.0f + expf(-g));
        // SwiGLU 公式: SiLU(gate) * up
        p_out[i] = utils::cast<T>(silu * utils::cast<float>(p_u[i]));
    }
}
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    switch (gate->dtype()) {
        case LLAISYS_DTYPE_F32:  swiglu_kernel<float>(out, gate, up); break;
        case LLAISYS_DTYPE_F16:  swiglu_kernel<fp16_t>(out, gate, up); break;
        case LLAISYS_DTYPE_BF16: swiglu_kernel<bf16_t>(out, gate, up); break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(gate->dtype());
    }
}
} // namespace llaisys::ops
