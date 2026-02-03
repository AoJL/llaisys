#include "op.hpp"

namespace llaisys::ops {
template<typename T>
void argmax_cpu_kernel(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    const T* p_vals = reinterpret_cast<const T*>(vals->data());
    size_t n = vals->numel();
    float best_v = -1e38;
    int64_t best_i = 0;
    for (size_t i = 0; i < n; ++i) {
        float v = utils::cast<float>(p_vals[i]);
        if (v > best_v) { best_v = v; best_i = i; }
    }
    reinterpret_cast<int64_t*>(max_idx->data())[0] = best_i;
    reinterpret_cast<T*>(max_val->data())[0] = utils::cast<T>(best_v);
}
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
 switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:  argmax_cpu_kernel<float>(max_idx, max_val, vals); break;
        case LLAISYS_DTYPE_F16:  argmax_cpu_kernel<fp16_t>(max_idx, max_val, vals); break;
        case LLAISYS_DTYPE_BF16: argmax_cpu_kernel<bf16_t>(max_idx, max_val, vals); break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }   
}
} // namespace llaisys::ops
