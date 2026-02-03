#include "op.hpp"
#include <cmath>  

namespace llaisys::ops {
template <typename T>
void rope_kernel(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    size_t seq_len = in->shape()[0], n_heads = in->shape()[1], d = in->shape()[2];
    const int64_t* p_pos = (int64_t*)pos_ids->data();
    
    for (size_t s = 0; s < seq_len; ++s) {
        float pos = (float)p_pos[s];
        for (size_t h = 0; h < n_heads; ++h) {
            T* p_in = (T*)in->data() + (s * n_heads + h) * d;
            T* p_out = (T*)out->data() + (s * n_heads + h) * d;
            for (size_t j = 0; j < d / 2; ++j) {
                float freq = pos / powf(theta, (2.0f * j) / d);
                float cos_a = cosf(freq), sin_a = sinf(freq);
                float x = utils::cast<float>(p_in[j]);
                float y = utils::cast<float>(p_in[j + d/2]);
                p_out[j] = utils::cast<T>(x * cos_a - y * sin_a);
                p_out[j + d/2] = utils::cast<T>(y * cos_a + x * sin_a);
            }
        }
    }
} 
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:  rope_kernel<float>(out, in, pos_ids, theta); break;
        case LLAISYS_DTYPE_F16:  rope_kernel<fp16_t>(out, in, pos_ids, theta); break;
        case LLAISYS_DTYPE_BF16: rope_kernel<bf16_t>(out, in, pos_ids, theta); break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops
