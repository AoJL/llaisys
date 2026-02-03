#include "op.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace llaisys::ops {
template <typename T>
void self_attention_cpu_kernel(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    size_t seq_len = q->shape()[0];    
    size_t nh = q->shape()[1];         
    size_t d_head = q->shape()[2];
    size_t total_len = k->shape()[0];  
    size_t nkvh = k->shape()[1];       
    size_t n_groups = nh / nkvh;

    T* p_q = (T*)q->data();
    T* p_k = (T*)k->data();
    T* p_v = (T*)v->data();
    T* p_out = (T*)attn_val->data();

    int diagonal_offset = (int)total_len - (int)seq_len;

    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t h = 0; h < nh; ++h) {
            size_t h_kv = h / n_groups;
            std::vector<float> scores(total_len);
            float max_score = -1e30f; 

            // --- 阶段 A: 计算 QK^T ---
            for (size_t t = 0; t < total_len; ++t) {
                if (t > (size_t)((int)s + diagonal_offset)) {
                    scores[t] = -1e30f;
                    continue;
                }

                float sum = 0.0f;
                T* q_ptr = p_q + (s * nh + h) * d_head;
                T* k_ptr = p_k + (t * nkvh + h_kv) * d_head;

                // 核心优化：内部求和使用 float 累加
                for (size_t d = 0; d < d_head; ++d) {
                    sum += utils::cast<float>(q_ptr[d]) * utils::cast<float>(k_ptr[d]);
                }
                scores[t] = sum * scale;
                if (scores[t] > max_score) max_score = scores[t];
            }

            // --- 阶段 B: Softmax ---
            float exp_sum = 0.0f;
            for (size_t t = 0; t < total_len; ++t) {
                if (scores[t] <= -1e20f) {
                    scores[t] = 0.0f;
                } else {
                    scores[t] = std::exp(scores[t] - max_score);
                    exp_sum += scores[t];
                }
            }
            if (exp_sum > 0) {
                for (size_t t = 0; t < total_len; ++t) scores[t] /= exp_sum;
            }

            // --- 阶段 C: 计算 Scores * V (高精度版本) ---
            T* out_ptr = p_out + (s * nh + h) * d_head;
            
            // 核心修复：创建一个 float 类型的临时数组来存储当前头的累加结果
            std::vector<float> accumulator(d_head, 0.0f);

            for (size_t t = 0; t < total_len; ++t) {
                if (scores[t] <= 0.0f) continue;
                
                T* v_ptr = p_v + (t * nkvh + h_kv) * d_head;
                float alpha = scores[t];
                
                // 所有的加法都在 float 精度下完成，不进行低精度中间转换
                for (size_t d = 0; d < d_head; ++d) {
                    accumulator[d] += alpha * utils::cast<float>(v_ptr[d]);
                }
            }

            // 只有在最后，才将高精度的结果一次性写回内存
            for (size_t d = 0; d < d_head; ++d) {
                out_ptr[d] = utils::cast<T>(accumulator[d]);
            }
        }
    }
}
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    switch (q->dtype()) {
        case LLAISYS_DTYPE_F32:  self_attention_cpu_kernel<float>(attn_val, q, k, v, scale); break;
        case LLAISYS_DTYPE_F16:  self_attention_cpu_kernel<fp16_t>(attn_val, q, k, v, scale); break;
        case LLAISYS_DTYPE_BF16: self_attention_cpu_kernel<bf16_t>(attn_val, q, k, v, scale); break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}
} // namespace llaisys::ops
