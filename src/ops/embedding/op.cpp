#include "op.hpp"
#include <cstring>  

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    const int64_t* p_idx = reinterpret_cast<const int64_t*>(index->data());
    size_t num_indices = index->numel();
    size_t hidden_size = weight->shape()[1];
    size_t esize = weight->elementSize();
    for (size_t i = 0; i < num_indices; ++i) {
        void* src = (char*)weight->data() + p_idx[i] * hidden_size * esize;
        void* dst = (char*)out->data() + i * hidden_size * esize;
        std::memcpy(dst, src, hidden_size * esize);
    }
}
} // namespace llaisys::ops
