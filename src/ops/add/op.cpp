#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <iostream>

namespace llaisys::ops {

// --- 第一部分：内部内核实现 (CPU Kernel) ---
namespace cpu {
    template <typename T>
    void add_kernel(void* c, const void* a, const void* b, size_t n) {
        T* pc = static_cast<T*>(c);
        const T* pa = static_cast<const T*>(a);
        const T* pb = static_cast<const T*>(b);

        for (size_t i = 0; i < n; ++i) {
            float va = utils::cast<float>(pa[i]);
            float vb = utils::cast<float>(pb[i]);
            pc[i] = utils::cast<T>(va + vb);
        }
    }

    void add(void* c, const void* a, const void* b, llaisysDataType_t dtype, size_t n) {
        switch (dtype) {
            case LLAISYS_DTYPE_F32:  add_kernel<float>(c, a, b, n); break;
            case LLAISYS_DTYPE_F16:  add_kernel<fp16_t>(c, a, b, n); break;
            case LLAISYS_DTYPE_BF16: add_kernel<bf16_t>(c, a, b, n); break;
            default: break;
        }
    }
} // namespace cpu

// --- 第二部分：外部包装接口 (这是链接器在找的符号) ---
void add(tensor_t c, tensor_t a, tensor_t b) {
    // 1. 安全检查（可选，但建议保留）
    if (a->numel() != b->numel() || a->numel() != c->numel()) {
        std::cerr << "Add size mismatch!" << std::endl;
        return;
    }

    // 2. 调用上面的 CPU 实现
    if (c->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    } else {
        // 如果有 NVIDIA GPU 实现可以在这里补充
        // TO_BE_IMPLEMENTED();
    }
}

} // namespace llaisys::ops