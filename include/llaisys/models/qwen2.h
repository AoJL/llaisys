#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>

// 核心：不要包含 include/llaisys/tensor.h，因为它只是 C 接口
// 我们需要包含你在 Assignment 1 实现的真正的 Tensor 类头文件
// 假设它的路径是下面这个（如果不对，请根据你的文件树调整）
#include "tensor/tensor.hpp" 

namespace llaisys {

// 使用我们在 Assignment 1 中定义的别名
// 如果 tensor.hpp 里已经定义了，这里可以不写，或者写成：
// using tensor_t = std::shared_ptr<Tensor>;

class Qwen2Model {
public:
    Qwen2Model() = default;
    ~Qwen2Model() = default;

    void load_weight(const std::string& name, tensor_t weight) {
        _weights[name] = weight;
    }

    tensor_t forward(tensor_t input_ids, int start_pos);

private:
    std::map<std::string, tensor_t> _weights;
    // 用于 KV Cache
    std::map<int, tensor_t> _k_cache; 
    std::map<int, tensor_t> _v_cache;
};

// C 接口：供 python/llaisys/models/qwen2.py 调用
extern "C" {
    void* llaisys_qwen2_create();
    void llaisys_qwen2_load_weight(void* model, const char* name, void* tensor_ptr);
    void* llaisys_qwen2_infer(void* model, void* input_ids_ptr, int start_pos);
}

} // namespace llaisys