#include "tensor/tensor.hpp"
#include "llaisys/models/qwen2.h"
#include "ops/embedding/op.hpp"
#include "ops/rms_norm/op.hpp"
#include "ops/linear/op.hpp"
#include "ops/rope/op.hpp"
#include "ops/self_attention/op.hpp"
#include "ops/add/op.hpp"
#include "ops/swiglu/op.hpp"

#include <cmath>
#include <iostream>

namespace llaisys {

// 辅助函数：安全获取权重
tensor_t get_w(const std::map<std::string, tensor_t>& weights, const std::string& name) {
    auto it = weights.find(name);
    if (it == weights.end()) {
        std::cerr << "FATAL: Missing weight: " << name << std::endl;
        exit(-1); // 发现缺失权重立即停止，防止 Segfault
    }
    return it->second;
}

tensor_t Qwen2Model::forward(tensor_t input_ids, int start_pos) {
    size_t seq_len = input_ids->shape()[0];
    auto embed_w = get_w(_weights, "model.embed_tokens.weight");
    llaisysDataType_t dtype = embed_w->dtype();
    size_t hidden_size = embed_w->shape()[1];
    
    // 1. Embedding
    auto x = Tensor::create({seq_len, hidden_size}, dtype);
    llaisys::ops::embedding(x, input_ids, embed_w);

    // 准备 RoPE 需要的 pos_ids Tensor (防止 Segfault)
    auto pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64);
    std::vector<int64_t> pos_data(seq_len);
    for (size_t i = 0; i < seq_len; ++i) pos_data[i] = start_pos + i;
    pos_ids->load(pos_data.data());

    int num_layers = 28; 

    for (int i = 0; i < num_layers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i);
        auto residual = x;
        
        // --- Attention Block ---
        auto norm_x = Tensor::create({seq_len, hidden_size}, dtype);
        llaisys::ops::rms_norm(norm_x, x, get_w(_weights, prefix + ".input_layernorm.weight"), 1e-6);
        
        size_t q_out_dim = get_w(_weights, prefix + ".self_attn.q_proj.weight")->shape()[0];
        size_t k_out_dim = get_w(_weights, prefix + ".self_attn.k_proj.weight")->shape()[0];
        size_t hd = 128; // Qwen2-1.5B 固定的 head_dim
        size_t nh = q_out_dim / hd;
        size_t nkvh = k_out_dim / hd;
        
        auto q = Tensor::create({seq_len, nh, hd}, dtype);
        auto k = Tensor::create({seq_len, nkvh, hd}, dtype);
        auto v = Tensor::create({seq_len, nkvh, hd}, dtype);
        
        llaisys::ops::linear(q, norm_x, get_w(_weights, prefix + ".self_attn.q_proj.weight"), nullptr);
        llaisys::ops::linear(k, norm_x, get_w(_weights, prefix + ".self_attn.k_proj.weight"), nullptr);
        llaisys::ops::linear(v, norm_x, get_w(_weights, prefix + ".self_attn.v_proj.weight"), nullptr);
        
        // RoPE (必须传入 pos_ids)
        llaisys::ops::rope(q, q, pos_ids, 10000.0f); 
        llaisys::ops::rope(k, k, pos_ids, 10000.0f);
        
        // TODO: 这里应实现 KV Cache 逻辑。目前是简化的 Full Attention。
        auto attn_out = Tensor::create({seq_len, nh, hd}, dtype);
        llaisys::ops::self_attention(attn_out, q, k, v, 1.0f / std::sqrt((float)hd));
        
        // 关键修复：Linear 要求 2D 输入，将 (seq, nh, hd) 转为 (seq, nh*hd)
        auto attn_out_2d = attn_out->view({seq_len, nh * hd});
        
        auto attn_proj = Tensor::create({seq_len, hidden_size}, dtype);
        llaisys::ops::linear(attn_proj, attn_out_2d, get_w(_weights, prefix + ".self_attn.o_proj.weight"), nullptr);
        
        x = Tensor::create({seq_len, hidden_size}, dtype);
        llaisys::ops::add(x, residual, attn_proj);

        // --- MLP Block ---
        residual = x;
        auto norm_x2 = Tensor::create({seq_len, hidden_size}, dtype);
        llaisys::ops::rms_norm(norm_x2, x, get_w(_weights, prefix + ".post_attention_layernorm.weight"), 1e-6);
        
        size_t inter_size = get_w(_weights, prefix + ".mlp.gate_proj.weight")->shape()[0];
        auto gate = Tensor::create({seq_len, inter_size}, dtype);
        auto up = Tensor::create({seq_len, inter_size}, dtype);
        llaisys::ops::linear(gate, norm_x2, get_w(_weights, prefix + ".mlp.gate_proj.weight"), nullptr);
        llaisys::ops::linear(up, norm_x2, get_w(_weights, prefix + ".mlp.up_proj.weight"), nullptr);
        
        auto mlp_out = Tensor::create({seq_len, inter_size}, dtype);
        llaisys::ops::swiglu(mlp_out, gate, up);
        
        auto mlp_proj = Tensor::create({seq_len, hidden_size}, dtype);
        llaisys::ops::linear(mlp_proj, mlp_out, get_w(_weights, prefix + ".mlp.down_proj.weight"), nullptr);
        
        x = Tensor::create({seq_len, hidden_size}, dtype);
        llaisys::ops::add(x, residual, mlp_proj);
    }

    auto final_norm = Tensor::create({seq_len, hidden_size}, dtype);
    llaisys::ops::rms_norm(final_norm, x, get_w(_weights, "model.norm.weight"), 1e-6);
    
    size_t vocab_size = get_w(_weights, "lm_head.weight")->shape()[0];
    auto logits = Tensor::create({seq_len, vocab_size}, dtype);
    llaisys::ops::linear(logits, final_norm, get_w(_weights, "lm_head.weight"), nullptr);
    
    return logits;
}

extern "C" {
    void* llaisys_qwen2_create() { 
        return new llaisys::Qwen2Model(); 
    }

    void llaisys_qwen2_load_weight(void* model, const char* name, void* t) {
        llaisys::tensor_t* weight_ptr = static_cast<llaisys::tensor_t*>(t);
        static_cast<llaisys::Qwen2Model*>(model)->load_weight(name, *weight_ptr);
    }

    void* llaisys_qwen2_infer(void* model, void* input_ids_ptr, int start_pos) {
        llaisys::tensor_t* ids_ptr = static_cast<llaisys::tensor_t*>(input_ids_ptr);
        llaisys::tensor_t res = static_cast<llaisys::Qwen2Model*>(model)->forward(*ids_ptr, start_pos);
        return new llaisys::tensor_t(res);
    }
}

} // namespace llaisys