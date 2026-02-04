from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DeviceType, DataType 
from ..tensor import Tensor
from ..ops import Ops 
import numpy as np
from pathlib import Path
import safetensors
import ml_dtypes 
import torch 

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.handle = LIB_LLAISYS.llaisys_qwen2_create()
        self.device = device
        model_path = Path(model_path)

        print("Loading weights into C++ backend...")
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="numpy", device="cpu") as data_:
                for name_ in data_.keys():
                    weight_data = data_.get_tensor(name_)
                    weight_data = np.ascontiguousarray(weight_data)
                    
                    if weight_data.dtype == np.float32:
                        ldtype = DataType.F32
                    elif weight_data.dtype == ml_dtypes.bfloat16:
                        ldtype = DataType.BF16
                    elif weight_data.dtype == np.int64:
                        ldtype = DataType.I64
                    else:
                        ldtype = DataType.F32
                    
                    ts = Tensor(shape=list(weight_data.shape), dtype=ldtype, device=self.device)
                    ts.load(weight_data.ctypes.data)
                    
                    LIB_LLAISYS.llaisys_qwen2_load_weight(
                        self.handle, 
                        name_.encode('utf-8'), 
                        ts._tensor 
                    )
        print("Model loaded successfully.")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 20,
        **kwargs
    ):
        generated_tokens = list(inputs)
        
        # --- 核心修复 1: 定义循环上限变量 ---
        max_tokens = max_new_tokens if max_new_tokens is not None else 20

        for i in range(max_tokens):
            # --- 核心修复 2: 全量模式 (无 KV Cache 时的正确做法) ---
            # 每次都将 generated_tokens 中所有的 token 转换为 numpy
            current_ids = np.array(generated_tokens, dtype=np.int64)
            current_ids = np.ascontiguousarray(current_ids)
            
            # 创建输入 Tensor
            input_tensor = Tensor(shape=list(current_ids.shape), dtype=DataType.I64, device=self.device)
            input_tensor.load(current_ids.ctypes.data)
            
            # 调用 C++ 推理接口。注意：start_pos 固定传 0
            # 这样 C++ 内部会从头计算整个序列的 Attention
            logits_handle = LIB_LLAISYS.llaisys_qwen2_infer(self.handle, input_tensor._tensor, 0)
            
            # 将 C++ 返回的指针包装成 Python Tensor
            logits_tensor = Tensor.from_handle(logits_handle)
            
            # 3. 采样逻辑
            # logits_tensor 的形状是 (seq_len, vocab_size)
            # 我们只需要预测序列中“最后一个”词的概率分布
            seq_len = logits_tensor.shape[0]
            last_token_logits = logits_tensor.slice(0, seq_len - 1, seq_len)
            
            # 准备 Argmax 容器
            max_idx = Tensor(shape=[1], dtype=DataType.I64, device=self.device)
            max_val = Tensor(shape=[1], dtype=logits_tensor.dtype, device=self.device)
            
            # 执行采样
            Ops.argmax(max_idx, max_val, last_token_logits)
            
            # 提取结果
            next_token = int(max_idx.to_numpy()[0])
            generated_tokens.append(next_token)
            
            # 实时打印，这能让你看到模型正在“说话”
            print(f"Step {i:02d}: token {next_token}") 
            
            # Qwen2 停止符判定 (EOS)
            if next_token in [151643, 151645]:
                break
                
        return generated_tokens