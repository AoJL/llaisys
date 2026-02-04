from typing import Sequence, Tuple
import numpy as np
import ctypes

from .libllaisys import (
    LIB_LLAISYS,
    llaisysTensor_t,
    llaisysDeviceType_t,
    DeviceType,
    llaisysDataType_t,
    DataType,
)
from ctypes import c_size_t, c_int, c_ssize_t, c_void_p


class Tensor:
    def __init__(
        self,
        shape: Sequence[int] = None,
        dtype: DataType = DataType.F32,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
        tensor: llaisysTensor_t = None,
    ):
        if tensor:
            # 如果提供了底层 C++ 指针，直接包装
            self._tensor = tensor
        else:
            # 正常创建流程
            if shape is None:
                raise ValueError("shape must be provided to create a new tensor")
            _ndim = len(shape)
            _shape = (c_size_t * _ndim)(*shape)
            self._tensor: llaisysTensor_t = LIB_LLAISYS.tensorCreate(
                _shape,
                c_size_t(_ndim),
                llaisysDataType_t(dtype.value if hasattr(dtype, 'value') else dtype),
                llaisysDeviceType_t(device.value if hasattr(device, 'value') else device),
                c_int(device_id),
            )

    @staticmethod
    def from_handle(handle):
        """
        核心修复：将 C++ 接口返回的指针句柄包装为 Python Tensor 对象
        """
        if not handle:
            return None
        return Tensor(tensor=handle)

    def __del__(self):
        if hasattr(self, "_tensor") and self._tensor is not None:
            LIB_LLAISYS.tensorDestroy(self._tensor)
            self._tensor = None

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        核心修复：将 shape 变为属性，支持 logits_tensor.shape[0] 这种写法
        """
        n = self.ndim()
        if n == 0: return ()
        buf = (c_size_t * n)()
        LIB_LLAISYS.tensorGetShape(self._tensor, buf)
        return tuple(buf[i] for i in range(n))

    def strides(self) -> Tuple[int, ...]:
        n = self.ndim()
        if n == 0: return ()
        buf = (c_ssize_t * n)()
        LIB_LLAISYS.tensorGetStrides(self._tensor, buf)
        return tuple(buf[i] for i in range(n))

    def ndim(self) -> int:
        return int(LIB_LLAISYS.tensorGetNdim(self._tensor))

    @property
    def dtype(self) -> DataType:
        return DataType(LIB_LLAISYS.tensorGetDataType(self._tensor))

    def device_type(self) -> DeviceType:
        return DeviceType(LIB_LLAISYS.tensorGetDeviceType(self._tensor))

    def device_id(self) -> int:
        return int(LIB_LLAISYS.tensorGetDeviceId(self._tensor))

    def data_ptr(self) -> c_void_p:
        return LIB_LLAISYS.tensorGetData(self._tensor)

    def lib_tensor(self) -> llaisysTensor_t:
        return self._tensor

    def debug(self):
        LIB_LLAISYS.tensorDebug(self._tensor)

    def __repr__(self):
        return f"<Tensor shape={self.shape}, dtype={self.dtype}, device={self.device_type()}:{self.device_id()}>"

    def load(self, data):
        """
        支持传入 numpy 数组的 .ctypes.data 地址
        """
        if isinstance(data, (int, c_void_p)):
            LIB_LLAISYS.tensorLoad(self._tensor, data)
        else:
            # 如果直接传了 numpy 数组，自动处理
            LIB_LLAISYS.tensorLoad(self._tensor, data.ctypes.data)

    def to_numpy(self):
        """
        核心修复：支持 qwen2.py 采样时将结果转回 numpy 进行处理
        """
        # 简单映射类型
        dtype_map = {
            DataType.F32: np.float32,
            DataType.I64: np.int64,
            DataType.F16: np.float16,
            # BF16 需要安装 ml_dtypes
        }
        
        # 获取总大小
        numel = 1
        for s in self.shape:
            numel *= s
        
        # 获取原始指针并转为 numpy 数组
        ptr = self.data_ptr()
        np_dtype = dtype_map.get(self.dtype, np.float32)
        
        # 创建一个不拷贝数据的视图
        array = np.ctypeslib.as_array(
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), # 这里视具体情况可能需要根据 dtype cast
            shape=self.shape
        )
        return array.copy() # 返回副本保证安全

    def is_contiguous(self) -> bool:
        return bool(LIB_LLAISYS.tensorIsContiguous(self._tensor))

    def view(self, *shape: int):
        _shape = (c_size_t * len(shape))(*shape)
        return Tensor(
            tensor=LIB_LLAISYS.tensorView(self._tensor, _shape, c_size_t(len(shape)))
        )

    def permute(self, *perm: int):
        assert len(perm) == self.ndim()
        _perm = (c_size_t * len(perm))(*perm)
        return Tensor(tensor=LIB_LLAISYS.tensorPermute(self._tensor, _perm))

    def slice(self, dim: int, start: int, end: int):
        return Tensor(
            tensor=LIB_LLAISYS.tensorSlice(
                self._tensor, c_size_t(dim), c_size_t(start), c_size_t(end)
            )
        )