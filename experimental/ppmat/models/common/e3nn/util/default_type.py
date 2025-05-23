from typing import Optional
from typing import Tuple
from typing import Union

import paddle


def torch_get_default_tensor_type():
    return str(paddle.empty(shape=[0]).dtype)


def _torch_get_default_dtype() -> paddle.dtype:
    """A torchscript-compatible version of torch.get_default_dtype()"""
    return paddle.empty(shape=[0]).dtype


def torch_get_default_device() -> Union[paddle.CPUPlace, paddle.CUDAPlace]:
    return paddle.empty(shape=[0]).place


def explicit_default_types(
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Union[paddle.CPUPlace, paddle.CUDAPlace]] = None,
) -> Tuple[paddle.dtype, Union[paddle.CPUPlace, paddle.CUDAPlace]]:
    """A torchscript-compatible type resolver"""
    if dtype is None:
        dtype = _torch_get_default_dtype()
    if device is None:
        device = torch_get_default_device()
    return dtype, device
