import paddle
from paddle.framework import core

EAGER_COMP_OP_BLACK_LIST = [
    "abs_grad",
    "cast_grad",
    "concat_grad",
    "cos_double_grad",
    "cos_grad",
    "cumprod_grad",
    "cumsum_grad",
    "dropout_grad",
    "erf_grad",
    "exp_grad",
    "expand_grad",
    "floor_grad",
    "gather_grad",
    "gather_nd_grad",
    "gelu_grad",
    "group_norm_grad",
    "instance_norm_grad",
    "layer_norm_grad",
    "leaky_relu_grad",
    "log_grad",
    "max_grad",
    "pad_grad",
    "pow_double_grad",
    "pow_grad",
    "prod_grad",
    "relu_grad",
    "roll_grad",
    "rsqrt_grad",
    "scatter_grad",
    "scatter_nd_add_grad",
    "sigmoid_grad",
    "silu_grad",
    "sin_double_grad",
    "sin_grad",
    "slice_grad",
    "split_grad",
    "sqrt_grad",
    "stack_grad",
    "sum_grad",
    "tanh_double_grad",
    "tanh_grad",
    "topk_grad",
    "transpose_grad",
    "add_double_grad",
    "add_grad",
    "assign_grad",
    "batch_norm_grad",
    "divide_grad",
    "elementwise_pow_grad",
    "maximum_grad",
    "min_grad",
    "minimum_grad",
    "multiply_grad",
    "subtract_grad",
    "tile_grad",
]
EAGER_COMP_OP_BLACK_LIST = list(set(EAGER_COMP_OP_BLACK_LIST))


def setting_eager_mode(enable=True, white_list=None):
    core.set_prim_eager_enabled(enable)
    if enable:
        new_black_list = EAGER_COMP_OP_BLACK_LIST
        if white_list is not None:
            assert isinstance(white_list, list)
            for op in white_list:
                if op in new_black_list:
                    new_black_list.remove(op)
        paddle.framework.core._set_prim_backward_blacklist(*new_black_list)
