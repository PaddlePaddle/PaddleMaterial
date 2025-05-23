import inspect

import paddle

_E3NN_COMPILE_MODE = "__e3nn_compile_mode__"
_VALID_MODES = "trace", "script", "unsupported", None


def compile_mode(mode: str):
    if mode not in _VALID_MODES:
        raise ValueError("Invalid compile mode")

    def decorator(obj):
        if not (inspect.isclass(obj) and issubclass(obj, paddle.nn.Layer)):
            raise TypeError(
                "@e3nn.util.jit.compile_mode can only decorate classes derived from paddle.nn.Layer"
            )
        setattr(obj, _E3NN_COMPILE_MODE, mode)
        return obj

    return decorator


def get_compile_mode(mod: paddle.nn.Layer) -> str:
    if hasattr(mod, _E3NN_COMPILE_MODE):
        mode = getattr(mod, _E3NN_COMPILE_MODE)
    else:
        mode = getattr(type(mod), _E3NN_COMPILE_MODE, None)
    assert mode in _VALID_MODES, "Invalid compile mode `%r`" % mode
    return mode


def compile(mod: paddle.nn.Layer, **kwargs):
    return mod


def script(mod: paddle.nn.Layer, **kwargs):
    return mod


def trace(mod: paddle.nn.Layer, **kwargs):
    return mod
