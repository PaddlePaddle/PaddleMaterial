from __future__ import annotations

from functools import wraps


def conditional_grad(dec):
    """Enable gradients only when model is configured for autograd forces."""

    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if getattr(self, "regress_forces", False) and not getattr(
                self, "direct_forces", False
            ):
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator
