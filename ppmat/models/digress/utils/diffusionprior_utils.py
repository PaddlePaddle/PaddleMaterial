from paddle import nn


# helper functions
def exists(val):
    return val is not None


def l2norm(t):
    return nn.functional.normalize(x=t, axis=-1)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]
