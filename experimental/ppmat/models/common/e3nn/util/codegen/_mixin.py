# from opt_einsum_fx import jitable


class CodeGenMixin:

    def _codegen_register(self, funcs):
        if not hasattr(self, "__codegen__"):
            self.__codegen__ = []
        self.__codegen__.extend(funcs.keys())
        for fname, func in funcs.items():
            setattr(self, fname, func)

    def __getstate__(self):
        if hasattr(super(CodeGenMixin, self), "__getstate__"):
            out = super(CodeGenMixin, self).__getstate__()
        else:
            out = self.__dict__
        return out

    def __setstate__(self, d):
        if hasattr(super(CodeGenMixin, self), "__setstate__"):
            super(CodeGenMixin, self).__setstate__(d)
        else:
            self.__dict__.update(d)
