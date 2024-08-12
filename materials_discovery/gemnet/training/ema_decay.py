"""
Copied from:
https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py
"""

from __future__ import division
from __future__ import unicode_literals

import copy
import weakref

import paddle


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from
            `model.parameters()`).
        decay: The exponential decay.
        use_num_updates: Whether to use number of updates when computing
            averages.
    """

    def __init__(self, parameters, decay: float, use_num_updates: bool = False):
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        parameters = list(parameters)
        self.shadow_params = [
            p.clone().detach() for p in parameters if not p.stop_gradient
        ]
        self.collected_params = []
        self._params_refs = [weakref.ref(p) for p in parameters]

    def _get_parameters(self, parameters=None):
        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError(
                    "(One of) the parameters with which this ExponentialMovingAverage was initialized no longer exists (was garbage collected); please either provide `parameters` explicitly or keep the model to which they belong from being garbage collected."
                )
            return parameters
        else:
            return parameters

    def update(self, parameters=None) -> None:
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object. If `None`, the
            parameters with which this `ExponentialMovingAverage` was
            initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with paddle.no_grad():
            parameters = [p for p in parameters if not p.stop_gradient]
            for s_param, param in zip(self.shadow_params, parameters):
                tmp = s_param - param
                tmp.multiply_(y=paddle.to_tensor(one_minus_decay))
                s_param.subtract_(y=paddle.to_tensor(tmp))

    def copy_to(self, parameters=None) -> None:
        """
        Copy current parameters into given collection of parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages. If `None`, the
            parameters with which this `ExponentialMovingAverage` was
            initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            if not param.stop_gradient:
                param.data.copy_(s_param.data)

    def store(self, parameters=None) -> None:
        """
        Save the current parameters for restoring later.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored. If `None`, the parameters of with which this
            `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.collected_params = [
            param.clone() for param in parameters if not param.stop_gradient
        ]

    def restore(self, parameters=None) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters. If `None`, the
            parameters with which this `ExponentialMovingAverage` was
            initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            if not param.stop_gradient:
                param.data.copy_(c_param.data)

    def state_dict(self) -> dict:
        """Returns the state of the ExponentialMovingAverage as a dict."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the ExponentialMovingAverage state.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(
            self.num_updates, int
        ), "Invalid num_updates"
        self.shadow_params = state_dict["shadow_params"]
        assert isinstance(self.shadow_params, list), "shadow_params must be a list"
        assert all(
            isinstance(p, paddle.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"
        self.collected_params = state_dict["collected_params"]
        assert isinstance(
            self.collected_params, list
        ), "collected_params must be a list"
        assert all(
            isinstance(p, paddle.Tensor) for p in self.collected_params
        ), "collected_params must all be Tensors"
