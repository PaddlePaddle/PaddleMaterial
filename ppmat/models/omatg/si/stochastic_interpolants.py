# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stochastic Interpolants main class.
"""

from enum import Enum
from typing import Callable, List, Sequence, Tuple, Union

import paddle

from ppmat.models.omatg.datamodule.omatg_data import OMATGData

# Global constants
SMALL_TIME: float = 1.0e-3
BIG_TIME: float = 1.0 - SMALL_TIME


class DataField(Enum):
    """Enum for data fields in OMATGData relevant for stochastic interpolants."""

    pos = "pos"
    cell = "cell"
    species = "species"


def reshape_t(
    t: paddle.Tensor, n_atoms: paddle.Tensor, data_field: DataField
) -> paddle.Tensor:
    """Reshape times tensor for batch configurations for the given data field."""
    assert len(t.shape) == 1
    assert len(n_atoms.shape) == 1

    # Repeat t for each atom
    t_per_atom = paddle.repeat_interleave(t, n_atoms.cast("int64"))
    sum_n_atoms = int(n_atoms.sum().item())
    batch_size = len(t)

    if data_field == DataField.pos:
        # Reshape to (sum_n_atoms, 3) by repeating 3 times
        return paddle.reshape(
            paddle.repeat_interleave(t_per_atom, 3),
            [sum_n_atoms, 3],
        )
    elif data_field == DataField.cell:
        # Reshape to (batch_size, 3, 3)
        return paddle.reshape(
            paddle.repeat_interleave(t, 9), [batch_size, 3, 3]
        )
    else:
        # species case
        return t_per_atom


class StochasticInterpolants:
    """
    Collection of several stochastic interpolants between points x_0 and x_1 from two distributions
    p_0 and p_1 at times t for different coordinate types x (like atom species, fractional coordinates,
    and lattice vectors).

    Every stochastic interpolant is associated with a data field and a cost factor.

    :param stochastic_interpolants:
        Sequence of stochastic interpolants for the different coordinate types.
    :param data_fields:
        Sequence of data fields for the different stochastic interpolants.
    :param integration_time_steps:
        Number of integration time steps for the integration.

    :raises ValueError:
        If the number of stochastic interpolants and costs are not equal.
        If the number of stochastic interpolants and data fields are not equal.
        If the number of integration time steps is not positive.
    """

    def __init__(
        self,
        stochastic_interpolants: Sequence,
        data_fields: Sequence[str],
        integration_time_steps: int,
        enable_progress_bar: bool = True,
    ) -> None:
        """Constructor of the StochasticInterpolants class."""
        super().__init__()

        if not len(stochastic_interpolants) == len(data_fields):
            raise ValueError(
                "The number of stochastic interpolants and data fields must be equal."
            )

        try:
            self._data_fields = [DataField[df.lower()] for df in data_fields]
        except AttributeError:
            raise ValueError(
                f"All data fields must be in {[d.value for d in DataField]}."
            )

        if not integration_time_steps > 0:
            raise ValueError("The number of integration time steps must be positive.")

        self._stochastic_interpolants = stochastic_interpolants
        self._integration_time_steps = integration_time_steps
        self._enable_progress_bar = enable_progress_bar

    def __len__(self) -> int:
        """
        Return the number of stochastic interpolants handled by this class.

        :return:
            Number of stochastic interpolants.
        """
        return len(self._stochastic_interpolants)

    def loss_keys(self) -> List[str]:
        """
        Return the keys of the losses returned by this class.

        :return:
            Keys of the losses.
        """
        loss_keys = []
        for df, si in zip(self._data_fields, self._stochastic_interpolants):
            for key in si.loss_keys():
                full_key = f"{df.value}_{key}"
                if full_key in loss_keys:
                    raise ValueError(f"Key {full_key} is already used as a loss key.")
                loss_keys.append(full_key)
        return loss_keys

    def _interpolate(
        self,
        t: paddle.Tensor,
        x_0: OMATGData,
        x_1: OMATGData,
    ) -> Tuple[OMATGData, OMATGData]:
        """
        Stochastically interpolate between the collection of points x_0 and x_1 from the
        collection of two distributions p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :param x_0:
            Collection of points from the collection of distributions p_0.
        :param x_1:
            Collection of points from the collection of distributions p_1.

        :return:
            Collection of stochastically interpolated points x_t, and the collection of z values.
        :rtype: tuple[OMATGData, OMATGData]
        """
        assert paddle.equal_all(x_0.batch, x_1.batch).item()
        assert paddle.equal_all(x_0.n_atoms, x_1.n_atoms).item()

        n_atoms = x_0.n_atoms
        x_t = x_0.clone()
        z_data = {}

        for stochastic_interpolant, data_field in zip(
            self._stochastic_interpolants, self._data_fields
        ):
            field_name = data_field.value
            reshaped_t = reshape_t(t, n_atoms, data_field)

            # Cell data requires different batch indices.
            if data_field == DataField.cell:
                batch_indices = paddle.arange(len(x_0.n_atoms))
            else:
                batch_indices = x_0.batch

            interpolated_x_t, z = stochastic_interpolant.interpolate(
                reshaped_t,
                getattr(x_0, field_name),
                getattr(x_1, field_name),
                batch_indices,
            )

            # Update x_t (using internal method)
            x_t.set_field(field_name, interpolated_x_t)
            z_data[field_name] = z

        return x_t, OMATGData._from_dict(z_data)

    def losses(
        self,
        model_function: Callable[[OMATGData, paddle.Tensor], OMATGData],
        t: paddle.Tensor,
        x_0: OMATGData,
        x_1: OMATGData,
    ) -> dict[str, paddle.Tensor]:
        """
        Compute the losses for the collection of stochastic interpolants.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta.
        :param t:
            Times in [0,1].
        :param x_0:
            Collection of points from the distribution p_0.
        :param x_1:
            Collection of points from the distribution p_1.

        :return:
            The losses for the collection of stochastic interpolants.
        :rtype: dict[str, paddle.Tensor]
        """
        # Interpolate everything first
        x_t, z = self._interpolate(t, x_0, x_1)

        assert paddle.equal_all(x_0.batch, x_1.batch).item()
        assert paddle.equal_all(x_0.n_atoms, x_1.n_atoms).item()
        n_atoms = x_0.n_atoms

        losses = {}
        for stochastic_interpolant, data_field in zip(
            self._stochastic_interpolants, self._data_fields
        ):
            field_name = data_field.value
            b_data_field = field_name + "_b"
            eta_data_field = field_name + "_eta"

            reshaped_t = reshape_t(t, n_atoms, data_field)

            # Cell data requires different batch indices.
            if data_field == DataField.cell:
                batch_indices = paddle.arange(len(x_0.n_atoms))
            else:
                batch_indices = x_0.batch

            def model_prediction_fn(x):
                # Create a copy of x_t and update the field
                x_t_clone = x_t.clone()
                x_t_clone.set_field(field_name, x)
                model_result = model_function(x_t_clone, t)
                return model_result[b_data_field], model_result[eta_data_field]

            l = stochastic_interpolant.loss(
                model_prediction_fn,
                reshaped_t,
                getattr(x_0, field_name),
                getattr(x_1, field_name),
                getattr(x_t, field_name),
                z.get_field(field_name),
                batch_indices,
            )

            for l_key, l_value in l.items():
                assert l_key not in losses
                losses[f"{field_name}_{l_key}"] = l_value

        return losses

    def integrate(
        self,
        x_0: OMATGData,
        model_function: Callable[[OMATGData, paddle.Tensor], OMATGData],
        save_intermediate: bool = False,
    ) -> Union[OMATGData, Tuple[OMATGData, List[OMATGData]]]:
        """
        Integrate the collection of points x_0 from time 0 to 1 based on the model
        that provides the velocity fields b and denoisers eta.

        :param x_0:
            Collection of points from the distribution p_0.
        :param model_function:
            Model function returning the velocity fields b and the denoisers eta.
        :param save_intermediate:
            If True, the intermediate points of the integration are saved and returned.

        :return:
            Collection of integrated points x_1.
            If save_intermediate is True, also returns a list of the intermediate points.
        """
        times = paddle.linspace(SMALL_TIME, BIG_TIME, self._integration_time_steps)

        x_t = x_0.clone()
        new_x_t = x_0.clone()

        if save_intermediate:
            inter_list = [x_t]
        else:
            inter_list = None

        with paddle.no_grad():
            for t_index in range(1, len(times)):
                t = times[t_index - 1]
                dt = times[t_index] - times[t_index - 1]

                for stochastic_interpolant, data_field in zip(
                    self._stochastic_interpolants, self._data_fields
                ):
                    field_name = data_field.value
                    b_data_field = field_name + "_b"
                    eta_data_field = field_name + "_eta"

                    def model_prediction_fn(time, x):
                        # Repeat time for each element in the batch
                        t_val = time.item() if time.numel() == 1 else float(time[0])
                        time = paddle.full_like(
                            x_t.n_atoms.cast("float32"),
                            t_val,
                        )
                        x_int = x_t.clone()
                        x_int.set_field(field_name, x)
                        model_result = model_function(x_int, time)
                        return model_result[b_data_field], model_result[eta_data_field]

                    # Cell data requires different batch indices.
                    if data_field == DataField.cell:
                        batch_indices = paddle.arange(len(x_0.n_atoms))
                    else:
                        batch_indices = x_0.batch

                    new_value = stochastic_interpolant.integrate(
                        model_prediction_fn,
                        x_t.get_field(field_name),
                        t,
                        dt,
                        batch_indices,
                    )
                    new_x_t.set_field(field_name, new_value)

                x_t = new_x_t.clone()

                if save_intermediate:
                    inter_list.append(x_t)

        if save_intermediate:
            return x_t, inter_list
        else:
            return x_t

    def get_stochastic_interpolant(self, data_field: str):
        """
        Return the stochastic interpolant associated with the data field.

        :param data_field:
            Data field for which the stochastic interpolant is requested.

        :return:
            Stochastic interpolant associated with the data field.
        """
        try:
            df = DataField[data_field.lower()]
        except AttributeError:
            raise ValueError(f"Data field must be in {[d.value for d in DataField]}.")

        index = self._data_fields.index(df)
        return self._stochastic_interpolants[index]
