from collections import defaultdict
from typing import TypeVar

import paddle
import paddle.distributed as dist

from tqdm.auto import tqdm

TensorOrStringType = TypeVar("TensorOrStringType", paddle.Tensor, list[str])


def maybe_to_tensor(values: list[TensorOrStringType]) -> TensorOrStringType:
    if isinstance(values[0], paddle.Tensor):
        return paddle.concat(x=values)
    return [el for x in values for el in x]


class SetPropertyScalers:
    """
    Utility callback; at the start of training, this computes the mean and std of the property data and adds the property
    scalers to the model.
    """

    @staticmethod
    def _compute_property_scalers(
        train_dataloader,
        property_embeddings: paddle.nn.LayerDict,
    ):
        property_values = defaultdict(list)
        property_names = [
            p.name
            for p in property_embeddings.values()
            if not isinstance(p.scaler, paddle.nn.Identity)
        ]
        if len(property_names) == 0:
            return
        for batch in tqdm(train_dataloader, desc=f"Fitting property scalers"):
            batch = batch["data"]
            for property_name in property_names:
                property_values[property_name].append(batch[property_name])
        for property_name in property_names:
            values = maybe_to_tensor(values=property_values[property_name])
            if dist.is_initialized():
                if isinstance(values, paddle.Tensor):
                    values_list = []
                    dist.all_gather(values_list, values)
                    values = paddle.concat(x=values_list)
                else:
                    print(f"Property {property_name} cannot be gathered")
            property_embeddings[property_name].fit_scaler(
                all_data=values
            )

    def on_fit_start(self, train_dataloader, model):
        model = model.model
        self._compute_property_scalers(
            train_dataloader=train_dataloader, property_embeddings=model.property_embeddings
        )
        if hasattr(model, "property_embeddings_adapt"):
            self._compute_property_scalers(
                train_dataloader=train_dataloader,
                property_embeddings=model.property_embeddings_adapt,
            )
