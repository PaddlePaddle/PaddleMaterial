import sys


from typing import Dict, Sequence, Union

import paddle
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.types import PropertySourceId, TargetProperty
from mattergen.common.utils.data_utils import get_atomic_number
from mattergen.common.utils.globals import MAX_ATOMIC_NUM, PROPERTY_SOURCE_IDS
from paddle_utils import *

_USE_UNCONDITIONAL_EMBEDDING = "_USE_UNCONDITIONAL_EMBEDDING"


def replace_use_unconditional_embedding(
    batch: ChemGraph, use_unconditional_embedding: Dict[PropertySourceId, paddle.Tensor]
) -> ChemGraph:
    """
    Set the use of conditional or unconditional embeddings for each conditional field in the batch.
    This utility will overwrite any batch._USE_CONDITIONAL_EMBEDDING keys included in use_unconditional_embedding
    but will keep the value of any keys in batch._USE_CONDITIONAL_EMBEDDING that are not in
    use_unconditional_embedding.

    Keyword arguments
    -----------------
    batch: ChemGraph -- the batch of data to be modified.
    use_unconditional_embedding: Dict[PropertyName, torch.BoolTensor] -- a dictionary whose values
        are torch.BoolTensors of shape (n_structures_in_batch, 1) stating whether to use the unconditional embedding for
        each conditional field. The keys are the names of the conditional fields in the batch.


    Returns
    -------
    ChemGraph -- the modified batch of data containing
        ChemGraph._USE_CONDITIONAL_EMBEDDING: Dict[PropertyName, torch.BoolTensor]. When
        ChemGraph[_USE_UNCONDITIONAL_EMBEDDING][cond_field][ii] is True, the iith data point will
        use its unconditional embedding for cond_field. When False, the conditional embedding will be used.
    """
    try:
        existing_use_unconditional_embedding = batch[_USE_UNCONDITIONAL_EMBEDDING]
        for k, v in use_unconditional_embedding.items():
            existing_use_unconditional_embedding[k] = v
        return batch.replace(
            **{_USE_UNCONDITIONAL_EMBEDDING: existing_use_unconditional_embedding}
        )
    except KeyError:
        return batch.replace(
            **{_USE_UNCONDITIONAL_EMBEDDING: use_unconditional_embedding}
        )


def get_use_unconditional_embedding(
    batch: ChemGraph, cond_field: PropertySourceId
) -> paddle.bool:
    """
    Returns
    -------
    torch.BoolTensor, shape=(n_structures_in_batch, 1) -- whether to use the unconditional embedding for cond_field.
        When True, we use unconditional embedding.

    NOTE: When _USE_UNCONDITIONAL_EMBEDDING is not in ChemGraph or cond_field is not
        in ChemGraph[_USE_UNCONDITIONAL_EMBEDDING] we return a torch.BoolTensor with False
        values. This allows a model trained conditional data to evaluate an unconditional score
        without having to specify any conditional data in ChemGraph.
    """
    try:
        return batch[_USE_UNCONDITIONAL_EMBEDDING][cond_field]
    except KeyError:
        return paddle.ones_like(x=batch["num_atoms"], dtype="bool").reshape(-1, 1)


def tensor_is_not_nan(x: paddle.Tensor) -> paddle.bool:
    """
    Keyword arguments
    -----------------
    x: torch.Tensor, shape = (n_structures_in_batch, Ndim) -- labels for a single conditional field.
        We assume that when a label is not present, the corresponding value is specified
        as torch.nan.

    Returns
    -------
    torch.BoolTensor, shape = (n_structures_in_batch,) -- index i is True if x[i] contains no NaNs
    """
    return paddle.all(
        x=paddle.reshape(
            x=paddle.logical_not(x=paddle.isnan(x=x)), shape=(tuple(x.shape)[0], -1)
        ),
        axis=1,
    )


def data_is_not_nan(
    x: Union[paddle.Tensor, list[str | None], list[list[str] | None]]
) -> paddle.bool:
    """
    Returns (n_structures_in_batch,) torch.BoolTensor of whether the conditional values
    for a given property are not nan.

    NOTE: Currently we enforce no restriction on the data type that properties can have in
    ChemGraph. The intent is that ChemGraph always contains property values in their
    representation and type seen by the user. This means however that we have to distribute
    handling of different types throughout the code, this function is one such place.

    """
    if isinstance(x, paddle.Tensor):
        return tensor_is_not_nan(x=x)
    else:
        return paddle.to_tensor(data=[(_x is not None) for _x in x])


def get_cond_field_names_in_batch(x: ChemGraph) -> list[str]:
    """
    Returns a list of field names that are known to be conditional properties in
    PROPERTY_SOURCE_IDS, which are present in x.
    """
    return [str(k) for k in x.keys() if k in PROPERTY_SOURCE_IDS]


class SetEmbeddingType:
    def __init__(self, p_unconditional: float, dropout_fields_iid: bool = False):
        """
        In PropertyEmbedding.forward we choose to concatenate either an unconditional embedding
        (ignores the value of a property) or a conditional embedding (depends on the value of a property)
        to the tensor that is input to the first node layer of each atom. This utility sets the internal state
        of ChemGraph to randomly select either the conditional or unconditional embedding for each structure
        in the batch.

        ChemGraph.[_USE_UNCONDITIONAL_EMBEDDING]: boolTensor, shape=(n_structures_in_batch, 1) stores a True
        value for structures where we intend to use the unconditional embedding for all atoms contained in
        that corresponding structure.

        This utility operates in 2 modes:
        1) dropout_fields_iid = True -- We randomly assign which conditional fields are unconditional and which
            are conditional for fields that are not nan independently of whether all conditional fields are not
            nan for that structure. This means that for a structure conditioned on (y1,y2) we can generate embeddings
            corresponding to p(x), p(x|y1), p(x|y2), p(x|y1,y2).
        2) dropout_fields_iid = False - We assign conditional or unconditional embeddings to all conditional fields
            of a single structure simultaneously. This means that for a structure conditioned on (y1,y2) we can
            only generate embeddings corresponding to p(x) and p(|y1,y2).

        Keyword args:
        -------------
        p_unconditional: float -- the probability of using the unconditional embedding in the score model.
        dropout_fields_iid: bool -- whether to mask the conditional embedding of fields independently and
            identically distributed according to p_unconditional. If False, the score model is only exposed
            to two scenarios: 1) all conditional fields have their unconditional embedding. 2) all conditional
            fields have their conditional embedding. If True, the score model is exposed to all possible
            combinations of conditional fields having their unconditional or conditional embeddings, ie the score
            model will learn p(x), p(x|y1), p(x_y2), p(x|y1,y2),...

            Note: when dropout_fields_iid=False, the conditional embedding will only be used when all
            conditional fields have data present. If no single data point has data present for all conditional
            fields, then the score model will only be exposed to the unconditional embedding state p(x) and the
            joint p(x|y1,y2,...) will not be learned.
        """
        self.p_unconditional = p_unconditional
        self.dropout_fields_iid = dropout_fields_iid

    def __call__(self, x: ChemGraph) -> ChemGraph:
        cond_fields: list[str] = get_cond_field_names_in_batch(x=x)
        if len(cond_fields) == 0:
            return x
        else:
            batch_size = len(x[cond_fields[0]])
            device = x["num_atoms"].place
            data_is_not_nan_dict: Dict[PropertySourceId, paddle.Tensor] = {
                cond_field: data_is_not_nan(x=x[cond_field]).to(device=device)
                for cond_field in cond_fields
            }
            alldata_is_not_nan: paddle.bool = paddle.all(
                x=paddle.concat(
                    x=[
                        cond_data_not_nan.reshape(-1, 1)
                        for cond_data_not_nan in data_is_not_nan_dict.values()
                    ],
                    axis=1,
                ),
                axis=1,
            )
            use_unconditional_embedding: Dict[PropertySourceId, paddle.Tensor] = {}
            for cond_field in cond_fields:
                embedding_type = paddle.ones(shape=(batch_size, 1), dtype="bool")
                if self.dropout_fields_iid:
                    cond_data_is_not_nan = data_is_not_nan_dict[cond_field]
                else:
                    cond_data_is_not_nan = alldata_is_not_nan
                embedding_type[cond_data_is_not_nan] = (
                    paddle.rand(shape=(cond_data_is_not_nan.sum(), 1))
                    <= self.p_unconditional
                )
                use_unconditional_embedding[cond_field] = embedding_type
            return replace_use_unconditional_embedding(
                batch=x, use_unconditional_embedding=use_unconditional_embedding
            )


class SetUnconditionalEmbeddingType:
    """
    In PropertyEmbedding.forward we choose to concatenate either an unconditional embedding
    (ignores the value of a property) or a conditional embedding (depends on the value of a property)
    to the tensor that is input to the first node layer of each atom. This utility sets the internal state
    of ChemGraph to use the unconditional embedding for all structures for all conditional fields present
    in the batch. Note that conditional fields in the batch are automatically determined by the presence
    of any PropertyName in ChemGraph.

    ChemGraph.[_USE_UNCONDITIONAL_EMBEDDING]: boolTensor, shape=(n_structures_in_batch, 1) stores True
    for all structures for all conditional properties present in ChemGraph.

    NOTE: If a conditional property was trained on by the model but is not
    specified in the batch, then it will be attributed an unconditional embedding
    in mattergen.property_embeddings.PropertyEmbedding.forward.
    This behaviour allows unconditional samples to be drawn from a model that was trained
    on certain conditions, without having to set any conditional values in ChemGraph.
    """

    def __call__(self, x: ChemGraph) -> ChemGraph:
        cond_fields = get_cond_field_names_in_batch(x=x)
        device = x["num_atoms"].place
        return replace_use_unconditional_embedding(
            batch=x,
            use_unconditional_embedding={
                cond_field: paddle.ones(shape=(len(x[cond_field]), 1), dtype="bool")
                for cond_field in cond_fields
            },
        )


class SetConditionalEmbeddingType:
    """
    In PropertyEmbedding.forward we choose to concatenate either an unconditional embedding
    (ignores the value of a property) or a conditional embedding (depends on the value of a property)
    to the tensor that is input to the first node layer of each atom. This utility sets the internal state
    of ChemGraph to use the unconditional embedding for all structures for all conditional fields present
    in the batch. Note that conditional fields in the batch are automatically determined by the presence
    of any PropertyName on in ChemGraph.

    ChemGraph.[_USE_UNCONDITIONAL_EMBEDDING]: boolTensor, shape=(n_structures_in_batch, 1) stores False
    for all structures for all conditional properties present in ChemGraph.

    NOTE: If a conditional property was trained on by the model but is not
    specified in the batch, then it will be attributed an unconditional embedding
    in mattergen.property_embeddings.PropertyEmbedding.forward.
    This behaviour allows unconditional samples to be drawn from a model that was trained
    on certain conditions, without having to set any conditional values in ChemGraph.
    """

    def __call__(self, x: ChemGraph) -> ChemGraph:
        cond_fields = get_cond_field_names_in_batch(x=x)
        device = x["num_atoms"].place
        use_unconditional_embedding = {}
        for cond_field in cond_fields:
            use_unconditional_embedding[cond_field] = paddle.zeros(
                shape=(len(x[cond_field]), 1), dtype="bool"
            )
        return replace_use_unconditional_embedding(
            batch=x, use_unconditional_embedding=use_unconditional_embedding
        )


class BaseUnconditionalEmbeddingModule(paddle.nn.Layer):
    only_depends_on_shape_of_input: bool
    hidden_dim: int


class EmbeddingVector(BaseUnconditionalEmbeddingModule):
    only_depends_on_shape_of_input: bool = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = paddle.nn.Embedding(num_embeddings=1, embedding_dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        This forward depends only on the shape of x and returns a tensor of zeros.
        """
        return self.embedding(paddle.zeros(shape=len(x), dtype="int64"))


class SpaceGroupEmbeddingVector(BaseUnconditionalEmbeddingModule):
    only_depends_on_shape_of_input: bool = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = paddle.nn.Embedding(
            num_embeddings=230, embedding_dim=hidden_dim
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Return embedding of the space group, 1 is subtracted from the space group number to
        make it zero-indexed.
        """
        return self.embedding(x.astype(dtype="int64") - 1)


class ZerosEmbedding(BaseUnconditionalEmbeddingModule):
    """
    Return a [n_crystals_in_batch, self.hidden_dim] tensor of zeros. This is helpfuln as the unconditional embedding
    for a property included in the adapter module if we do not want to change the unconditional score
    of the base model when properties are added in the adapter module.
    """

    only_depends_on_shape_of_input: bool = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: (paddle.Tensor | list[str])) -> paddle.Tensor:
        """
        This forward depends only on the shape of x.
        """
        return paddle.zeros(shape=[len(x), self.hidden_dim])


class ChemicalSystemMultiHotEmbedding(paddle.nn.Layer):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = paddle.nn.Linear(
            in_features=MAX_ATOMIC_NUM + 1, out_features=hidden_dim
        )

    @property
    def device(self):
        return self.parameters()[0].place
        # return next(self.parameters()).place

    @staticmethod
    def _sequence_to_multi_hot(
        x: Sequence[str], device: (paddle.CPUPlace, paddle.CUDAPlace, str)
    ) -> paddle.Tensor:
        """
        Converts a sequence of unique elements present in a single structure to a multi-hot
        vectors of 1s (present) and 0s (not present) for each unique element.

        Returns
        -------
        torch.Tensor, shape = (1, MAX_ATOMIC_NUM + 1)
        """
        chemical_system_numbers: paddle.int64 = paddle.to_tensor(
            data=[get_atomic_number(symbol=_element) for _element in x],
            dtype=int,
            place=device,
        )
        chemical_system_condition = paddle.zeros(shape=MAX_ATOMIC_NUM + 1)
        chemical_system_condition[chemical_system_numbers] = 1.0
        return chemical_system_condition.reshape(1, -1)

    @staticmethod
    def sequences_to_multi_hot(
        x: list[list[str]], device: (paddle.CPUPlace, paddle.CUDAPlace, str)
    ) -> paddle.Tensor:
        """
        Convert a list of sequences of unique elements present in a list of structures to a multi-hot
        tensor of 1s (present) and 0s (not present) for each unique element.

        Returns
        -------
        torch.Tensor, shape = (n_structures_in_batch, MAX_ATOMIC_NUM + 1)
        """
        return paddle.concat(
            x=[
                ChemicalSystemMultiHotEmbedding._sequence_to_multi_hot(
                    _x, device=device
                )
                for _x in x
            ],
            axis=0,
        )

    @staticmethod
    def convert_to_list_of_str(x: (list[str] | list[list[str]])) -> list[list[str]]:
        """
        Returns
        -------
        list[list[str]] -- a list of length n_structures_in_batch of chemical systems for each structure
            where the chemical system is specified as a list of unique elements in the structure.
        """
        if isinstance(x[0], str):
            x = [_x.split("-") for _x in x if isinstance(_x, str)]
        return x

    def forward(self, x: (list[str] | list[list[str]])) -> paddle.Tensor:
        """
        Keyword arguments
        -----------------
        x: Union[list[str], list[Sequence[str]]] -- if elements are a string, they are assumed to be
            a '-' delimited list of unique elements. If a sequence of strings, it is assumed to be a list of
            unique elements in the structure.
        """
        x = self.convert_to_list_of_str(x=x)
        multi_hot_representation: paddle.Tensor = self.sequences_to_multi_hot(
            x=x, device=self.device
        )
        return self.embedding(multi_hot_representation)


class PropertyEmbedding(paddle.nn.Layer):
    def __init__(
        self,
        name: PropertySourceId,
        conditional_embedding_module: paddle.nn.Layer,
        unconditional_embedding_module: BaseUnconditionalEmbeddingModule,
        scaler: paddle.nn.Layer = paddle.nn.Identity(),
    ):
        super().__init__()
        self.name = name
        self.conditional_embedding_module = conditional_embedding_module
        self.unconditional_embedding_module = unconditional_embedding_module
        self.scaler = scaler
        assert (
            self.name in PROPERTY_SOURCE_IDS
        ), f"PropertyEmbedding.name {self.name} not found in the database. Available property_source_ids: {PROPERTY_SOURCE_IDS}"

    def forward(self, batch: ChemGraph) -> paddle.Tensor:
        """
        ChemGraph[_USE_UNCONDITIONAL_EMBEDDING]: Dict[str, torch.BoolTensor]
        has values torch.BoolTensor, shape=(n_structures_in_batch, 1) that when True, denote that
        we should use the unconditional embedding (instead of the conditional embedding) as input
        for that property to the input nodes of each atom in the structure.

        In this forward, we return a torch.Tensor, shape=(n_structures_in_batch, hidden_dim) of
        embedding values for this property for each structure in the batch. Based on the state of
        ChemGraph[_USE_UNCONDITIONAL_EMBEDDING] we return either the unconditional or conditional
        embedding for each element i in torch.Tensor[i].

        NOTE: when self.name is not in ChemGraph[_USE_UNCONDITIONAL_EMBEDDING] we apply the
        unconditional embedding. This is to adopt the behaviour that when no conditional value is
        specified in ChemGraph, a model that was trained on said property will generate an
        unconditional score.
        """
        use_unconditional_embedding: paddle.bool = get_use_unconditional_embedding(
            batch=batch, cond_field=self.name
        )
        if (
            paddle.all(x=use_unconditional_embedding)
            and self.unconditional_embedding_module.only_depends_on_shape_of_input
        ):
            return self.unconditional_embedding_module(x=batch["num_atoms"]).to(
                batch.pos.place
            )
        else:
            data = batch[self.name]
            if isinstance(data, paddle.Tensor) and data.dim() == 2:
                data = data.squeeze(axis=-1)
            data = self.scaler(data)
            conditional_embedding: paddle.Tensor = self.conditional_embedding_module(
                data
            )
            unconditional_embedding: paddle.Tensor = (
                self.unconditional_embedding_module(x=data).to(batch.pos.place)
            )
            return paddle.where(
                condition=use_unconditional_embedding,
                x=unconditional_embedding,
                y=conditional_embedding,
            )

    def fit_scaler(self, all_data):
        if isinstance(self.scaler, paddle.nn.Identity):
            return
        self.scaler.fit(all_data)


def get_property_embeddings(
    batch: ChemGraph, property_embeddings: paddle.nn.LayerDict
) -> paddle.Tensor:
    """
    Keyword arguments
    -----------------
    property_embeddings: torch.nn.ModuleDict[PropertyToConditonOn, PropertyEmbedding] -- a dictionary
        of property embeddings. The keys are the names of the conditional fields in the batch.
    """
    ordered_keys = sorted(property_embeddings.keys())
    if len(ordered_keys) > 0:
        return paddle.concat(
            x=[property_embeddings[k].forward(batch=batch) for k in ordered_keys],
            axis=-1,
        )
    else:
        return paddle.to_tensor(data=[], place=batch["num_atoms"].place)


def set_conditional_property_values(
    batch: ChemGraph, properties: TargetProperty
) -> ChemGraph:
    not_numeric = [k for k, v in properties.items() if not isinstance(v, (int, float))]
    cond_values = {
        k: (
            [properties[k]] * len(batch["num_atoms"])
            if k in not_numeric
            else paddle.full_like(x=batch["num_atoms"], fill_value=v).reshape(-1, 1)
        )
        for k, v in properties.items()
    }
    return batch.replace(**cond_values)
