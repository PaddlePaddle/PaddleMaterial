import paddle
import pytest
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.transform import set_chemical_system_string
from mattergen.common.utils.globals import MAX_ATOMIC_NUM
from mattergen.denoiser import mask_disallowed_elements
from mattergen.property_embeddings import (ChemicalSystemMultiHotEmbedding,
                                           SetConditionalEmbeddingType,
                                           SetEmbeddingType,
                                           SetUnconditionalEmbeddingType,
                                           get_use_unconditional_embedding,
                                           replace_use_unconditional_embedding)
from paddle_scatter import scatter_add


@pytest.mark.parametrize("p_unconditional", [0.0, 1.0])
def test_pre_corruption_fn(p_unconditional: float):
    pre_corruption_fn = SetEmbeddingType(
        p_unconditional=p_unconditional, dropout_fields_iid=False
    )
    pos = paddle.rand(shape=(10, 2, 3))
>>>>>>    pos[0, 0, 0] = torch.nan
>>>>>>    pos[2, 1, 2] = torch.nan
>>>>>>    pos[3] = torch.nan
    x_with_mask = pre_corruption_fn(
        x=ChemGraph(
            pos=pos,
            num_atoms=paddle.ones(shape=(10, 1), dtype=int),
            dft_bulk_modulus=pos,
        )
    )
    mask = get_use_unconditional_embedding(
        batch=x_with_mask, cond_field="dft_bulk_modulus"
    )
    num_masked = {(0.0): 3, (1.0): 10}[p_unconditional]
    assert mask.sum() == num_masked


@pytest.mark.parametrize(
    "p_unconditional, dropout_fields_iid",
    [(0.0, True), (0.0, False), (1.0, True), (1.0, False)],
)
def test_pre_corruption_fn_multi(p_unconditional: float, dropout_fields_iid: bool):
    pre_corruption_fn = SetEmbeddingType(
        p_unconditional=p_unconditional, dropout_fields_iid=dropout_fields_iid
    )
    pos = paddle.rand(shape=(10, 2, 3))
>>>>>>    pos[0, 0, 0] = torch.nan
>>>>>>    pos[2, 1, 2] = torch.nan
>>>>>>    pos[3] = torch.nan
    cell = paddle.rand(shape=(10, 3, 3))
>>>>>>    cell[1, 0, 0] = torch.nan
>>>>>>    cell[2, 1, 0] = torch.nan
>>>>>>    cell[4, 0, 2] = torch.nan
>>>>>>    cell[5, 1, 1] = torch.nan
    x_with_mask = pre_corruption_fn(
        x=ChemGraph(
            pos=pos,
            num_atoms=paddle.ones(shape=(10, 1), dtype=int),
            cell=cell,
            dft_bulk_modulus=pos,
            dft_shear_modulus=cell,
        )
    )
    for cond_field in ["dft_bulk_modulus", "dft_shear_modulus"]:
        mask = get_use_unconditional_embedding(batch=x_with_mask, cond_field=cond_field)
        number_masked = mask.sum()
        if p_unconditional == 0.0:
            if dropout_fields_iid:
                expected_number_masked = {
                    "dft_bulk_modulus": 3,
                    "dft_shear_modulus": 4,
                }[cond_field]
            else:
                expected_number_masked = 6
        elif p_unconditional == 1.0:
            expected_number_masked = 10
        else:
            raise Exception("p_unconditional must be 0.0 or 1.0")
        assert number_masked == expected_number_masked
    pre_corruption_fn = SetEmbeddingType(
        p_unconditional=p_unconditional, dropout_fields_iid=dropout_fields_iid
    )
    _ = pre_corruption_fn(
        x=ChemGraph(
            pos=pos,
            cell=cell,
            dft_bulk_modulus=pos,
            dft_shear_modulus=cell,
            num_atoms=paddle.ones(shape=(10, 1)),
        )
    )


def test_remove_conditioning_fn():
    x = ChemGraph(
        pos=paddle.rand(shape=[10, 3]),
        forces=paddle.rand(shape=[10, 3]),
        atomic_numbers=paddle.ones(shape=(10,), dtype="int32"),
        num_atoms=paddle.ones(shape=(10, 1), dtype=int),
        dft_bulk_modulus=paddle.randn(shape=[10, 3]),
        dft_shear_modulus=paddle.randn(shape=[10, 3]),
    )
    cond_fields = ["dft_bulk_modulus", "dft_shear_modulus"]
    mask_all = SetUnconditionalEmbeddingType()
    x_with_mask = mask_all(x=x)
    for cond_field in cond_fields:
        assert paddle.allclose(
            x=get_use_unconditional_embedding(batch=x_with_mask, cond_field=cond_field),
            y=paddle.ones(shape=(10, 1), dtype="bool"),
        ).item(), ""


def test_keep_conditioning_fn():
    x = ChemGraph(
        pos=paddle.rand(shape=[10, 3]),
        forces=paddle.rand(shape=[10, 3]),
        atomic_numbers=paddle.ones(shape=(10,), dtype="int32"),
        num_atoms=paddle.ones(shape=(10, 1), dtype=int),
        dft_bulk_modulus=paddle.rand(shape=[10, 3]),
        dft_shear_modulus=paddle.randn(shape=[10, 3]),
    )
    x_with_mask = SetConditionalEmbeddingType()(x=x)
    assert paddle.allclose(
        x=get_use_unconditional_embedding(
            batch=x_with_mask, cond_field="dft_bulk_modulus"
        ),
        y=paddle.zeros(shape=(10, 1), dtype="bool"),
    ).item(), ""
    assert paddle.allclose(
        x=get_use_unconditional_embedding(
            batch=x_with_mask, cond_field="dft_shear_modulus"
        ),
        y=paddle.zeros(shape=(10, 1), dtype="bool"),
    ).item(), ""


@pytest.mark.parametrize("zero_based_predictions", [True, False])
def test_mask_disallowed_elements(zero_based_predictions: bool):
    paddle.seed(seed=23232)
    samples = [
        ChemGraph(
            pos=paddle.rand(shape=[10, 3]),
            num_atoms=paddle.to_tensor(data=[10]),
            atomic_numbers=6 * paddle.ones(shape=(10,), dtype="int32"),
            cell=paddle.eye(num_rows=3),
        ),
        ChemGraph(
            pos=paddle.rand(shape=[5, 3]),
            num_atoms=paddle.to_tensor(data=[5]),
            atomic_numbers=paddle.to_tensor(data=[57, 11, 8, 51, 21]),
            cell=paddle.eye(num_rows=3),
        ),
        ChemGraph(
            pos=paddle.rand(shape=[15, 3]),
            num_atoms=paddle.to_tensor(data=[15]),
            atomic_numbers=paddle.to_tensor(
                data=[57, 11, 8, 51, 21, 57, 11, 8, 51, 21, 57, 11, 8, 51, 21]
            ),
            cell=paddle.eye(num_rows=3),
        ),
    ]
    transform = set_chemical_system_string
    batch = collate([transform(sample) for sample in samples])
    assert hasattr(batch, "chemical_system")
    assert hasattr(batch, "pos")
    assert hasattr(batch, "batch")
    assert hasattr(batch, "cell")
    assert hasattr(batch, "atomic_numbers")
    assert hasattr(batch, "num_atoms")
    mask = paddle.to_tensor(data=[0, 0, 1], dtype="bool")[:, None]
    batch_chemgraph = ChemGraph(
        pos=batch.pos,
        cell=batch.cell,
        atomic_numbers=batch.atomic_numbers,
        num_atoms=batch.num_atoms,
        chemical_system=batch.chemical_system,
    )
    batch_chemgraph = replace_use_unconditional_embedding(
        batch=batch_chemgraph, use_unconditional_embedding={"chemical_system": mask}
    )
    example_logits = paddle.randn(shape=[batch.pos.shape[0], MAX_ATOMIC_NUM + 1])
    masked_logits = mask_disallowed_elements(
        logits=example_logits,
        x=batch_chemgraph,
        batch_idx=batch.batch,
        predictions_are_zero_based=zero_based_predictions,
    )
    sampled = paddle.distribution.Categorical(logits=masked_logits).sample() + int(
        zero_based_predictions
    )
    sampled_onehot = paddle.eye(num_rows=MAX_ATOMIC_NUM + 1)[sampled]
    sampled_chemical_systems = scatter_add(sampled_onehot, batch.batch, dim=0)
    chemsys_multi_hot: paddle.int64 = (
        ChemicalSystemMultiHotEmbedding.sequences_to_multi_hot(
            x=ChemicalSystemMultiHotEmbedding.convert_to_list_of_str(
                x=batch.chemical_system
            ),
            device=mask.place,
        )
    )
    for ix, system in enumerate(sampled_chemical_systems):
        sampled_types = system.nonzero()[:, 0].tolist()
        chemsys = chemsys_multi_hot[ix].nonzero()[:, 0].tolist()
        if mask[ix] == 0:
            assert set(sampled_types).difference(set(chemsys)) == set()
        else:
            assert set(sampled_types).difference(set(chemsys)) != set()
