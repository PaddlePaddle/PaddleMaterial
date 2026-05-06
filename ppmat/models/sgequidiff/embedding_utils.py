"""元素/空间群/Wyckoff 嵌入工具类。"""
import json
from pathlib import Path
from typing import Optional

import paddle
import paddle.nn.functional as F

import ppmat.models.sgequidiff.global_vars as global_vars
from ppmat.models.sgequidiff.constants import (
    NUM_ELEMENTS,
    NUM_SPACE_GROUPS,
    MAX_WYCKOFF_SITES,
)


class EmbeddingTools:
    """
    元素/空间群/Wyckoff 位置嵌入工具。单例模式，初始化一次后通过 global_vars.embedding_tools 全局访问。
    """

    @paddle.no_grad()
    def __init__(
        self,
        space_group_embedding_json_path: Optional[str] = None,
        element_embedding_json_path: Optional[str] = None,
        wyckoff_embedding_json_path: Optional[str] = None,
        chemistry_embedding_type: str = "identity",
        device: str = "cpu",
    ):
        self.space_group_embedding_dict = None
        self.element_embedding_dict = None
        self.wyckoff_embedding_dict = None
        self.chemistry_embedding_type = chemistry_embedding_type
        self.device = device

        data_directory = global_vars.DATA_DIRECTORY

        if space_group_embedding_json_path is not None:
            fp = Path(data_directory / space_group_embedding_json_path).as_posix()
            with open(fp, "r") as file:
                self.space_group_embedding_dict = json.load(file)
            self.space_group_embedding_length = len(
                self.space_group_embedding_dict["1"]
            )
            self.space_group_embedding_tensor = paddle.to_tensor(
                [
                    self.space_group_embedding_dict[str(sg_num)]
                    for sg_num in range(1, 231)
                ],
                dtype=paddle.float32,
            )  # (230, space_group_embedding_length)
        else:
            self.space_group_embedding_length = NUM_SPACE_GROUPS

        if element_embedding_json_path is not None:
            fp = Path(data_directory / element_embedding_json_path).as_posix()
            with open(fp, "r") as file:
                self.element_embedding_dict = json.load(file)
            self.element_embedding_length = len(self.element_embedding_dict["0"])
            self.element_embedding_tensor = paddle.to_tensor(
                [
                    self.element_embedding_dict[str(atomic_number)]
                    for atomic_number in range(NUM_ELEMENTS + 1)
                ],
                dtype=paddle.float32,
            )  # (NUM_ELEMENTS+1, element_embedding_length)
        else:
            self.element_embedding_length = NUM_ELEMENTS

        if wyckoff_embedding_json_path is not None:
            fp = Path(data_directory / wyckoff_embedding_json_path).as_posix()
            with open(fp, "r") as file:
                self.wyckoff_embedding_dict = json.load(file)
            self.wyckoff_embedding_length = len(
                self.wyckoff_embedding_dict["1"]["a"]
            )

            wyckoff_emb_list = []
            n_wyckoffs_list = []
            for sg_num in range(1, 231):
                wyckoff_dict_of_sg = self.wyckoff_embedding_dict[str(sg_num)]
                letters = list(wyckoff_dict_of_sg.keys())

                wyckoff_ascii = [ord(l) for l in letters]
                wyckoff_idxs = [
                    ai - 97 if ai >= 97 else ai - 65 + 26 for ai in wyckoff_ascii
                ]
                sorted_letters = [
                    l
                    for l, _ in sorted(
                        zip(letters, wyckoff_idxs), key=lambda pair: pair[1]
                    )
                ]

                emb_array = paddle.to_tensor(
                    [wyckoff_dict_of_sg[l] for l in sorted_letters],
                    dtype=paddle.float32,
                )  # (n_wyckoffs, wyckoff_embedding_length)
                padding = paddle.zeros(
                    [MAX_WYCKOFF_SITES - len(letters), self.wyckoff_embedding_length]
                )
                wyckoff_emb_list.append(paddle.concat([emb_array, padding], axis=0))
                n_wyckoffs_list.append(len(letters))

            self.wyckoff_embedding_tensor = paddle.stack(wyckoff_emb_list, axis=0)
            # (230, MAX_WYCKOFF_SITES, wyckoff_embedding_length)
            self.n_wyckoffs_per_space_group = paddle.to_tensor(
                n_wyckoffs_list, dtype=paddle.int64
            )
            # (230,)
        else:
            self.wyckoff_embedding_length = MAX_WYCKOFF_SITES

    def get_space_group_embedding(self, space_group_index: paddle.Tensor) -> paddle.Tensor:
        """(batch_size,) -> (batch_size, space_group_embedding_length)"""
        assert space_group_index.dtype == paddle.int64
        if self.space_group_embedding_dict is None:
            return F.one_hot(space_group_index, NUM_SPACE_GROUPS).cast(paddle.float32)
        else:
            return self.space_group_embedding_tensor[space_group_index]

    @paddle.no_grad()
    def get_element_embedding(self, atomic_number: paddle.Tensor) -> paddle.Tensor:
        """(batch_size,) -> (batch_size, element_embedding_length)"""
        assert atomic_number.dtype == paddle.int64
        if self.element_embedding_dict is None:
            return F.one_hot(
                atomic_number - 1, NUM_ELEMENTS
            ).cast(paddle.float32)
        else:
            return self.element_embedding_tensor[atomic_number]

    @paddle.no_grad()
    def get_wyckoff_embedding(
        self,
        wyckoff_index: paddle.Tensor,
        space_group_index: paddle.Tensor,
    ) -> paddle.Tensor:
        """根据空间群和 Wyckoff 索引获取嵌入。"""
        if self.wyckoff_embedding_dict is None:
            return F.one_hot(wyckoff_index, MAX_WYCKOFF_SITES).cast(paddle.float32)
        else:
            valid_mask = self.n_wyckoffs_per_space_group[space_group_index] > wyckoff_index
            assert valid_mask.all().item(), "Invalid space group-Wyckoff index pairs"
            return self.wyckoff_embedding_tensor[space_group_index, wyckoff_index, :]

    def get_chemistry_embedding(self, chemistry: paddle.Tensor) -> paddle.Tensor:
        if self.chemistry_embedding_type == "identity":
            return chemistry
        else:
            raise ValueError(f"Invalid chemistry_embedding_type: {self.chemistry_embedding_type}")

def set_global_embedding_tools(
    space_group_embedding_json_path: Optional[str] = None,
    element_embedding_json_path: Optional[str] = None,
    wyckoff_embedding_json_path: Optional[str] = None,
    chemistry_embedding_type: str = "identity",
    device: str = "cpu",
) -> None:
    """初始化并设置全局 embedding_tools。"""
    global_vars.embedding_tools = EmbeddingTools(
        space_group_embedding_json_path=space_group_embedding_json_path,
        element_embedding_json_path=element_embedding_json_path,
        wyckoff_embedding_json_path=wyckoff_embedding_json_path,
        chemistry_embedding_type=chemistry_embedding_type,
        device=device,
    )
