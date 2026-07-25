# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MatterChat trainer: BLIP-2 3-stage training on top of BaseTrainer."""

from __future__ import annotations

import json
import os
from typing import Optional

import paddle
import paddle.nn as nn
from paddle.io import Dataset

from ppmat.models.matterchat.q_former.q_former_complete import Blip2MistralInstruct
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import logger


# =============================================================================
# LoRA
# =============================================================================


class LoRALinear(nn.Layer):
    """Low-Rank Adaptation wrapper for an existing nn.Linear."""

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        for p in base_linear.parameters():
            p.stop_gradient = True
        self.base = base_linear
        in_features = base_linear.weight.shape[0]
        out_features = base_linear.weight.shape[1]
        lora_dtype = base_linear.weight.dtype
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            dtype=lora_dtype,
            default_initializer=nn.initializer.KaimingUniform(
                negative_slope=0, fan_in=in_features if in_features > 0 else 1
            ),
        )
        self.lora_B = self.create_parameter(
            shape=[r, out_features],
            dtype=lora_dtype,
            default_initializer=nn.initializer.Constant(value=0.0),
        )
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        base_out = self.base(x)
        lora_update = (self.dropout(x) @ self.lora_A) @ self.lora_B
        return base_out + lora_update * self.scaling

    def merge_into_base(self) -> nn.Linear:
        """Fold LoRA weights into the base linear and return a plain nn.Linear."""
        with paddle.no_grad():
            # lora_A: [in, r], lora_B: [r, out] -> A@B: [in, out] == base.weight shape
            merged_w = self.base.weight + (self.lora_A @ self.lora_B) * self.scaling
            new_linear = nn.Linear(
                self.base.weight.shape[0],
                self.base.weight.shape[1],
                bias_attr=self.base.bias is not None,
            )
            new_linear.weight.set_value(merged_w)
            if self.base.bias is not None:
                new_linear.bias.set_value(self.base.bias)
        return new_linear


# We avoid lm_head and embed_tokens to keep the LLM's vocabulary mapping untouched.
MISTRAL_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def apply_lora_to_mistral(
    llm_model: nn.Layer,
    target_names=MISTRAL_LORA_TARGETS,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> int:
    """Replace targeted nn.Linear layers in the Mistral LLM with LoRALinear.

    Matching is done by leaf attribute name (e.g. ``q_proj``), so every
    projection with that name across all decoder layers is replaced.
    """
    n_replaced = 0
    for _, parent in llm_model.named_sublayers():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_names:
                lora = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora)
                n_replaced += 1
    logger.info(f"Applied LoRA to {n_replaced} linear layers in the LLM (r={r}).")
    return n_replaced


# =============================================================================
# Module wrapper (BaseTrainer-compatible forward)
# =============================================================================


class MatterChatModule(nn.Layer):
    """Wraps Blip2MistralInstruct to expose the BaseTrainer API."""

    def __init__(
        self,
        blip2_model: Blip2MistralInstruct,
        stage: int = 2,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_amp: bool = True,
    ) -> None:
        super().__init__()
        self.blip2 = blip2_model
        self.stage = stage
        self.use_lora = use_lora
        self.use_amp = use_amp
        # Freeze CHGNet (always)
        for p in self.blip2.material_encoder.parameters():
            p.stop_gradient = True
        # Stage 2/3: train Q-Former + llm_proj
        if stage in (2, 3):
            for p in self.blip2.Qformer.parameters():
                p.stop_gradient = False
            self.blip2.query_tokens.stop_gradient = False
            for p in self.blip2.llm_proj.parameters():
                p.stop_gradient = False
        # Stage 1: only Q-Former
        if stage == 1:
            for p in self.blip2.Qformer.parameters():
                p.stop_gradient = False
            self.blip2.query_tokens.stop_gradient = False
        # Stage 2/3: LLM is either fully frozen (stage 2) or LoRA-only (stage 3).
        # Mistral __init__ already freezes everything; we selectively re-enable LoRA.
        for p in self.blip2.llm_model.parameters():
            p.stop_gradient = True
        if use_lora:
            apply_lora_to_mistral(
                self.blip2.llm_model,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        n_trainable = sum(p.numel() for p in self.parameters() if not p.stop_gradient)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MatterChatModule: stage={stage}, use_lora={use_lora}; "
            f"trainable params {n_trainable:,} / {n_total:,} "
            f"({100.0 * n_trainable / max(n_total, 1):.2f}%)"
        )

    def forward(self, batch_data: dict) -> dict:
        """BaseTrainer-compatible forward."""
        if self.stage == 1:
            return self._forward_stage1(batch_data)
        return self._forward_stage23(batch_data)

    def _forward_stage23(self, batch_data: dict) -> dict:
        from pymatgen.core import Structure as _Structure

        # Reconstruct pymatgen.Structure from serialized dicts.
        # MTCollator preserves the as_dict output as opaque dicts.
        structures = [_Structure.from_dict(d) for d in batch_data["structure"]]
        # Encode each structure with the frozen CHGNet encoder.
        # NOTE: This loops per-structure because CHGNet's
        # predict_structure_embedding handles a single Structure. Batched
        # encoding would require a batch graph converter and is a future
        # optimization; the loop is wrapped in no_grad so it is cheap.
        with paddle.no_grad():
            embedding_list = []
            embedding_mask = []
            for s in structures:
                emb = self.blip2.material_encoder.predict_structure_embedding(s)
                embedding_list.append(emb)
                att = paddle.ones([emb.shape[0]], dtype=paddle.int64)
                embedding_mask.append(att)
        samples = {
            "text_input": batch_data["text_input"],
            "text_output": batch_data.get("text_output", None),
            "text": batch_data.get("text_output", ""),
        }
        loss = self.blip2(samples, embedding_list, embedding_mask)
        if isinstance(loss, dict):
            loss_value = loss.get("loss", None)
            if loss_value is None:
                loss_value = list(loss.values())[0]
        else:
            loss_value = loss
        return {
            "loss_dict": {"loss": loss_value},
            "pred_dict": {},
        }

    def _forward_stage1(self, batch_data: dict) -> dict:
        """Stage 1: ITC + ITM + ITG on Q-Former only (not yet implemented)."""
        raise NotImplementedError(
            "Stage 1 (ITC+ITM+ITG) is not implemented yet. "
            "Use Stage 2 / 3 for the available training modes."
        )

    def assert_all_on_gpu(self) -> tuple[int, int]:
        """Verify every parameter lives on the GPU. Returns (gpu_count, cpu_count)."""
        gpu = cpu = 0
        for p in self.parameters():
            place = str(p.place).lower()
            if "gpu" in place or "cuda" in place:
                gpu += 1
            else:
                cpu += 1
        return gpu, cpu


# =============================================================================
# Trainer (extends BaseTrainer)
# =============================================================================


class MatterChatTrainer(BaseTrainer):
    """Trainer for the MatterChat BLIP-2-style model."""

    def __init__(
        self,
        config: dict,
        model: MatterChatModule,
        train_dataloader=None,
        val_dataloader=None,
        optimizer=None,
        lr_scheduler=None,
        compute_metric_func_dict=None,
    ) -> None:
        gpu, cpu = model.assert_all_on_gpu()
        if cpu > 0:
            raise RuntimeError(
                f"MatterChatTrainer: {cpu} parameters are still on CPU. "
                "Build the LLM in float16 on GPU."
            )
        logger.info(f"MatterChatTrainer: verified {gpu} parameters are on GPU, " f"0 on CPU.")
        super().__init__(
            config=config,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            compute_metric_func_dict=compute_metric_func_dict,
        )


# =============================================================================
# Dataset (instruction-tuning for crystal Q&A)
# =============================================================================


class MTDataset(Dataset):
    """Instruction-tuning dataset for crystal Q&A."""

    def __init__(
        self,
        material_pkl_path: Optional[str] = None,
        question_json_path: Optional[str] = None,
        answer_json_path: Optional[str] = None,
        task_mapping_pkl_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        if material_pkl_path:
            material_pkl_path = os.path.expanduser(material_pkl_path)
        if question_json_path:
            question_json_path = os.path.expanduser(question_json_path)
        if answer_json_path:
            answer_json_path = os.path.expanduser(answer_json_path)
        if task_mapping_pkl_path:
            task_mapping_pkl_path = os.path.expanduser(task_mapping_pkl_path)
        self.material_pkl_path = material_pkl_path
        if material_pkl_path and os.path.exists(material_pkl_path):
            self._load_real(
                material_pkl_path,
                question_json_path,
                answer_json_path,
                task_mapping_pkl_path,
            )
        else:
            logger.warning(
                f"MTDataset: no real data at {material_pkl_path!r}; "
                "falling back to a small built-in demo set."
            )
            self._load_demo()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_real(
        self,
        mat_pkl,
        q_json,
        a_json,
        task_pkl,
    ) -> None:
        import pickle

        with open(mat_pkl, "rb") as f:
            materials = pickle.load(f)
        questions = json.load(open(q_json)) if q_json else []
        answers = json.load(open(a_json)) if a_json else []
        task_mapping = pickle.load(open(task_pkl, "rb")) if task_pkl else {}
        self.samples = []
        for mat_id, mat in enumerate(materials):
            qa_list = task_mapping.get(mat_id, [])
            for qa_idx in qa_list:
                if qa_idx < len(questions) and qa_idx < len(answers):
                    self.samples.append(
                        {
                            "structure": mat,
                            "text_input": questions[qa_idx],
                            "text_output": answers[qa_idx],
                        }
                    )
        logger.info(f"MTDataset: loaded {len(self.samples)} samples from {mat_pkl}")

    def _load_demo(self) -> None:
        from pymatgen.core import Lattice, Structure

        si = Structure(Lattice.cubic(5.43), ["Si"] * 2, [[0, 0, 0], [0.25, 0.25, 0.25]])
        gan = Structure(Lattice.cubic(4.5), ["Ga", "N"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.samples = [
            {
                "structure": si,
                "text_input": "what is the chemical formula of this material?",
                "text_output": "Si",
            },
            {
                "structure": si,
                "text_input": "Is this material stable?",
                "text_output": "Yes",
            },
            {
                "structure": gan,
                "text_input": "what is the chemical formula of this material?",
                "text_output": "GaN",
            },
            {
                "structure": gan,
                "text_input": "Is this material metal or not metal?",
                "text_output": "No",
            },
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        # Serialize pymatgen.Structure to a plain dict so Paddle DataLoader
        # can pickle/batch it. Reconstruction happens in _forward_stage23
        # via Structure.from_dict(...).
        sample = self.samples[idx]
        structure = sample["structure"]
        return {
            "structure": structure.as_dict(),
            "text_input": sample["text_input"],
            "text_output": sample["text_output"],
        }


class MTCollator:
    """Collate for MTDataset that keeps pymatgen Structure dicts opaque."""

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def _is_pmg_as_dict(d):
        return isinstance(d, dict) and d.get("@module", "").startswith("pymatgen.core")

    def __call__(self, batch):
        # Paddle's DataLoader silently drops batches with NO Tensors,
        # so we include a small metadata Tensor.
        result = {}
        keys = batch[0].keys()
        for k in keys:
            items = [b[k] for b in batch]
            sample_item = items[0]
            if self._is_pmg_as_dict(sample_item):
                result[k] = items
            elif (
                isinstance(sample_item, (list, tuple)) and items and self._is_pmg_as_dict(items[0])
            ):
                if all(len(it) == 1 for it in items):
                    result[k] = [it[0] for it in items]
                else:
                    result[k] = [it[0] if len(it) == 1 else it for it in items]
            else:
                result[k] = items
        result["_batch_size"] = paddle.to_tensor(len(batch))
        return result


__all__ = [
    "LoRALinear",
    "apply_lora_to_mistral",
    "MISTRAL_LORA_TARGETS",
    "MatterChatModule",
    "MatterChatTrainer",
    "MTDataset",
    "MTCollator",
]
