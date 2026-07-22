"""CLIP embedding adapter for KnowMat image-text alignment.

Supports: openai/clip-vit-base-patch32, openai/clip-vit-large-patch14
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np

from . import register_embedding
from .base import EmbeddingAdapter

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "openai/clip-vit-base-patch32"


@register_embedding("clip")
class CLIPAdapter(EmbeddingAdapter):
    """Multimodal embedding using CLIP (text + image).

    Parameters
    ----------
    model_name : str
        HuggingFace CLIP model name.
    device : str
        "cpu" or "cuda".
    """

    backend_name = "clip"
    modalities = ["text", "image"]

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._processor = None
        self._model = None
        self._torch = None

    def embed_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self._lazy_load()
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            with self._torch.no_grad():
                feats = self._model.get_text_features(**inputs)
                if not isinstance(feats, self._torch.Tensor):
                    feats = feats.pooler_output
            all_vecs.append(feats.cpu().numpy())
        result = np.vstack(all_vecs).astype(np.float32)
        return result

    def embed_image(self, image_paths: List[str], batch_size: int = 16) -> np.ndarray:
        self._lazy_load()
        from PIL import Image

        all_vecs = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    # Create a blank image if loading fails
                    images.append(Image.new("RGB", (224, 224)))
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                if not isinstance(feats, self._torch.Tensor):
                    feats = feats.pooler_output
            all_vecs.append(feats.cpu().numpy())
        return np.vstack(all_vecs).astype(np.float32)

    def _lazy_load(self) -> None:
        if self._model is not None:
            return

        # Must set before importing transformers to prevent background network threads
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._torch = torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for CLIP.\n"
                "Run: pip install transformers torch Pillow"
            )

        load_kwargs = {"local_files_only": True}
        try:
            self._processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=True)
        except Exception:
            logger.info("CLIP processor not in local cache, downloading from network...")
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            load_kwargs.pop("local_files_only", None)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)

        try:
            self._model = CLIPModel.from_pretrained(self.model_name, **load_kwargs)
        except Exception:
            logger.info("CLIP model not in local cache, downloading from network...")
            self._model = CLIPModel.from_pretrained(self.model_name)
        self._model.eval().to(self.device)
        self.embedding_dim = self._model.config.projection_dim
        logger.info("Loaded %s (dim=%d)", self.model_name, self.embedding_dim)
