"""Base embedding adapter interface for KnowMat image-text alignment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingAdapter(ABC):
    """Abstract base class for all embedding backends."""

    backend_name: str = "base"
    embedding_dim: int = 0
    modalities: List[str] = []

    @abstractmethod
    def embed_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of strings into embedding vectors.

        Returns
        -------
        np.ndarray shape [N, D] float32
        """
        pass

    @abstractmethod
    def embed_image(self, image_paths: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode a list of image file paths into embedding vectors.

        Returns
        -------
        np.ndarray shape [N, D] float32
        """
        pass

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between two sets of embeddings.

        a : [M, D]   b : [N, D]
        Returns [M, N] float32
        """
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return (a @ b.T).astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(backend={self.backend_name!r}, dim={self.embedding_dim})"
        )
