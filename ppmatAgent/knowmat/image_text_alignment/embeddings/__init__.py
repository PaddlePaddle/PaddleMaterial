"""Embedding registry for KnowMat image-text alignment."""

from __future__ import annotations

import logging
from typing import Dict, List, Type

from .base import EmbeddingAdapter

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Type[EmbeddingAdapter]] = {}


def register_embedding(name: str):
    """Class decorator: register an EmbeddingAdapter under *name*."""

    def decorator(cls: Type[EmbeddingAdapter]) -> Type[EmbeddingAdapter]:
        if name in _REGISTRY:
            logger.warning("Embedding '%s' already registered; overwriting.", name)
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_embedding(name: str, **kwargs) -> EmbeddingAdapter:
    """Instantiate a registered embedding backend by name."""
    _ensure_defaults_registered()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown embedding backend '{name}'. Available: {list_embeddings()}"
        )
    return _REGISTRY[name](**kwargs)


def list_embeddings() -> List[str]:
    _ensure_defaults_registered()
    return sorted(_REGISTRY.keys())


_defaults_registered = False


def _ensure_defaults_registered() -> None:
    global _defaults_registered
    if _defaults_registered:
        return
    _defaults_registered = True

    try:
        from . import clip_adapter
        logger.debug("Registered CLIP adapter")
    except ImportError as e:
        logger.debug("CLIP adapter not loaded: %s", e)
