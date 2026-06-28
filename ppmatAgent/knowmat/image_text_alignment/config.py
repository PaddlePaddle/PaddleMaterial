"""Configuration for image-text alignment."""

from dataclasses import dataclass


@dataclass
class AlignmentConfig:
    """Configuration for image-text alignment."""

    enabled: bool = True
    model: str = "clip"
    device: str = "cpu"
    top_k: int = 5
    min_score: float = 0.0
    batch_size: int = 32
    # caption_blend: 0 = pure image vector; 0.3 = 70% image + 30% caption text.
    # Blending improves alignment for figures whose visual content alone is
    # ambiguous — the caption text anchors the vector to the correct domain.
    caption_blend: float = 0.0
    # save_dataset: write sci-align-compatible JSONL + npy files to output_dir
    # alongside the normal LangGraph extraction output.
    save_dataset: bool = False
