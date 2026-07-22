"""CLI entry point for building an alignment dataset from OCR outputs.

Usage:
    python -m ppmatAgent.knowmat.image_text_alignment.build_dataset \\
        --input_dir /path/to/ocr_outputs \\
        --output_dir /path/to/dataset \\
        --model clip \\
        --caption_blend 0.3 \\
        --top_k 5
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Build a sci-align-compatible dataset from PaddleOCR/compatible OCR outputs."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Root dir containing paper subdirs (or a single-paper dir)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for the dataset files")
    parser.add_argument("--model", default="clip",
                        help="Embedding backend (default: clip)")
    parser.add_argument("--device", default="cpu",
                        help="Device (default: cpu)")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--caption_blend", type=float, default=0.0,
                        help="Caption-blend alpha (0=off, 0.3 recommended)")
    parser.add_argument("--no_save_embeddings", action="store_true",
                        help="Skip saving .npy embedding files")
    args = parser.parse_args()

    from ppmatAgent.knowmat.image_text_alignment.dataset_builder import DatasetBuilder, DatasetBuildConfig

    cfg = DatasetBuildConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        device=args.device,
        top_k=args.top_k,
        min_score=args.min_score,
        batch_size=args.batch_size,
        caption_blend=args.caption_blend,
        save_embeddings=not args.no_save_embeddings,
    )

    result = DatasetBuilder(cfg).build()
    print(f"\n{result}")
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for e in result.errors:
            print(f"  - {e}")
    return 0 if not result.errors else 1


if __name__ == "__main__":
    sys.exit(main())
