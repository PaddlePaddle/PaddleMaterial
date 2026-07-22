"""Image-text alignment logic for KnowMat.

Adapted from sci-align's pipeline builder.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embeddings import get_embedding
from .tokenizer import OcrItemTokenizer, SentenceToken, VisualToken

logger = logging.getLogger(__name__)

# Reranking bonus/penalty weights
FIG_MATCH_BONUS = 0.20
PAGE_NEAR_BONUS = 0.05
WRONG_FIG_PENALTY = 0.15


def _confidence(final: float, has_same: bool) -> str:
    if has_same or final >= 0.6:
        return "high"
    if final >= 0.4:
        return "medium"
    return "low"


def _compute_final_score(
    cosine: float, vis_fids: List[str], vis_page: Optional[int],
    sent_figs: List[str], sent_page: Optional[int],
) -> Tuple[float, Optional[str], bool, bool]:
    bonus = 0.0
    matched_fig = None
    has_same = False
    has_wrong = False

    if sent_figs:
        common = [f for f in sent_figs if f in vis_fids]
        if common:
            has_same = True
            matched_fig = common[0]
            bonus += FIG_MATCH_BONUS
        else:
            has_wrong = True
            bonus -= WRONG_FIG_PENALTY

    if vis_page is not None and sent_page is not None:
        if abs(vis_page - sent_page) <= 1:
            bonus += PAGE_NEAR_BONUS

    return cosine + bonus, matched_fig, has_same, has_wrong


@dataclass
class RelatedSentence:
    """A sentence related to an image."""

    text: str
    page: Optional[int]
    section: Optional[str]
    score: float         # final reranked score (cosine + bonuses/penalties)
    cosine_score: float  # raw CLIP cosine before reranking
    caption_text_cosine: float  # cosine(image_caption_text, sentence) — diagnostic
    rank: int
    mentioned_figures: List[str]
    has_same_figure_anchor: bool
    has_wrong_figure_anchor: bool
    source: str
    token_id: str
    confidence: str = "low"  # "high" | "medium" | "low"


@dataclass
class ImageTextAlignment:
    """Alignment result for one image."""

    image_id: str
    image_path: str
    figure_num: Optional[str]
    caption: str
    normalized_figure_id: Optional[str]
    figure_number: Optional[int]
    subfigure_id: Optional[str]
    all_figure_ids: List[str]
    page_number: Optional[int]
    related_sentences: List[RelatedSentence]


class ImageTextAligner:
    """Align images with sentences using embedding similarity.

    When save_dataset=True, also writes a sci-align-compatible dataset to
    output_dir (topk_pairs.jsonl, direct_pairs.jsonl, image_units.jsonl,
    sentence_units.jsonl, embeddings, metadata.json, human_check.md).
    Embeddings are computed only once for both outputs.
    """

    def __init__(
        self,
        model: str = "clip",
        device: str = "cpu",
        top_k: int = 5,
        min_score: float = 0.0,
        batch_size: int = 32,
        caption_blend: float = 0.0,
        save_dataset: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.top_k = top_k
        self.min_score = min_score
        self.batch_size = batch_size
        self.caption_blend = caption_blend
        self.save_dataset = save_dataset
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = get_embedding(self.model, device=self.device)
        return self._encoder

    def align(
        self,
        ocr_items: List[Dict[str, Any]],
        paper_id: str,
        output_dir: Optional[str] = None,
    ) -> List[ImageTextAlignment]:
        """
        Align images with sentences from ocr_items.

        When save_dataset=True is set on the aligner, also writes
        sci-align-compatible JSONL + npy files to output_dir.
        Embeddings are computed only once.

        Parameters
        ----------
        ocr_items : List[Dict[str, Any]]
            The ocr_items list from KnowMat's parse_pdf node.
        paper_id : str
            The paper identifier.
        output_dir : Optional[str]
            Output directory — used for resolving relative image paths,
            and (when save_dataset=True) for writing dataset files.

        Returns
        -------
        List[ImageTextAlignment]
            Alignment results for each image (for the LangGraph state).
        """
        # ── Tokenize ──────────────────────────────────────────────────────
        tokenizer = OcrItemTokenizer()
        visual_tokens, sentence_tokens = tokenizer.tokenize(
            ocr_items, paper_id, output_dir
        )

        if not visual_tokens:
            logger.warning("No visual tokens found")
            return []

        encoder = self._get_encoder()
        image_paths = [vt.image_path for vt in visual_tokens if vt.image_path]
        if not image_paths:
            logger.warning("No image paths found")
            return []

        # Build index mapping: position in visual_tokens -> position in img_vecs
        # Only tokens with a valid image_path get embedded.
        vt_to_embed_idx = {}
        embed_i = 0
        for i, vt in enumerate(visual_tokens):
            if vt.image_path:
                vt_to_embed_idx[i] = embed_i
                embed_i += 1

        # ── Embed ─────────────────────────────────────────────────────────
        logger.info("Embedding %d images...", len(image_paths))
        img_vecs = encoder.embed_image(image_paths, batch_size=self.batch_size)
        img_vecs = img_vecs / (np.linalg.norm(img_vecs, axis=1, keepdims=True) + 1e-8)

        # Embed captions (used for caption_blend and caption_text_cosine diagnostic)
        captions = [vt.caption for vt in visual_tokens]
        logger.info("Embedding %d captions...", len(captions))
        cap_vecs = encoder.embed_text(captions, batch_size=self.batch_size)
        cap_vecs = cap_vecs / (np.linalg.norm(cap_vecs, axis=1, keepdims=True) + 1e-8)

        # caption_blend: blend image vector with caption text vector so that
        # visually ambiguous figures are anchored to their caption's domain.
        if self.caption_blend > 0.0:
            logger.info("Applying caption_blend alpha=%.2f ...", self.caption_blend)
            img_vecs = (1.0 - self.caption_blend) * img_vecs + self.caption_blend * cap_vecs
            img_vecs = img_vecs / (np.linalg.norm(img_vecs, axis=1, keepdims=True) + 1e-8)

        # TopK pool: exclude caption_text sentences
        topk_sent_positions = [
            i for i, s in enumerate(sentence_tokens) if s.source != "caption_text"
        ]
        topk_sents = [sentence_tokens[i] for i in topk_sent_positions]
        logger.info(
            "TopK pool: %d body sentences (excluded %d caption_text)",
            len(topk_sents), len(sentence_tokens) - len(topk_sents),
        )

        if not topk_sents:
            logger.warning("No body sentences found")
            return [
                ImageTextAlignment(
                    image_id=vt.token_id, image_path=vt.image_path,
                    figure_num=str(vt.figure_number) if vt.figure_number else None,
                    caption=vt.caption, normalized_figure_id=vt.normalized_figure_id,
                    figure_number=vt.figure_number, subfigure_id=vt.subfigure_id,
                    all_figure_ids=vt.all_figure_ids, page_number=vt.page_number,
                    related_sentences=[],
                )
                for vt in visual_tokens
            ]

        logger.info("Embedding %d sentences...", len(topk_sents))
        txt_vecs = encoder.embed_text(
            [s.text for s in topk_sents], batch_size=self.batch_size
        )
        txt_vecs = txt_vecs / (np.linalg.norm(txt_vecs, axis=1, keepdims=True) + 1e-8)

        # ── Per-paper sub-matrix matching ─────────────────────────────────
        paper_vis_idx: dict = defaultdict(list)
        paper_sent_idx: dict = defaultdict(list)
        for i, vt in enumerate(visual_tokens):
            if i in vt_to_embed_idx:
                paper_vis_idx[vt.paper_id].append(i)
        for j, st in enumerate(topk_sents):
            paper_sent_idx[st.paper_id].append(j)

        # Collect both output formats in a single pass
        alignments: List[ImageTextAlignment] = []
        topk_pairs_raw: List[Dict] = []    # for dataset export
        img_topk_raw: List[Dict] = []      # legacy format

        for pid, vis_indices in paper_vis_idx.items():
            sent_indices = paper_sent_idx.get(pid, [])
            if not sent_indices:
                continue

            embed_indices = [vt_to_embed_idx[i] for i in vis_indices]
            v_mat = img_vecs[embed_indices]
            c_mat = cap_vecs[embed_indices]
            s_mat = txt_vecs[np.array(sent_indices)]
            paper_sim = v_mat @ s_mat.T
            paper_cap_sim = c_mat @ s_mat.T

            for local_i, global_i in enumerate(vis_indices):
                vt = visual_tokens[global_i]
                row = paper_sim[local_i]
                cap_row = paper_cap_sim[local_i]

                vis_fids = vt.all_figure_ids or (
                    [vt.normalized_figure_id] if vt.normalized_figure_id else []
                )

                anchor_scored, clean_scored, wrong_scored = [], [], []
                for local_j, global_j in enumerate(sent_indices):
                    st = topk_sents[global_j]
                    cosine = float(row[local_j])
                    cap_cosine = float(cap_row[local_j])
                    final, matched_fig, has_same, has_wrong = _compute_final_score(
                        cosine, vis_fids, vt.page_number,
                        st.mentioned_figures, st.page_number,
                    )
                    entry = (final, cosine, cap_cosine, global_j, st,
                             matched_fig, has_same, has_wrong)
                    if has_same:
                        anchor_scored.append(entry)
                    elif has_wrong:
                        wrong_scored.append(entry)
                    else:
                        clean_scored.append(entry)

                for lst in (anchor_scored, clean_scored, wrong_scored):
                    lst.sort(key=lambda x: x[0], reverse=True)
                selected = (anchor_scored + clean_scored + wrong_scored)[:self.top_k]

                # Build both output formats from the same selected list
                related_sentences: List[RelatedSentence] = []
                top_sents_legacy = []
                for rank, (final, cosine, cap_cosine, global_j, st,
                            matched_fig, has_same, has_wrong) in enumerate(selected, 1):
                    if final < self.min_score:
                        continue
                    conf = _confidence(final, has_same)

                    # LangGraph format
                    related_sentences.append(RelatedSentence(
                        text=st.text, page=st.page_number, section=st.section,
                        score=round(final, 6), cosine_score=round(cosine, 6),
                        caption_text_cosine=round(cap_cosine, 6), rank=rank,
                        mentioned_figures=st.mentioned_figures,
                        has_same_figure_anchor=has_same,
                        has_wrong_figure_anchor=has_wrong,
                        source=st.source, token_id=st.token_id, confidence=conf,
                    ))

                    # Dataset format (same data, dict shape)
                    topk_pairs_raw.append({
                        "pair_type": "topk_similarity",
                        "visual_token_id": vt.token_id,
                        "visual_paper_id": vt.paper_id,
                        "image_path": vt.image_path,
                        "normalized_figure_id": vt.normalized_figure_id,
                        "text_token_id": st.token_id,
                        "text_paper_id": st.paper_id,
                        "text": st.text,
                        "source": st.source,
                        "rank": rank,
                        "cosine_score": round(cosine, 6),
                        "caption_text_cosine": round(cap_cosine, 6),
                        "final_score": round(final, 6),
                        "score": round(final, 6),
                        "mentioned_figures": st.mentioned_figures,
                        "matched_figure_id": matched_fig,
                        "has_same_figure_anchor": has_same,
                        "has_wrong_figure_anchor": has_wrong,
                        "confidence": conf,
                    })
                    top_sents_legacy.append({
                        "token_id": st.token_id, "paper_id": st.paper_id,
                        "text": st.text, "source": st.source,
                        "cosine_score": round(cosine, 6),
                        "final_score": round(final, 6),
                        "score": round(final, 6),
                        "rank": rank, "confidence": conf,
                    })

                alignments.append(ImageTextAlignment(
                    image_id=vt.token_id, image_path=vt.image_path,
                    figure_num=str(vt.figure_number) if vt.figure_number else None,
                    caption=vt.caption, normalized_figure_id=vt.normalized_figure_id,
                    figure_number=vt.figure_number, subfigure_id=vt.subfigure_id,
                    all_figure_ids=vt.all_figure_ids, page_number=vt.page_number,
                    related_sentences=related_sentences,
                ))
                img_topk_raw.append({
                    "token_id": vt.token_id, "paper_id": vt.paper_id,
                    "figure_id": vt.figure_id, "image_path": vt.image_path,
                    "caption": vt.caption, "top_k": top_sents_legacy,
                })

        # ── Save dataset files (same embedding pass, no re-embedding) ─────
        if self.save_dataset and output_dir:
            self._save_dataset(
                output_dir=output_dir,
                visual_tokens=visual_tokens,
                sentence_tokens=sentence_tokens,
                topk_sents=topk_sents,
                img_vecs=img_vecs,
                txt_vecs_all=self._rebuild_full_txt_vecs(
                    sentence_tokens, topk_sents, txt_vecs
                ),
                topk_pairs=topk_pairs_raw,
                img_topk=img_topk_raw,
                paper_vis_idx=paper_vis_idx,
                paper_sent_idx=paper_sent_idx,
                img_vecs_orig=img_vecs,
                txt_vecs=txt_vecs,
                vt_to_embed_idx=vt_to_embed_idx,
            )

        logger.info(
            "Aligned %d images with %d related sentences total",
            len(alignments),
            sum(len(a.related_sentences) for a in alignments),
        )
        return alignments

    # ── Dataset export ─────────────────────────────────────────────────────

    def _rebuild_full_txt_vecs(
        self, sentence_tokens, topk_sents, topk_vecs: np.ndarray
    ) -> np.ndarray:
        """Rebuild full-length txt_vecs array (all sentences, not just topk pool)."""
        dim = topk_vecs.shape[1] if topk_vecs.ndim == 2 else 512
        full = np.zeros((len(sentence_tokens), dim), dtype=np.float32)
        topk_set = {st.token_id: i for i, st in enumerate(topk_sents)}
        for i, st in enumerate(sentence_tokens):
            idx = topk_set.get(st.token_id)
            if idx is not None:
                full[i] = topk_vecs[idx]
        return full

    def _save_dataset(
        self,
        output_dir: str,
        visual_tokens: List[VisualToken],
        sentence_tokens: List[SentenceToken],
        topk_sents: List[SentenceToken],
        img_vecs: np.ndarray,
        txt_vecs_all: np.ndarray,
        topk_pairs: List[Dict],
        img_topk: List[Dict],
        paper_vis_idx: dict,
        paper_sent_idx: dict,
        img_vecs_orig: np.ndarray,
        txt_vecs: np.ndarray,
        vt_to_embed_idx: dict = None,
    ) -> None:
        """Write sci-align-compatible dataset files to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        encoder = self._get_encoder()
        model_name = getattr(encoder, "model_name", self.model)
        embedding_dim = int(getattr(encoder, "embedding_dim", img_vecs.shape[1]))

        # Unit metadata
        self._write_jsonl(out / "image_units.jsonl",
                          [vt.to_dict() for vt in visual_tokens])
        self._write_jsonl(out / "sentence_units.jsonl",
                          [st.to_dict() for st in sentence_tokens])

        # Embeddings
        np.save(str(out / "image_embeddings.npy"), img_vecs.astype(np.float32))
        logger.info("Wrote image_embeddings.npy  shape=%s", list(img_vecs.shape))
        np.save(str(out / "sentence_embeddings.npy"), txt_vecs_all.astype(np.float32))
        logger.info("Wrote sentence_embeddings.npy  shape=%s", list(txt_vecs_all.shape))

        index = {
            "image_embeddings": {
                "path": "image_embeddings.npy",
                "shape": list(img_vecs.shape),
                "token_ids": [vt.token_id for vt in visual_tokens],
            },
            "sentence_embeddings": {
                "path": "sentence_embeddings.npy",
                "shape": list(txt_vecs_all.shape),
                "token_ids": [st.token_id for st in sentence_tokens],
            },
            "model": {
                "name": model_name,
                "embedding_dim": embedding_dim,
                "normalized": True,
                "caption_blend": self.caption_blend,
            },
        }
        (out / "embedding_index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False)
        )

        # Direct pairs: image ↔ own caption, score=1.0
        direct_pairs = [
            {
                "pair_type": "direct_caption",
                "visual_token_id": vt.token_id,
                "visual_paper_id": vt.paper_id,
                "image_path": vt.image_path,
                "text_token_id": f"{vt.token_id}::caption",
                "text": vt.caption,
                "score": 1.0,
                "caption_source": "image_caption",
                "page_number": vt.page_number,
            }
            for vt in visual_tokens
        ]
        self._write_jsonl(out / "direct_pairs.jsonl", direct_pairs)
        self._write_jsonl(out / "topk_pairs.jsonl", topk_pairs)
        self._write_jsonl(out / "image_topk.jsonl", img_topk)

        # Reverse: sentence → top-K images
        txt_topk = self._build_txt_topk(
            visual_tokens, topk_sents, img_vecs_orig, txt_vecs,
            paper_vis_idx, paper_sent_idx, vt_to_embed_idx,
        )
        self._write_jsonl(out / "text_topk.jsonl", txt_topk)

        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_papers": len(set(vt.paper_id for vt in visual_tokens)),
            "n_images": len(visual_tokens),
            "n_sentences": len(sentence_tokens),
            "n_direct_pairs": len(direct_pairs),
            "n_topk_pairs": len(topk_pairs),
            "top_k": self.top_k,
            "embedding_backend": self.model,
            "caption_blend": self.caption_blend,
            "embedding_dim": embedding_dim,
            "embedding_model": model_name,
            "normalized": True,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        logger.info("Wrote metadata.json")

        if topk_pairs:
            self._write_human_check(out, visual_tokens, topk_pairs)

        logger.info(
            "Dataset saved to %s  (images=%d  sentences=%d  direct=%d  topk=%d)",
            out, len(visual_tokens), len(sentence_tokens),
            len(direct_pairs), len(topk_pairs),
        )

    def _build_txt_topk(
        self, visual_tokens, topk_sents, img_vecs, txt_vecs,
        paper_vis_idx, paper_sent_idx, vt_to_embed_idx=None,
    ) -> List[Dict]:
        txt_topk = []
        k = self.top_k
        for paper_id, sent_indices in paper_sent_idx.items():
            vis_indices = paper_vis_idx.get(paper_id, [])
            if not vis_indices:
                continue
            if vt_to_embed_idx is not None:
                embed_indices = [vt_to_embed_idx[i] for i in vis_indices]
            else:
                embed_indices = vis_indices
            v_mat = img_vecs[embed_indices]
            s_mat = txt_vecs[sent_indices]
            paper_sim = v_mat @ s_mat.T
            k_txt = min(k, len(vis_indices))
            for local_j, global_j in enumerate(sent_indices):
                st = topk_sents[global_j]
                col = paper_sim[:, local_j]
                top_idx = np.argpartition(col, -k_txt)[-k_txt:]
                top_idx = top_idx[np.argsort(col[top_idx])[::-1]]
                txt_topk.append({
                    "token_id": st.token_id,
                    "paper_id": st.paper_id,
                    "text": st.text,
                    "source": st.source,
                    "top_k": [
                        {
                            "token_id": visual_tokens[vis_indices[i]].token_id,
                            "paper_id": visual_tokens[vis_indices[i]].paper_id,
                            "figure_id": visual_tokens[vis_indices[i]].figure_id,
                            "image_path": visual_tokens[vis_indices[i]].image_path,
                            "score": float(col[i]),
                        }
                        for i in top_idx
                    ],
                })
        return txt_topk

    def _write_human_check(
        self, out: Path, visual_tokens: List[VisualToken], topk_pairs: List[Dict]
    ) -> None:
        per_vis: dict = defaultdict(list)
        for rec in topk_pairs:
            per_vis[rec["visual_token_id"]].append(rec)

        lines = ["# Human Check Report\n\n"]
        lines.append(
            "Generated by KnowMat ImageTextAligner. "
            "Each section shows one figure with its top-matched sentences.\n\n---\n"
        )
        for vt in visual_tokens:
            records = sorted(per_vis.get(vt.token_id, []), key=lambda r: r.get("rank", 999))
            lines.append(f"\n## {vt.figure_id}  `{vt.paper_id}`\n")
            lines.append(f"- **Image**: `{vt.image_path}`\n")
            lines.append(f"- **Caption**: {vt.caption or '*(none)*'}\n")
            lines.append(f"- **Parsed figure ID**: `{vt.normalized_figure_id or 'n/a'}`\n")
            lines.append(f"- **Page**: {vt.page_number}\n")
            if not records:
                lines.append("\n*(no topk sentences)*\n\n---\n")
                continue
            lines.append("\n| Rank | Cosine | CapTxt | Final | Conf | same | wrong | Figures | Text |\n")
            lines.append("|------|--------|--------|-------|------|------|-------|---------|------|\n")
            for r in records[:self.top_k]:
                text_short = r.get("text", "").replace("|", "\\|").replace("\n", " ")[:80]
                mentioned = ", ".join(r.get("mentioned_figures") or []) or "-"
                lines.append(
                    f"| {r.get('rank')} "
                    f"| {r.get('cosine_score', 0):.4f} "
                    f"| {r.get('caption_text_cosine', 0):.4f} "
                    f"| {r.get('final_score', 0):.4f} "
                    f"| {r.get('confidence')} "
                    f"| {'✓' if r.get('has_same_figure_anchor') else '✗'} "
                    f"| {'✓' if r.get('has_wrong_figure_anchor') else '✗'} "
                    f"| {mentioned} "
                    f"| {text_short} |\n"
                )
            lines.append("\n---\n")

        path = out / "human_check.md"
        path.write_text("".join(lines), encoding="utf-8")
        logger.info("Wrote human_check.md (%d images)", len(visual_tokens))

    @staticmethod
    def _write_jsonl(path: Path, records: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote %s (%d records)", path.name, len(records))
