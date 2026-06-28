"""DatasetBuilder: produce a sci-align-compatible alignment dataset from OCR outputs.

Supports two input modes:
  1. Directory mode: scan a root dir for paper subdirs, each containing a
     compatible OCR content list. Converts each paper to ocr_items, then embeds
     + aligns + exports.
  2. Direct mode: pass pre-converted ocr_items dicts to build_from_ocr_items().

Output layout (same as sci-align)
----------------------------------
  <output_dir>/
    image_units.jsonl         metadata for every visual token
    sentence_units.jsonl      metadata for every sentence token
    image_embeddings.npy      float32 [N_img, D], L2-normalised
    sentence_embeddings.npy   float32 [N_txt, D], L2-normalised
    embedding_index.json      maps token_ids → embedding row indices
    direct_pairs.jsonl        image ↔ own caption, score=1.0
    topk_pairs.jsonl          image → top-K body sentences (reranked)
    image_topk.jsonl          legacy: one line per image
    text_topk.jsonl           legacy: one line per sentence
    metadata.json             summary stats
    human_check.md            human-readable table per image

Usage
-----
    from ppmatAgent.knowmat.image_text_alignment.dataset_builder import DatasetBuilder, DatasetBuildConfig

    cfg = DatasetBuildConfig(
        input_dir="/path/to/ocr_outputs",
        output_dir="/path/to/dataset",
        model="clip",
        caption_blend=0.3,
    )
    result = DatasetBuilder(cfg).build()
    print(result)
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Reranking weights (same as aligner.py and sci-align)
FIG_MATCH_BONUS = 0.20
PAGE_NEAR_BONUS = 0.05
WRONG_FIG_PENALTY = 0.15


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class DatasetBuildConfig:
    input_dir: str
    output_dir: str
    model: str = "clip"
    device: str = "cpu"
    top_k: int = 5
    min_score: float = 0.0
    batch_size: int = 32
    caption_blend: float = 0.0   # 0 = pure image; 0.3 = 70% img + 30% caption text
    save_embeddings: bool = True


@dataclass
class DatasetBuildResult:
    n_papers: int = 0
    n_images: int = 0
    n_sentences: int = 0
    n_direct_pairs: int = 0
    n_topk_pairs: int = 0
    output_dir: str = ""
    elapsed: float = 0.0
    errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"DatasetBuildResult("
            f"papers={self.n_papers}, images={self.n_images}, "
            f"sentences={self.n_sentences}, direct={self.n_direct_pairs}, "
            f"topk={self.n_topk_pairs}, elapsed={self.elapsed:.1f}s, "
            f"errors={len(self.errors)})"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_content_list(paper_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """Return (content_list_path, md_path, paper_id) or (None, None, None)."""
    auto_dir = paper_dir / "auto"
    search_dir = auto_dir if auto_dir.is_dir() else paper_dir

    cl_files = sorted(search_dir.glob("*_content_list.json"))
    cl_files = [f for f in cl_files if "_content_list_v2" not in f.name] or cl_files
    if not cl_files:
        return None, None, None

    cl_path = cl_files[0]
    paper_id = cl_path.stem.replace("_content_list", "")

    md_files = sorted(search_dir.glob("*.md"))
    md_path = md_files[0] if md_files else None

    return cl_path, md_path, paper_id


def _confidence(final: float, has_same: bool) -> str:
    if has_same or final >= 0.6:
        return "high"
    if final >= 0.4:
        return "medium"
    return "low"


def _compute_final_score(cosine: float, vis_all_fids: List[str],
                          vis_page: Optional[int], sent_figs: List[str],
                          sent_page: Optional[int]) -> Tuple[float, Optional[str], bool, bool]:
    bonus = 0.0
    matched_fig = None
    has_same = False
    has_wrong = False

    if sent_figs:
        common = [f for f in sent_figs if f in vis_all_fids]
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


# ── JSONL writer ───────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d records)", path.name, len(records))


# ── DatasetBuilder ─────────────────────────────────────────────────────────────

class DatasetBuilder:
    """Build a sci-align-compatible dataset from PaddleOCR/compatible OCR outputs."""

    def __init__(self, config: DatasetBuildConfig) -> None:
        self.cfg = config

    # ── public API ─────────────────────────────────────────────────────────────

    def build(self) -> DatasetBuildResult:
        """
        Scan input_dir for paper subdirs, convert + embed + export.

        input_dir can be:
          - A single-paper dir (contains *_content_list.json directly)
          - A multi-paper root dir (each subdir is a paper)
        """
        t0 = time.time()
        cfg = self.cfg
        result = DatasetBuildResult(output_dir=cfg.output_dir)
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        input_path = Path(cfg.input_dir)
        is_single = (input_path / "auto").is_dir() or bool(
            list(input_path.glob("*_content_list.json"))
        )
        if is_single:
            paper_dirs = [input_path]
        else:
            paper_dirs = sorted(
                p for p in input_path.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )

        if not paper_dirs:
            logger.error("No paper directories found under %s", cfg.input_dir)
            return result

        # Stage 1: locate + convert papers → ocr_items
        logger.info("[1/4] Converting %d paper dir(s) to ocr_items", len(paper_dirs))
        # Compatibility path for historical content_list exports. PaddleOCR API
        # and local PaddleOCR flows should prefer build_from_ocr_items().
        from ppmatAgent.knowmat.pdf.mineru_result_converter import convert_mineru_to_knowmat
        from ppmatAgent.knowmat.image_text_alignment.tokenizer import OcrItemTokenizer

        all_visual = []
        all_sentences = []
        tok = OcrItemTokenizer()

        for pdir in paper_dirs:
            cl_path, md_path, paper_id = _find_content_list(pdir)
            if cl_path is None:
                msg = f"No content_list.json in {pdir}"
                logger.warning(msg)
                result.errors.append(msg)
                continue
            try:
                with open(cl_path) as f:
                    content_list = json.load(f)
                full_md = md_path.read_text(encoding="utf-8") if md_path else ""
                _, _, ocr_items = convert_mineru_to_knowmat(
                    content_list=content_list,
                    full_md=full_md,
                    pdf_path="",
                    extracted_dir=cl_path.parent,
                    figures_dest=out / "figures",
                )
                vis, sents = tok.tokenize(ocr_items, paper_id, str(out))
                all_visual.extend(vis)
                all_sentences.extend(sents)
                logger.info("  %s: %d images, %d sentences", paper_id, len(vis), len(sents))
            except Exception as exc:
                msg = f"Failed for {paper_id}: {exc}"
                logger.warning(msg)
                result.errors.append(msg)

        result.n_images = len(all_visual)
        result.n_sentences = len(all_sentences)
        result.n_papers = len(paper_dirs) - len(result.errors)

        if not all_visual:
            logger.warning("No visual tokens — aborting.")
            return result

        # Stage 2: embed
        result = self._embed_and_build(all_visual, all_sentences, out, result)
        result.elapsed = time.time() - t0
        logger.info(
            "=== Build done | %s ===", result
        )
        return result

    def build_from_ocr_items(
        self,
        ocr_items_by_paper: Dict[str, List[Dict[str, Any]]],
    ) -> DatasetBuildResult:
        """
        Build dataset from pre-converted ocr_items.

        Parameters
        ----------
        ocr_items_by_paper : dict
            {paper_id: [ocr_item, ...]}
        """
        t0 = time.time()
        cfg = self.cfg
        result = DatasetBuildResult(output_dir=cfg.output_dir)
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        from ppmatAgent.knowmat.image_text_alignment.tokenizer import OcrItemTokenizer
        tok = OcrItemTokenizer()
        all_visual = []
        all_sentences = []

        for paper_id, ocr_items in ocr_items_by_paper.items():
            try:
                vis, sents = tok.tokenize(ocr_items, paper_id, str(out))
                all_visual.extend(vis)
                all_sentences.extend(sents)
                logger.info("  %s: %d images, %d sentences", paper_id, len(vis), len(sents))
            except Exception as exc:
                msg = f"Tokenize failed for {paper_id}: {exc}"
                logger.warning(msg)
                result.errors.append(msg)

        result.n_images = len(all_visual)
        result.n_sentences = len(all_sentences)
        result.n_papers = len(ocr_items_by_paper)

        if not all_visual:
            logger.warning("No visual tokens — aborting.")
            return result

        result = self._embed_and_build(all_visual, all_sentences, out, result)
        result.elapsed = time.time() - t0
        return result

    # ── internal ───────────────────────────────────────────────────────────────

    def _embed_and_build(self, all_visual, all_sentences, out: Path,
                          result: DatasetBuildResult) -> DatasetBuildResult:
        cfg = self.cfg

        # Stage 2: embed
        logger.info("[2/4] Embedding with %s on %s", cfg.model, cfg.device)
        from ppmatAgent.knowmat.image_text_alignment.embeddings import get_embedding

        enc = get_embedding(cfg.model, device=cfg.device)
        model_name = getattr(enc, "model_name", cfg.model)
        embedding_dim = getattr(enc, "embedding_dim", 512)

        image_paths = [vt.image_path for vt in all_visual]
        img_vecs = enc.embed_image(image_paths, batch_size=cfg.batch_size)
        img_vecs = img_vecs / (np.linalg.norm(img_vecs, axis=1, keepdims=True) + 1e-8)

        captions = [vt.caption for vt in all_visual]
        cap_vecs = enc.embed_text(captions, batch_size=cfg.batch_size)
        cap_vecs = cap_vecs / (np.linalg.norm(cap_vecs, axis=1, keepdims=True) + 1e-8)

        if cfg.caption_blend > 0.0:
            logger.info("  Applying caption_blend alpha=%.2f", cfg.caption_blend)
            img_vecs = (1.0 - cfg.caption_blend) * img_vecs + cfg.caption_blend * cap_vecs
            img_vecs = img_vecs / (np.linalg.norm(img_vecs, axis=1, keepdims=True) + 1e-8)

        txt_vecs = None
        if all_sentences:
            texts = [s.text for s in all_sentences]
            txt_vecs = enc.embed_text(texts, batch_size=cfg.batch_size)
            txt_vecs = txt_vecs / (np.linalg.norm(txt_vecs, axis=1, keepdims=True) + 1e-8)

        embedding_dim = int(enc.embedding_dim) if enc.embedding_dim else img_vecs.shape[1]

        # Stage 3: build pairs
        logger.info("[3/4] Building pairs (top_k=%d)", cfg.top_k)
        direct_pairs, topk_pairs, img_topk, txt_topk = self._build_pairs(
            all_visual, all_sentences, img_vecs, cap_vecs, txt_vecs
        )
        result.n_direct_pairs = len(direct_pairs)
        result.n_topk_pairs = len(topk_pairs)

        # Stage 4: export
        logger.info("[4/4] Exporting to %s", cfg.output_dir)
        self._export(
            out=out,
            all_visual=all_visual,
            all_sentences=all_sentences,
            img_vecs=img_vecs,
            txt_vecs=txt_vecs,
            direct_pairs=direct_pairs,
            topk_pairs=topk_pairs,
            img_topk=img_topk,
            txt_topk=txt_topk,
            model_name=model_name,
            embedding_dim=embedding_dim,
            result=result,
        )
        return result

    def _build_pairs(self, all_visual, all_sentences, img_vecs, cap_vecs,
                      txt_vecs) -> Tuple[List, List, List, List]:
        cfg = self.cfg

        # Direct pairs: image ↔ own caption, score=1.0
        direct_pairs = []
        for vt in all_visual:
            direct_pairs.append({
                "pair_type": "direct_caption",
                "visual_token_id": vt.token_id,
                "visual_paper_id": vt.paper_id,
                "image_path": vt.image_path,
                "text_token_id": f"{vt.token_id}::caption",
                "text": vt.caption,
                "score": 1.0,
                "caption_source": "image_caption",
                "page_number": vt.page_number,
            })

        topk_pairs: List[Dict] = []
        img_topk: List[Dict] = []
        txt_topk: List[Dict] = []

        if txt_vecs is None:
            return direct_pairs, topk_pairs, img_topk, txt_topk

        # TopK pool: exclude caption_text sentences
        topk_sent_pos = [i for i, s in enumerate(all_sentences) if s.source != "caption_text"]
        topk_sents = [all_sentences[i] for i in topk_sent_pos]
        topk_txt_vecs = (
            txt_vecs[np.array(topk_sent_pos)]
            if topk_sent_pos
            else np.zeros((0, txt_vecs.shape[1]), dtype=txt_vecs.dtype)
        )
        logger.info(
            "  TopK pool: %d body sentences (excluded %d caption_text)",
            len(topk_sents), len(all_sentences) - len(topk_sents),
        )

        # Per-paper sub-matrices (avoids cross-paper contamination)
        paper_vis_idx: dict = defaultdict(list)
        paper_sent_idx: dict = defaultdict(list)
        for i, vt in enumerate(all_visual):
            paper_vis_idx[vt.paper_id].append(i)
        for j, st in enumerate(topk_sents):
            paper_sent_idx[st.paper_id].append(j)

        k = cfg.top_k

        for paper_id, vis_indices in paper_vis_idx.items():
            sent_indices = paper_sent_idx.get(paper_id, [])
            if not sent_indices:
                continue

            v_mat = img_vecs[vis_indices]
            c_mat = cap_vecs[vis_indices]
            s_mat = topk_txt_vecs[sent_indices]
            paper_sim = v_mat @ s_mat.T
            paper_cap_sim = c_mat @ s_mat.T
            paper_k = min(k, len(sent_indices))

            for local_i, global_i in enumerate(vis_indices):
                vt = all_visual[global_i]
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
                    entry = (final, cosine, cap_cosine, global_j, st, matched_fig, has_same, has_wrong)
                    if has_same:
                        anchor_scored.append(entry)
                    elif has_wrong:
                        wrong_scored.append(entry)
                    else:
                        clean_scored.append(entry)

                for lst in (anchor_scored, clean_scored, wrong_scored):
                    lst.sort(key=lambda x: x[0], reverse=True)
                selected = (anchor_scored + clean_scored + wrong_scored)[:paper_k]

                top_sents_legacy = []
                for rank, (final, cosine, cap_cosine, global_j, st,
                            matched_fig, has_same, has_wrong) in enumerate(selected, 1):
                    if final < cfg.min_score:
                        continue
                    conf = _confidence(final, has_same)
                    topk_pairs.append({
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
                        "token_id": st.token_id,
                        "paper_id": st.paper_id,
                        "text": st.text,
                        "source": st.source,
                        "cosine_score": round(cosine, 6),
                        "final_score": round(final, 6),
                        "score": round(final, 6),
                        "rank": rank,
                        "confidence": conf,
                    })
                img_topk.append({
                    "token_id": vt.token_id,
                    "paper_id": vt.paper_id,
                    "figure_id": vt.figure_id,
                    "image_path": vt.image_path,
                    "caption": vt.caption,
                    "top_k": top_sents_legacy,
                })

        # Reverse: sentence → top-K images
        for paper_id, sent_indices in paper_sent_idx.items():
            vis_indices = paper_vis_idx.get(paper_id, [])
            if not vis_indices:
                continue
            v_mat = img_vecs[vis_indices]
            s_mat = topk_txt_vecs[sent_indices]
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
                            "token_id": all_visual[vis_indices[i]].token_id,
                            "paper_id": all_visual[vis_indices[i]].paper_id,
                            "figure_id": all_visual[vis_indices[i]].figure_id,
                            "image_path": all_visual[vis_indices[i]].image_path,
                            "score": float(col[i]),
                        }
                        for i in top_idx
                    ],
                })

        return direct_pairs, topk_pairs, img_topk, txt_topk

    def _export(self, out: Path, all_visual, all_sentences, img_vecs, txt_vecs,
                 direct_pairs, topk_pairs, img_topk, txt_topk,
                 model_name, embedding_dim, result: DatasetBuildResult) -> None:
        cfg = self.cfg

        # Unit metadata
        _write_jsonl(out / "image_units.jsonl", [vt.to_dict() for vt in all_visual])
        _write_jsonl(out / "sentence_units.jsonl", [st.to_dict() for st in all_sentences])

        # Embeddings
        if cfg.save_embeddings:
            np.save(str(out / "image_embeddings.npy"), img_vecs.astype(np.float32))
            logger.info("Wrote image_embeddings.npy  shape=%s", list(img_vecs.shape))

            sent_vecs = txt_vecs if txt_vecs is not None else np.empty((0, embedding_dim), dtype=np.float32)
            np.save(str(out / "sentence_embeddings.npy"), sent_vecs.astype(np.float32))
            logger.info("Wrote sentence_embeddings.npy  shape=%s", list(sent_vecs.shape))

            index = {
                "image_embeddings": {
                    "path": "image_embeddings.npy",
                    "shape": list(img_vecs.shape),
                    "token_ids": [vt.token_id for vt in all_visual],
                },
                "sentence_embeddings": {
                    "path": "sentence_embeddings.npy",
                    "shape": list(sent_vecs.shape),
                    "token_ids": [st.token_id for st in all_sentences],
                },
                "model": {
                    "name": model_name,
                    "embedding_dim": embedding_dim,
                    "normalized": True,
                    "caption_blend": cfg.caption_blend,
                },
            }
            (out / "embedding_index.json").write_text(
                json.dumps(index, indent=2, ensure_ascii=False)
            )
            logger.info("Wrote embedding_index.json")

        # Pairs
        _write_jsonl(out / "direct_pairs.jsonl", direct_pairs)
        _write_jsonl(out / "topk_pairs.jsonl", topk_pairs)
        _write_jsonl(out / "image_topk.jsonl", img_topk)
        _write_jsonl(out / "text_topk.jsonl", txt_topk)

        # Metadata
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_papers": result.n_papers,
            "n_images": result.n_images,
            "n_sentences": result.n_sentences,
            "n_direct_pairs": result.n_direct_pairs,
            "n_topk_pairs": result.n_topk_pairs,
            "top_k": cfg.top_k,
            "embedding_backend": cfg.model,
            "caption_blend": cfg.caption_blend,
            "n_image_embeddings": len(all_visual) if cfg.save_embeddings else 0,
            "n_sentence_embeddings": len(all_sentences) if cfg.save_embeddings else 0,
            "embedding_dim": embedding_dim,
            "embedding_model": model_name,
            "normalized": True,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        logger.info("Wrote metadata.json")

        # human_check.md
        if topk_pairs:
            self._write_human_check(out, all_visual, topk_pairs)

    def _write_human_check(self, out: Path, all_visual, topk_pairs: List[Dict]) -> None:
        per_vis: dict = defaultdict(list)
        for rec in topk_pairs:
            per_vis[rec["visual_token_id"]].append(rec)

        lines = ["# Human Check Report\n\n"]
        lines.append(
            "Generated by KnowMat DatasetBuilder. "
            "Each section shows one figure with its top-matched sentences.\n\n---\n"
        )

        for vt in all_visual:
            records = sorted(per_vis.get(vt.token_id, []), key=lambda r: r.get("rank", 999))
            lines.append(f"\n## {vt.figure_id}  `{vt.paper_id}`\n")
            lines.append(f"- **Image**: `{vt.image_path}`\n")
            lines.append(f"- **Caption**: {vt.caption or '*(none)*'}\n")
            lines.append(f"- **Parsed figure ID**: `{vt.normalized_figure_id or 'n/a'}`\n")
            lines.append(f"- **Page**: {vt.page_number}\n")

            if not records:
                lines.append("\n*(no topk sentences)*\n\n---\n")
                continue

            lines.append(f"\n| Rank | Cosine | CapTxt | Final | Conf | same | wrong | Figures | Text |\n")
            lines.append("|------|--------|--------|-------|------|------|-------|---------|------|\n")
            for r in records[:self.cfg.top_k]:
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
        logger.info("Wrote human_check.md (%d images)", len(all_visual))
