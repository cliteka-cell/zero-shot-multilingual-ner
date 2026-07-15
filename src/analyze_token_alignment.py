"""
analyze_token_alignment.py
Extends the Section 7 alignment-gap methodology (visualize_alignment.py) from
CLS-level projected embeddings to RAW BACKBONE TOKEN representations - the
representations NER training/eval actually reads (get_token_embeddings(), not
forward()/the projection head).

Motivation: Section 7 shows contrastive training produces a large true-pair
vs. shuffled-pair alignment gap at the CLS+projection-head level (0.558-0.636)
that does NOT translate into improved NER F1 (Section 5's ablation). The
standing hypothesis (DEVELOPMENT.md Section 7.4) is that this is because the
projection head aligns pooled [CLS] sentence geometry, which may not transfer
to the raw per-token backbone states NER actually classifies. That hypothesis
was previously only supported indirectly (the projection-head routing probe
in run_probe.py shows making the projection head REACHABLE by NER doesn't
help - but that doesn't measure whether the raw token representations
themselves shifted at all).

This script measures that directly: mean-pooled RAW backbone token embeddings
(get_token_embeddings(), excluding padding/special tokens) for the same
OPUS-100 (English, target) sentence pairs used in Section 7, comparing a
fresh pretrained backbone ("before") against the Phase-1 alignment checkpoint
("after"). If the token-level alignment gap after training is comparable in
magnitude to the CLS-level gap, that would REFUTE the "alignment doesn't
reach token representations" hypothesis (token reps did shift, so something
else must explain no NER benefit). If the token-level gap is much smaller,
that CONFIRMS the standing hypothesis with a direct measurement rather than
an indirect probe.

Run with: python -m src.analyze_token_alignment
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contrastive_model import ZeroShotAligner
from src.data_processing import MultiLingualNERPipeline

TARGET_LANGS = ("tr", "de", "ar")
N_SAMPLES = 200
MAX_LEN = 64


def mean_pool_tokens(token_embs, attention_mask):
    """Mean-pool over real (non-padding) tokens. token_embs: [B, L, H], attention_mask: [B, L]."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (token_embs * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_pairs_token_level(model, processor, device, target_lang, n=N_SAMPLES):
    parallel = processor.load_parallel_corpus("en", target_lang)
    subset = parallel["train"].select(range(n))
    en_texts = [ex["translation"]["en"] for ex in subset]
    tgt_texts = [ex["translation"][target_lang] for ex in subset]

    model.eval()
    with torch.no_grad():
        en_enc = processor.tokenizer(
            en_texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
        ).to(device)
        tgt_enc = processor.tokenizer(
            tgt_texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
        ).to(device)

        en_tok = model.get_token_embeddings(en_enc["input_ids"], en_enc["attention_mask"])
        tgt_tok = model.get_token_embeddings(tgt_enc["input_ids"], tgt_enc["attention_mask"])

        en_pooled = mean_pool_tokens(en_tok, en_enc["attention_mask"])
        tgt_pooled = mean_pool_tokens(tgt_tok, tgt_enc["attention_mask"])

        # L2-normalize for cosine similarity (raw backbone states are not
        # pre-normalized, unlike forward()'s projected CLS output)
        en_pooled = torch.nn.functional.normalize(en_pooled, dim=1).cpu().numpy()
        tgt_pooled = torch.nn.functional.normalize(tgt_pooled, dim=1).cpu().numpy()

    return en_pooled, tgt_pooled


def alignment_stats(en_emb, tgt_emb):
    true_sim = (en_emb * tgt_emb).sum(axis=1)
    shuffled = np.roll(tgt_emb, 1, axis=0)
    random_sim = (en_emb * shuffled).sum(axis=1)
    return {
        "true_pair_mean": float(true_sim.mean()),
        "true_pair_std": float(true_sim.std()),
        "shuffled_pair_mean": float(random_sim.mean()),
        "shuffled_pair_std": float(random_sim.std()),
        "alignment_gap": float(true_sim.mean() - random_sim.mean()),
    }


def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MultiLingualNERPipeline(model_name="xlm-roberta-base", max_length=MAX_LEN)

    print("Token-level embeddings with BEFORE model (fresh pretrained, no contrastive training)...")
    model_before = ZeroShotAligner("xlm-roberta-base").to(device)
    stats_before = {}
    for lang in TARGET_LANGS:
        en_emb, tgt_emb = embed_pairs_token_level(model_before, processor, device, lang)
        stats_before[lang] = alignment_stats(en_emb, tgt_emb)
        print(f"  [before, token-level] {lang}: gap={stats_before[lang]['alignment_gap']:.4f}")
    del model_before
    torch.cuda.empty_cache()

    print("Token-level embeddings with AFTER model (checkpoints/alignment_checkpoint.pt)...")
    model_after = ZeroShotAligner("xlm-roberta-base").to(device)
    state = torch.load(os.path.join("checkpoints", "alignment_checkpoint.pt"), map_location=device, weights_only=True)
    model_after.load_state_dict(state["model"])
    stats_after = {}
    for lang in TARGET_LANGS:
        en_emb, tgt_emb = embed_pairs_token_level(model_after, processor, device, lang)
        stats_after[lang] = alignment_stats(en_emb, tgt_emb)
        print(f"  [after, token-level] {lang}: gap={stats_after[lang]['alignment_gap']:.4f}")

    # CLS-level gaps from Section 7 (results/alignment_metrics.json), for direct comparison
    cls_level_path = os.path.join("results", "alignment_metrics.json")
    cls_level = None
    if os.path.exists(cls_level_path):
        with open(cls_level_path) as f:
            cls_level = json.load(f)

    summary = {"before_token_level": stats_before, "after_token_level": stats_after}
    if cls_level:
        summary["cls_level_for_comparison"] = cls_level

    out_path = os.path.join("results", "token_alignment_metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'Lang':<6}{'Token gap (after)':>20}{'CLS gap (after)':>20}")
    for lang in TARGET_LANGS:
        token_gap = stats_after[lang]["alignment_gap"]
        cls_gap = cls_level["after"][lang]["alignment_gap"] if cls_level else float("nan")
        print(f"{lang:<6}{token_gap:>20.4f}{cls_gap:>20.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
