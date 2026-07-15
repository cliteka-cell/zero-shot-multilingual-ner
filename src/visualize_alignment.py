"""
visualize_alignment.py
Visualizes whether contrastive alignment (Phase 1) actually pulled English and
target-language sentence embeddings together, two ways:

  1. Quantitative: mean cosine similarity of TRUE (en, target) translation pairs,
     compared against a shuffled-pair baseline (same points, mismatched order).
     The shuffled baseline matters because normalized LM sentence embeddings are
     known to be anisotropic (clustered in a narrow cone) - a high *raw* cosine
     similarity can be a property of the embedding space itself, not evidence of
     alignment. The meaningful signal is the GAP between true-pair and
     shuffled-pair similarity.
  2. Qualitative: t-SNE projection of the same embeddings to 2D, before vs. after
     contrastive training, colored by language.

Compares a FRESH pretrained backbone ("before") against the checkpoint saved at
the end of Phase 1 ("after", default: checkpoints/alignment_checkpoint.pt - the
config E / headline run's alignment checkpoint).

Run with: python -m src.visualize_alignment
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contrastive_model import ZeroShotAligner
from src.data_processing import MultiLingualNERPipeline

TARGET_LANGS = ("tr", "de", "ar")
N_SAMPLES = 200
MAX_LEN = 64
COLORS = {"en": "tab:gray", "tr": "tab:blue", "de": "tab:orange", "ar": "tab:green"}


def embed_pairs(model, processor, device, target_lang, n=N_SAMPLES):
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
        en_emb = model(en_enc).cpu().numpy()
        tgt_emb = model(tgt_enc).cpu().numpy()
    return en_emb, tgt_emb


def alignment_stats(en_emb, tgt_emb):
    """Embeddings are already L2-normalized (model.forward applies F.normalize),
    so cosine similarity is just the dot product."""
    true_sim = (en_emb * tgt_emb).sum(axis=1)
    shuffled = np.roll(tgt_emb, 1, axis=0)  # deterministic mismatch, not random
    random_sim = (en_emb * shuffled).sum(axis=1)
    return {
        "true_pair_mean": float(true_sim.mean()),
        "true_pair_std": float(true_sim.std()),
        "shuffled_pair_mean": float(random_sim.mean()),
        "shuffled_pair_std": float(random_sim.std()),
        "alignment_gap": float(true_sim.mean() - random_sim.mean()),
    }


def collect_embeddings(model, processor, device):
    """Returns (points [N,hidden], labels [N] language codes) for en + all target langs."""
    points, labels = [], []
    stats = {}
    for lang in TARGET_LANGS:
        en_emb, tgt_emb = embed_pairs(model, processor, device, lang)
        stats[lang] = alignment_stats(en_emb, tgt_emb)
        points.append(en_emb)
        labels += ["en"] * len(en_emb)
        points.append(tgt_emb)
        labels += [lang] * len(tgt_emb)
    return np.concatenate(points, axis=0), labels, stats


def plot_tsne(ax, points, labels, title):
    proj = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(points)
    labels = np.array(labels)
    for lang in ["en"] + list(TARGET_LANGS):
        mask = labels == lang
        ax.scatter(proj[mask, 0], proj[mask, 1], s=12, alpha=0.6, c=COLORS[lang], label=lang)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--after_checkpoint", default="alignment_checkpoint.pt")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MultiLingualNERPipeline(model_name="xlm-roberta-base", max_length=MAX_LEN)

    print("Embedding with BEFORE model (fresh pretrained XLM-R, no contrastive training)...")
    model_before = ZeroShotAligner("xlm-roberta-base").to(device)
    points_before, labels_before, stats_before = collect_embeddings(model_before, processor, device)
    del model_before
    torch.cuda.empty_cache()

    print(f"Embedding with AFTER model (loaded {args.after_checkpoint})...")
    model_after = ZeroShotAligner("xlm-roberta-base").to(device)
    ckpt_path = os.path.join("checkpoints", args.after_checkpoint)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_after.load_state_dict(state["model"])
    points_after, labels_after, stats_after = collect_embeddings(model_after, processor, device)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_tsne(axes[0], points_before, labels_before, "Before contrastive training")
    plot_tsne(axes[1], points_after, labels_after, "After contrastive training")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Sentence embeddings: English vs. target language (t-SNE)")
    fig.tight_layout()
    out_path = os.path.join("results", "embedding_alignment_tsne.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    summary = {"before": stats_before, "after": stats_after}
    out_json = os.path.join("results", "alignment_metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'Lang':<6}{'Before gap':>14}{'After gap':>14}")
    for lang in TARGET_LANGS:
        print(f"{lang:<6}{stats_before[lang]['alignment_gap']:>14.4f}{stats_after[lang]['alignment_gap']:>14.4f}")
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
