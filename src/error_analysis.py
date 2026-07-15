"""
error_analysis.py
Confusion matrices + qualitative error examples for the zero-shot NER model,
per target language (Turkish, German, Arabic).

Loads the full-pipeline checkpoint (config E, `checkpoints/ner_checkpoint.pt`)
by default. Pass --checkpoint to analyze a different config's checkpoint
(e.g. one of the ablation checkpoints).

Outputs:
  results/confusion_<lang>.png       - 4x4 entity-type confusion matrix heatmap
  results/error_analysis.json        - raw confusion counts + summary stats per language
  results/error_analysis.md          - qualitative error examples (mismatched sentences)

Run with: python -m src.error_analysis
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
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_zero_shot import ZeroShotNERTrainer
from src.data_processing import ID2LABEL, LABEL2ID

ENTITY_TYPES = ["O", "PER", "ORG", "LOC"]
TARGET_LANGS = ("tr", "de", "ar")

BASE_CONFIG = {
    "base_model": "xlm-roberta-base",
    "temp": 0.07,
    "max_seq_length": 128,
    "batch_size": 32,
}


def collapse(tag):
    return tag if tag == "O" else tag.split("-", 1)[1]


def get_predictions(trainer, loader):
    """Mirrors trainer._evaluate but returns the raw tag sequences instead of just F1."""
    trainer.model.eval()
    trainer.ner_head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)
            token_embs = trainer.model.get_token_embeddings(input_ids, attention_mask)
            logits = trainer.ner_head(token_embs)
            preds = logits.argmax(dim=-1)
            for pred_seq, label_seq in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                pred_tags = [ID2LABEL[p] for p, l in zip(pred_seq, label_seq) if l != -100]
                true_tags = [ID2LABEL[l] for l in label_seq if l != -100]
                all_preds.append(pred_tags)
                all_labels.append(true_tags)
    return all_labels, all_preds


def confusion_for_lang(true_seqs, pred_seqs):
    """Token-level confusion matrix collapsed to entity type (O/PER/ORG/LOC)."""
    flat_true = [collapse(t) for seq in true_seqs for t in seq]
    flat_pred = [collapse(t) for seq in pred_seqs for t in seq]
    cm = confusion_matrix(flat_true, flat_pred, labels=ENTITY_TYPES)
    return cm


def plot_confusion(cm, lang, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(ENTITY_TYPES)))
    ax.set_yticks(range(len(ENTITY_TYPES)))
    ax.set_xticklabels(ENTITY_TYPES)
    ax.set_yticklabels(ENTITY_TYPES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Entity-type confusion: {lang} (zero-shot)")
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = cm / row_sums
    for i in range(len(ENTITY_TYPES)):
        for j in range(len(ENTITY_TYPES)):
            ax.text(j, i, f"{cm[i,j]}\n({norm[i,j]*100:.1f}%)",
                     ha="center", va="center",
                     color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def find_qualitative_examples(trainer, lang, n_examples=8, max_scan=500):
    """Re-loads raw (untokenized) test examples so the markdown report can show
    the original surface tokens alongside true/predicted tags."""
    from datasets import load_dataset

    raw = load_dataset("unimelb-nlp/wikiann", lang)["test"].select(range(max_scan))
    examples = []

    trainer.model.eval()
    trainer.ner_head.eval()
    with torch.no_grad():
        for ex in raw:
            tokens = ex["tokens"]
            true_ids = ex["ner_tags"]
            enc = trainer.processor.tokenizer(
                tokens, is_split_into_words=True, truncation=True,
                padding="max_length", max_length=trainer.config.get("max_seq_length", 128),
                return_tensors="pt",
            )
            word_ids = enc.word_ids(batch_index=0)
            input_ids = enc["input_ids"].to(trainer.device)
            attention_mask = enc["attention_mask"].to(trainer.device)

            token_embs = trainer.model.get_token_embeddings(input_ids, attention_mask)
            logits = trainer.ner_head(token_embs)
            pred_ids_seq = logits.argmax(dim=-1)[0].cpu().numpy()

            pred_word_tags = {}
            prev_word_idx = None
            for pos, w_idx in enumerate(word_ids):
                if w_idx is None or w_idx == prev_word_idx:
                    continue
                pred_word_tags[w_idx] = ID2LABEL[pred_ids_seq[pos]]
                prev_word_idx = w_idx

            true_tags = [ID2LABEL[t] for t in true_ids]
            pred_tags = [pred_word_tags.get(i, "O") for i in range(len(tokens))]

            if any(collapse(t) != collapse(p) for t, p in zip(true_tags, pred_tags)):
                examples.append({"tokens": tokens, "true": true_tags, "pred": pred_tags})
            if len(examples) >= n_examples:
                break

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ner_checkpoint.pt",
                         help="Checkpoint filename under checkpoints/ (default: full pipeline, config E)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    trainer = ZeroShotNERTrainer(BASE_CONFIG)
    trainer._load_checkpoint(args.checkpoint, include_head=True)

    summary = {}
    md_lines = [f"# Error Analysis (checkpoint: `{args.checkpoint}`)\n"]

    for lang in TARGET_LANGS:
        print(f"\n=== Error analysis: {lang} ===")
        dataset = trainer.processor.load_ner_dataset(lang)
        test_loader = DataLoader(dataset["test"], batch_size=BASE_CONFIG["batch_size"], num_workers=0)
        true_seqs, pred_seqs = get_predictions(trainer, test_loader)

        cm = confusion_for_lang(true_seqs, pred_seqs)
        png_path = os.path.join("results", f"confusion_{lang}.png")
        plot_confusion(cm, lang, png_path)
        print(f"Saved: {png_path}")

        per_class_recall = {}
        for i, etype in enumerate(ENTITY_TYPES):
            total = cm[i].sum()
            per_class_recall[etype] = float(cm[i, i] / total) if total > 0 else None

        summary[lang] = {
            "confusion_matrix": cm.tolist(),
            "labels": ENTITY_TYPES,
            "per_class_recall": per_class_recall,
        }

        examples = find_qualitative_examples(trainer, lang)
        md_lines.append(f"\n## {lang}\n")
        md_lines.append(f"Per-class recall: {per_class_recall}\n")
        md_lines.append(f"Confusion matrix (rows=true, cols=pred, order={ENTITY_TYPES}):\n```\n{cm}\n```\n")
        md_lines.append(f"\n### Qualitative errors ({len(examples)} examples)\n")
        for k, ex in enumerate(examples, 1):
            md_lines.append(f"\n**Example {k}:**\n")
            md_lines.append("| Token | True | Pred |")
            md_lines.append("|---|---|---|")
            for tok, t, p in zip(ex["tokens"], ex["true"], ex["pred"]):
                marker = " **<-- mismatch**" if collapse(t) != collapse(p) else ""
                md_lines.append(f"| {tok} | {t} | {p}{marker} |")

    with open(os.path.join("results", "error_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join("results", "error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\nSaved: results/error_analysis.json, results/error_analysis.md, results/confusion_<lang>.png")


if __name__ == "__main__":
    main()
