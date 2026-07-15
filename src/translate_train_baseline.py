"""
translate_train_baseline.py
Task #8: translate-train baseline, the standard alternative to zero-shot
transfer in cross-lingual NER. Instead of training on English and transferring
zero-shot (config E), this trains on the ENTIRE English training set after
word-level lexicon substitution into the target language (every translatable
word swapped via the same MUSE bilingual lexicons used for code-switching,
switch_prob=1.0, not mixed with real English), then evaluates on the real
target-language test set - same evaluation protocol as the rest of the
project, different training data.

Caveat (see load_ner_dataset_translated() in data_processing.py): this is a
simplified PROXY for translate-train, not a full MT-based pipeline. MUSE
gives static word-for-word translations - no word reordering, no morphological
adaptation, no handling of words missing from the lexicon (which stay
English). Real translate-train baselines in the literature use a trained NMT
system plus word-alignment-based label projection, which is a substantially
larger undertaking. This baseline isolates one specific question: does
training on (lexically) target-language surface forms with correct labels,
even without real translation fluency, outperform zero-shot transfer from
English?

Each language trains a fresh model from scratch (config matches B/E: full
backbone fine-tune, layer-wise LR decay, 5 epochs) - independent runs, not
continuing from any other checkpoint.

Run with: python -m src.translate_train_baseline
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_zero_shot import ZeroShotNERTrainer

BASE_CONFIG = {
    "base_model": "xlm-roberta-base",
    "max_seq_length": 128,
    "batch_size": 16,
    "ner_lr": 2e-5,
    "head_lr": 1e-3,
    "layer_decay": 0.95,
    "ner_epochs": 5,
}

TARGET_LANGS = ("tr", "de", "ar")

# Zero-shot baseline (config E) for direct comparison, DEVELOPMENT.md Section 4.4.
ZERO_SHOT_BASELINE = {"tr": 0.7675, "de": 0.7435, "ar": 0.4563}


def main():
    os.makedirs("results", exist_ok=True)
    results = {}

    for lang in TARGET_LANGS:
        trainer = ZeroShotNERTrainer(BASE_CONFIG)
        ckpt_name = f"ner_checkpoint_translate_train_{lang}.pt"
        f1 = trainer.translate_train_finetune(
            target_lang=lang, epochs=BASE_CONFIG["ner_epochs"], checkpoint_name=ckpt_name
        )
        results[lang] = f1
        print(f"[{lang}] translate-train F1: {f1:.4f}")

    out_path = os.path.join("results", "translate_train_results.json")
    with open(out_path, "w") as f:
        json.dump({"translate_train": results, "zero_shot_baseline": ZERO_SHOT_BASELINE}, f, indent=2)

    print(f"\n\n{'='*60}\nTRANSLATE-TRAIN vs. ZERO-SHOT\n{'='*60}")
    print(f"{'Lang':<8}{'Zero-shot (E)':>16}{'Translate-train':>18}{'Delta':>10}")
    for lang in TARGET_LANGS:
        zs = ZERO_SHOT_BASELINE[lang]
        tt = results[lang]
        print(f"{lang:<8}{zs:>16.4f}{tt:>18.4f}{tt-zs:>10.4f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
