"""
run_multiseed.py
Multi-seed validation for the ablation study's headline finding (DEVELOPMENT.md Section 5):
config B (full backbone fine-tune, no contrastive, no code-switch) scored highest on a single
seed, slightly above the full pipeline (E). The deltas between B/C/D/E were small (-0.013 to
-0.026 avg F1) and flagged as possibly within run-to-run noise.

This script re-runs config B for 3 seeds, each starting from a FRESH pretrained XLM-R, and
reports mean +/- std per language. If the std is comparable to or larger than the ablation
deltas, those deltas should be treated as noise rather than a real effect.

Run with: python -m src.run_multiseed
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_zero_shot import ZeroShotNERTrainer, set_seed

BASE_CONFIG = {
    "base_model": "xlm-roberta-base",
    "temp": 0.07,
    "max_seq_length": 128,
    "batch_size": 16,
    "lr": 2e-5,
    "ner_lr": 2e-5,
    "head_lr": 1e-3,
    "layer_decay": 0.95,
    "contrastive_epochs": 1,
    "ner_epochs": 5,
    "max_parallel_examples": 50000,
    "switch_prob": 0.3,
    "augment_ratio": 0.5,
}

TARGET_LANGS = ("tr", "de", "ar")
SEEDS = (42, 43, 44)


def run_seed(seed):
    print(f"\n{'='*70}\nMULTI-SEED RUN: config B (full-finetune only), seed={seed}\n{'='*70}")
    set_seed(seed)
    trainer = ZeroShotNERTrainer(BASE_CONFIG)
    ckpt_name = f"ner_checkpoint_B_seed{seed}.pt"
    trainer.ner_finetune(
        lang="en",
        augment_langs=TARGET_LANGS,
        freeze_backbone=False,
        use_code_switch=False,
        checkpoint_name=ckpt_name,
    )
    return trainer.zero_shot_eval_multi(target_langs=TARGET_LANGS, checkpoint_name=ckpt_name)


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5


def main():
    os.makedirs("results", exist_ok=True)
    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = run_seed(seed)

    summary = {}
    for lang in TARGET_LANGS:
        values = [per_seed[str(s)][lang] for s in SEEDS]
        mean, std = mean_std(values)
        summary[lang] = {"mean": mean, "std": std, "values": values}

    avg_per_seed = [sum(per_seed[str(s)].values()) / len(TARGET_LANGS) for s in SEEDS]
    avg_mean, avg_std = mean_std(avg_per_seed)
    summary["avg"] = {"mean": avg_mean, "std": avg_std, "values": avg_per_seed}

    out = {"per_seed": per_seed, "summary": summary}
    out_path = os.path.join("results", "multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n\n{'='*70}\nMULTI-SEED SUMMARY (config B, seeds={SEEDS})\n{'='*70}")
    for lang in TARGET_LANGS:
        s = summary[lang]
        print(f"  {lang}: {s['mean']:.4f} +/- {s['std']:.4f}  (values: {[round(v,4) for v in s['values']]})")
    print(f"  avg: {summary['avg']['mean']:.4f} +/- {summary['avg']['std']:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
