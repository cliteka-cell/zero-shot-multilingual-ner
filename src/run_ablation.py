"""
run_ablation.py
4-way ablation isolating the contribution of each pipeline lever, against the
full-pipeline result already recorded in DEVELOPMENT.md Section 4:

  (A) baseline       - frozen backbone, no contrastive, no code-switch
                        (mirrors the original prototype's approach)
  (B) full-finetune  - full backbone fine-tune, no contrastive, no code-switch
  (C) +contrastive   - full backbone fine-tune + contrastive alignment, no code-switch
  (D) +code-switch   - full backbone fine-tune + code-switch augmentation, no contrastive
  (E) full pipeline  - full backbone fine-tune + contrastive + code-switch
                        (NOT re-run here; reuses the completed full-pipeline run's
                        results since nothing about that code path has changed)

Each of A-D trains from a FRESH pretrained XLM-R (a new ZeroShotNERTrainer per
config), so no config is contaminated by a previous config's fine-tuning.
Results are saved to results/ablation_results.json.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_zero_shot import ZeroShotNERTrainer

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

ABLATIONS = [
    {"name": "A_baseline_frozen",    "freeze_backbone": True,  "use_contrastive": False, "use_code_switch": False},
    {"name": "B_full_finetune_only", "freeze_backbone": False, "use_contrastive": False, "use_code_switch": False},
    {"name": "C_plus_contrastive",   "freeze_backbone": False, "use_contrastive": True,  "use_code_switch": False},
    {"name": "D_plus_code_switch",   "freeze_backbone": False, "use_contrastive": False, "use_code_switch": True},
]

# E (full pipeline) was already run end-to-end - see DEVELOPMENT.md Section 4.
# Reused here rather than re-run to save ~30-40 min of redundant compute.
FULL_PIPELINE_RESULT = {"tr": 0.7675, "de": 0.7435, "ar": 0.4563}


def run_config(cfg):
    print(f"\n{'='*70}\nABLATION CONFIG: {cfg['name']}\n{'='*70}")
    trainer = ZeroShotNERTrainer(BASE_CONFIG)

    if cfg["use_contrastive"]:
        trainer.contrastive_train(
            target_langs=TARGET_LANGS,
            checkpoint_name=f"alignment_checkpoint_{cfg['name']}.pt",
        )

    ckpt_name = f"ner_checkpoint_{cfg['name']}.pt"
    trainer.ner_finetune(
        lang="en",
        augment_langs=TARGET_LANGS,
        freeze_backbone=cfg["freeze_backbone"],
        use_code_switch=cfg["use_code_switch"],
        checkpoint_name=ckpt_name,
    )
    return trainer.zero_shot_eval_multi(target_langs=TARGET_LANGS, checkpoint_name=ckpt_name)


def main():
    os.makedirs("results", exist_ok=True)
    all_results = {"E_full_pipeline": FULL_PIPELINE_RESULT}

    for cfg in ABLATIONS:
        all_results[cfg["name"]] = run_config(cfg)

    out_path = os.path.join("results", "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*70}\nABLATION SUMMARY\n{'='*70}")
    print(f"{'Config':<25}{'tr':>8}{'de':>8}{'ar':>8}{'avg':>8}")
    for name, res in all_results.items():
        avg = sum(res.values()) / len(res)
        print(f"{name:<25}{res['tr']:>8.4f}{res['de']:>8.4f}{res['ar']:>8.4f}{avg:>8.4f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
