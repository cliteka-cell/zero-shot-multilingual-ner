"""
run_probe.py
Task #14: probe whether the embedding alignment confirmed in Section 7 actually
transfers to NER F1 when the NER head is forced to use it.

Section 7 found contrastive training produces a large, real sentence-embedding
alignment (true/shuffled cosine-sim gap 0.56-0.64). Section 5.4's multi-seed
ablation found no confirmed NER F1 benefit from contrastive training (config C
vs. B). The likely explanation: NER training/eval use get_token_embeddings()
(raw backbone token states) and never touch the projection_head where that
alignment was actually optimized.

Config F here is config C (contrastive + full fine-tune, no code-switch) with
one change: NER training/eval route token embeddings through the projection
head (route_through_projection=True in ner_finetune). This isolates exactly
the variable in question - if F clearly beats C, the projection head WAS the
missing link; if F looks like C (within multi-seed noise, ~0.01), the
explanation doesn't hold and the dissociation has some other cause.

Run with: python -m src.run_probe
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

# Config C's single-seed result, for direct comparison (DEVELOPMENT.md Section 5.2).
CONFIG_C_RESULT = {"tr": 0.7360, "de": 0.7447, "ar": 0.4467}


def main():
    os.makedirs("results", exist_ok=True)
    print(f"\n{'='*70}\nPROBE: config F (contrastive + full fine-tune, routed through projection head)\n{'='*70}")

    trainer = ZeroShotNERTrainer(BASE_CONFIG)
    trainer.contrastive_train(
        target_langs=TARGET_LANGS,
        checkpoint_name="alignment_checkpoint_F_projection_routed.pt",
    )
    trainer.ner_finetune(
        lang="en",
        augment_langs=TARGET_LANGS,
        freeze_backbone=False,
        use_code_switch=False,
        route_through_projection=True,
        checkpoint_name="ner_checkpoint_F_projection_routed.pt",
    )
    results = trainer.zero_shot_eval_multi(
        target_langs=TARGET_LANGS,
        checkpoint_name="ner_checkpoint_F_projection_routed.pt",
    )

    out = {"F_projection_routed": results, "C_baseline_for_comparison": CONFIG_C_RESULT}
    out_path = os.path.join("results", "probe_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    avg_f = sum(results.values()) / len(results)
    avg_c = sum(CONFIG_C_RESULT.values()) / len(CONFIG_C_RESULT)
    print(f"\n\n{'='*70}\nPROBE SUMMARY\n{'='*70}")
    print(f"{'Config':<30}{'tr':>8}{'de':>8}{'ar':>8}{'avg':>8}")
    print(f"{'F (projection-routed)':<30}{results['tr']:>8.4f}{results['de']:>8.4f}{results['ar']:>8.4f}{avg_f:>8.4f}")
    print(f"{'C (baseline, not routed)':<30}{CONFIG_C_RESULT['tr']:>8.4f}{CONFIG_C_RESULT['de']:>8.4f}{CONFIG_C_RESULT['ar']:>8.4f}{avg_c:>8.4f}")
    print(f"{'Delta (F - C)':<30}{results['tr']-CONFIG_C_RESULT['tr']:>8.4f}{results['de']-CONFIG_C_RESULT['de']:>8.4f}{results['ar']-CONFIG_C_RESULT['ar']:>8.4f}{avg_f-avg_c:>8.4f}")
    print(f"\nSaved: {out_path}")
    print("\nNote: config B's multi-seed std (avg 0.0099, Section 5.4) is the relevant noise floor")
    print("for judging whether any delta here is a real effect.")


if __name__ == "__main__":
    main()
