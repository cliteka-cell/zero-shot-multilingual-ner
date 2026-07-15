"""
few_shot_curve.py
Task #7: how quickly does zero-shot NER performance recover as a handful of
LABELED target-language examples become available? Starting from the config E
(full pipeline) checkpoint, fine-tunes independently on k = 10/50/100/500
labeled examples per target language (each k starts fresh from the zero-shot
checkpoint - NOT cumulative), then evaluates on the full test set. k=0 reuses
the already-measured zero-shot baseline (DEVELOPMENT.md Section 4.4).

This answers a practically important question the rest of the project doesn't:
if a deployment can label even a small amount of target-language data, how much
of the gap (especially Arabic's) closes immediately?

Run with: python -m src.few_shot_curve
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_zero_shot import ZeroShotNERTrainer

BASE_CONFIG = {
    "base_model": "xlm-roberta-base",
    "max_seq_length": 128,
}

TARGET_LANGS = ("tr", "de", "ar")
K_VALUES = (10, 50, 100, 500)

# k=0 zero-shot baseline, config E, already measured (DEVELOPMENT.md Section 4.4).
ZERO_SHOT_BASELINE = {"tr": 0.7675, "de": 0.7435, "ar": 0.4563}


def main():
    os.makedirs("results", exist_ok=True)
    curve = {lang: {"0": ZERO_SHOT_BASELINE[lang]} for lang in TARGET_LANGS}

    for lang in TARGET_LANGS:
        for k in K_VALUES:
            trainer = ZeroShotNERTrainer(BASE_CONFIG)
            trainer._load_checkpoint("ner_checkpoint.pt", include_head=True)
            f1 = trainer.few_shot_finetune(lang, k_examples=k)
            curve[lang][str(k)] = f1
            print(f"[{lang}] k={k}: F1={f1:.4f}")

    out_path = os.path.join("results", "few_shot_curve.json")
    with open(out_path, "w") as f:
        json.dump(curve, f, indent=2)

    print(f"\n\n{'='*60}\nFEW-SHOT RECOVERY CURVE\n{'='*60}")
    header = "k".ljust(8) + "".join(lang.rjust(10) for lang in TARGET_LANGS)
    print(header)
    for k in ["0"] + [str(k) for k in K_VALUES]:
        row = k.ljust(8) + "".join(f"{curve[lang][k]:>10.4f}" for lang in TARGET_LANGS)
        print(row)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
