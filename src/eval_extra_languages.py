"""
eval_extra_languages.py
Extends zero-shot evaluation (Section 4.4) to typologically distinct languages
beyond the original tr/de/ar set, using the ALREADY-TRAINED headline checkpoint
(checkpoints/ner_checkpoint.pt, config E) - no retraining needed, this is purely
inference. Languages chosen to stress-test different distances from English and
from the original three:

  - Korean (ko): different script (Hangul), agglutinative like Turkish but
    typologically unrelated to it; tests whether the Section 6 ORG-over-prediction
    pattern is Arabic-specific or a broader "distant from English" effect.
  - Finnish (fi): agglutinative, Latin script, distantly related to English
    (Uralic, not Indo-European) - a middle ground between German (close) and
    Korean/Arabic (distant).
  - Swahili (sw): Bantu, Latin script, noun-class agreement system very
    different from English morphology, geographically/typologically the most
    distant language tested in this project.
  - Indonesian (id): added as a SECOND "distant morphology, Latin script"
    control, deliberately from a different family than Swahili (Austronesian,
    largely isolating morphology vs. Swahili's Bantu noun-class agglutination).
    If the script-not-morphology thesis holds, Indonesian should land in the
    same 0.67-0.83 Latin-script band regardless of family/morphology type.
  - Russian (ru): added as a SECOND "non-Latin script" control, deliberately
    from a family CLOSE to English (Indo-European, Slavic branch) rather than
    a distant one - the sharpest possible test of the script-not-morphology
    thesis. If script alone drives the gap, Russian should underperform the
    Latin-script cluster despite being Indo-European; if family/relatedness is
    what actually matters, Russian (much closer to English than Arabic/Korean)
    should transfer well despite its Cyrillic script.

Run with: python -m src.eval_extra_languages
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

EXTRA_LANGS = ("ko", "fi", "sw", "id", "ru")


def main():
    os.makedirs("results", exist_ok=True)
    trainer = ZeroShotNERTrainer(BASE_CONFIG)
    results = trainer.zero_shot_eval_multi(target_langs=EXTRA_LANGS, checkpoint_name="ner_checkpoint.pt")

    out_path = os.path.join("results", "extra_languages_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
