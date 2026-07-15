"""
analyze_fragmentation.py
Follow-up analysis: the script-vs-morphology thesis (DEVELOPMENT.md Section 4.6)
was complicated by two new controls (Indonesian, Russian) that don't fit a clean
Latin-script-good / non-Latin-script-bad binary. This script tests an alternative,
more mechanistic hypothesis: XLM-R's SUBWORD FRAGMENTATION RATE per language
(average number of subword tokens per whitespace-delimited word) may be a better
predictor of zero-shot F1 than script identity itself. Script and fragmentation
are correlated (non-Latin scripts often fragment more because XLM-R's 250k-token
vocabulary allocates disproportionately few subword slots to lower-resource
scripts) but are not the same variable - Indonesian's poor performance despite a
Latin script is the concrete case this analysis targets.

For each of the 8 zero-shot-evaluated languages (tr, de, ar + ko, fi, sw, id, ru),
this computes mean subwords-per-word over a sample of WikiANN test-set sentences
using the same xlm-roberta-base tokenizer used throughout the project, and
reports the correlation with observed zero-shot F1.

Run with: python -m src.analyze_fragmentation
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from transformers import AutoTokenizer

LANGS_F1 = {
    "tr": 0.7675, "de": 0.7435, "ar": 0.4563,
    "ko": 0.5086, "fi": 0.7578, "sw": 0.6748,
    "id": 0.4967, "ru": 0.6102,
}

SAMPLE_SIZE = 2000


def mean_subwords_per_word(tokenizer, dataset, sample_size=SAMPLE_SIZE):
    n = min(sample_size, len(dataset))
    total_words = 0
    total_subwords = 0
    for i in range(n):
        words = dataset[i]["tokens"]
        for w in words:
            total_words += 1
            # encode without special tokens to count pure subword pieces
            total_subwords += len(tokenizer.encode(w, add_special_tokens=False))
    return total_subwords / total_words if total_words else float("nan")


def main():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    results = {}
    for lang in LANGS_F1:
        print(f"Processing {lang}...")
        ds = load_dataset("unimelb-nlp/wikiann", lang)["test"]
        frag = mean_subwords_per_word(tokenizer, ds)
        results[lang] = {"f1": LANGS_F1[lang], "subwords_per_word": round(frag, 4)}
        print(f"  [{lang}] F1={LANGS_F1[lang]:.4f}  subwords/word={frag:.4f}")

    # Pearson correlation between fragmentation and F1 (expect negative if
    # fragmentation drives the gap)
    langs = list(results.keys())
    f1_vals = [results[l]["f1"] for l in langs]
    frag_vals = [results[l]["subwords_per_word"] for l in langs]

    n = len(langs)
    mean_f1 = sum(f1_vals) / n
    mean_frag = sum(frag_vals) / n
    cov = sum((f1_vals[i] - mean_f1) * (frag_vals[i] - mean_frag) for i in range(n))
    std_f1 = sum((v - mean_f1) ** 2 for v in f1_vals) ** 0.5
    std_frag = sum((v - mean_frag) ** 2 for v in frag_vals) ** 0.5
    r = cov / (std_f1 * std_frag) if std_f1 and std_frag else float("nan")

    print(f"\nPearson r (subwords/word vs. F1): {r:.4f}")

    out = {"per_language": results, "pearson_r_fragmentation_vs_f1": round(r, 4)}
    out_path = os.path.join("results", "fragmentation_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
