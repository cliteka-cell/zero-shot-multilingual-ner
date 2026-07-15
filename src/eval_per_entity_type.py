"""
eval_per_entity_type.py
Entity-type-level factor analysis: does the "no single factor explains the
gap" finding (Sections 4.6-4.8, DEVELOPMENT.md) hold at the finer grain of
individual entity types (PER/ORG/LOC), or does breaking down by entity type
reveal a pattern the aggregate F1 correlations miss?

Motivation: Section 6's error analysis already shows Arabic's weak aggregate
F1 is driven by a SPECIFIC, non-uniform failure - systematic ORG
over-prediction - not a uniform "harder language" effect. If that kind of
entity-specific pattern generalizes across languages, some of the 4 factors
tested at the aggregate level (script, fragmentation, corpus size,
phonological distance) might correlate much better with one entity type than
with others, which aggregate F1 correlation would completely hide.

Re-evaluates the existing config E checkpoint (no retraining) on all 8
zero-shot target languages, capturing per-entity-type F1 from seqeval's
classification_report(output_dict=True), then reports Pearson r for each of
the 4 factors against each entity type's F1 separately.

Run with: python -m src.eval_per_entity_type
"""

import json
import os
import sys

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_zero_shot import ZeroShotNERTrainer

BASE_CONFIG = {
    "base_model": "xlm-roberta-base",
    "max_seq_length": 128,
}

ALL_LANGS = ("tr", "de", "ar", "ko", "fi", "sw", "id", "ru")

# Factor values already established in Sections 4.6-4.8 / 5.2-5.5
SCRIPT_LATIN = {"tr", "de", "fi", "sw", "id"}  # 1 = Latin, 0 = non-Latin
FRAGMENTATION = {
    "tr": 1.7513, "de": 1.6093, "ar": 1.7153, "ko": 2.2124,
    "fi": 1.9546, "sw": 1.5527, "id": 1.5940, "ru": 1.9139,
}
CORPUS_GIB = {
    "tr": 20.9, "de": 66.6, "ar": 28.0, "ko": 54.2,
    "fi": 54.3, "sw": 1.6, "id": 148.3, "ru": 278.0,
}


def evaluate_per_class(trainer, lang):
    from seqeval.metrics import classification_report
    dataset = trainer.processor.load_ner_dataset(lang)
    test_loader = DataLoader(dataset["test"], batch_size=32, num_workers=0)

    import torch
    trainer.model.eval()
    trainer.ner_head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)
            token_embs = trainer.get_ner_embeddings(input_ids, attention_mask)
            logits = trainer.ner_head(token_embs)
            preds = logits.argmax(dim=-1)
            for pred_seq, label_seq in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                pred_tags = [trainer.id2label[p] for p, l in zip(pred_seq, label_seq) if l != -100]
                true_tags = [trainer.id2label[l] for l in label_seq if l != -100]
                all_preds.append(pred_tags)
                all_labels.append(true_tags)

    report = classification_report(all_labels, all_preds, output_dict=True)
    return {
        "PER": report.get("PER", {}).get("f1-score", None),
        "ORG": report.get("ORG", {}).get("f1-score", None),
        "LOC": report.get("LOC", {}).get("f1-score", None),
        "micro_avg": report.get("micro avg", {}).get("f1-score", None),
    }


def pearson_r(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")


def main():
    os.makedirs("results", exist_ok=True)
    trainer = ZeroShotNERTrainer(BASE_CONFIG)
    trainer._load_checkpoint("ner_checkpoint.pt", include_head=True)

    per_lang = {}
    for lang in ALL_LANGS:
        print(f"Evaluating {lang}...")
        per_lang[lang] = evaluate_per_class(trainer, lang)
        print(f"  [{lang}] PER={per_lang[lang]['PER']:.4f} ORG={per_lang[lang]['ORG']:.4f} "
              f"LOC={per_lang[lang]['LOC']:.4f} micro={per_lang[lang]['micro_avg']:.4f}")

    # Load phonological distance results (already computed)
    phon_path = os.path.join("results", "phonological_distance_analysis.json")
    phon_dist = {}
    if os.path.exists(phon_path):
        with open(phon_path) as f:
            phon_data = json.load(f)
        phon_dist = {l: v["phon_distance_from_english"] for l, v in phon_data["per_language"].items()}

    langs = list(ALL_LANGS)
    correlations = {}
    for entity_type in ["PER", "ORG", "LOC", "micro_avg"]:
        f1s = [per_lang[l][entity_type] for l in langs]
        correlations[entity_type] = {
            "script_latin_binary": round(pearson_r([1 if l in SCRIPT_LATIN else 0 for l in langs], f1s), 4),
            "fragmentation": round(pearson_r([FRAGMENTATION[l] for l in langs], f1s), 4),
            "corpus_size_gib": round(pearson_r([CORPUS_GIB[l] for l in langs], f1s), 4),
        }
        if phon_dist:
            correlations[entity_type]["phon_distance"] = round(
                pearson_r([phon_dist[l] for l in langs], f1s), 4
            )

    print("\n--- Pearson r by entity type and factor ---")
    for entity_type, corrs in correlations.items():
        print(f"  {entity_type}: {corrs}")

    out = {"per_language_per_class_f1": per_lang, "pearson_r_by_entity_type": correlations}
    out_path = os.path.join("results", "per_entity_type_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
