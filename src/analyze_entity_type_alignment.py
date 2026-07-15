"""
analyze_entity_type_alignment.py
Bridges the two most novel analyses added on top of the Lauscher et al. (2020)
replication: Section 5.6 (entity-type-level factor correlations, showing PER
transfer correlates strongly with script while ORG/LOC do not) and Section 6.2
(token-level contrastive alignment measurement, averaged uniformly across all
tokens). This script asks the question those two sections leave open: does the
token-level alignment gap ITSELF differ by entity type? If PER-tagged token
representations show a larger true-pair-vs-shuffled-pair alignment gap than
ORG/LOC-tagged tokens after contrastive training, that would be a genuine
mechanistic bridge - PER may transfer better partly because PER-relevant
representations are more strongly pulled together by the contrastive
objective, while ORG/LOC representations are not.

Method: OPUS-100 (English, target) sentence pairs are tokenized once. The
FULLY-TRAINED config E model (backbone + NER head, the best available
zero-shot tagger) predicts a PER/ORG/LOC/O label for every subword position
on BOTH sides of each pair - this is purely a labeling tool, decoupled from
the embeddings being measured. Separately, the "before" (fresh pretrained)
and "after" (checkpoints/alignment_checkpoint.pt, Phase-1-only, matching
Section 6.2) models provide RAW backbone token embeddings for those same
positions. For each entity type, subword positions carrying that predicted
label are mean-pooled per sentence (English side and target side separately);
a sentence pair contributes to an entity type's statistics only if BOTH sides
have at least one token of that type. The true-pair-vs-shuffled-pair cosine
similarity gap (identical methodology to visualize_alignment.py and
analyze_token_alignment.py) is then computed per entity type per language.

Caveat: entity-type labels are model-PREDICTED (from the zero-shot NER
system itself, most reliable for tr/de, least reliable for ar where F1 is
~0.46), not gold - there is no gold parallel entity-annotated data available
for this cross-lingual comparison. This is an exploratory measurement, not a
confirmatory one; see PAPER.md Limitations for the full caveat.

Run with: python -m src.analyze_entity_type_alignment
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contrastive_model import ZeroShotAligner, NERClassifier
from src.data_processing import MultiLingualNERPipeline

TARGET_LANGS = ("tr", "de", "ar")
N_SAMPLES = 1000
MAX_LEN = 64
ENTITY_TYPES = ("PER", "ORG", "LOC", "O")

# LABEL2ID/ID2LABEL from data_processing.py (duplicated here to avoid a load-time dependency)
ID2LABEL = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC"}


def collapse_label(label):
    if label == "O":
        return "O"
    return label[2:]  # strip B-/I- prefix


def load_tagger(device):
    """Loads the fully-trained config E model (backbone + NER head) purely as
    a labeling tool - not the embeddings source."""
    model = ZeroShotAligner("xlm-roberta-base").to(device)
    head = NERClassifier(hidden_size=model.backbone.config.hidden_size, num_labels=7).to(device)
    state = torch.load(os.path.join("checkpoints", "ner_checkpoint.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    head.load_state_dict(state["ner_head"])
    model.eval()
    head.eval()
    return model, head


def predict_tags(tagger_model, tagger_head, input_ids, attention_mask):
    """Returns [B, L] array of collapsed entity-type strings per subword position."""
    with torch.no_grad():
        token_embs = tagger_model.get_token_embeddings(input_ids, attention_mask)
        logits = tagger_head(token_embs)
        pred_ids = logits.argmax(dim=-1).cpu().numpy()
    labels = np.vectorize(lambda i: collapse_label(ID2LABEL[i]))(pred_ids)
    return labels


def mean_pool_by_type(token_embs, tags, attention_mask, entity_type):
    """token_embs: [B, L, H] tensor. tags: [B, L] string array. attention_mask: [B, L].
    Returns [B, H] array (nan-filled rows where the sentence has no token of this type)."""
    B, L, H = token_embs.shape
    mask_np = attention_mask.cpu().numpy()
    embs_np = token_embs.cpu().numpy()
    pooled = np.full((B, H), np.nan, dtype=np.float32)
    for i in range(B):
        sel = (tags[i] == entity_type) & (mask_np[i] == 1)
        if sel.sum() > 0:
            pooled[i] = embs_np[i][sel].mean(axis=0)
    return pooled


def alignment_stats(en_emb, tgt_emb):
    """en_emb, tgt_emb: [N, H] arrays, already restricted to valid (non-nan) rows."""
    en_norm = en_emb / np.linalg.norm(en_emb, axis=1, keepdims=True)
    tgt_norm = tgt_emb / np.linalg.norm(tgt_emb, axis=1, keepdims=True)
    true_sim = (en_norm * tgt_norm).sum(axis=1)
    shuffled = np.roll(tgt_norm, 1, axis=0)
    random_sim = (en_norm * shuffled).sum(axis=1)
    return {
        "n": int(len(en_emb)),
        "true_pair_mean": float(true_sim.mean()),
        "shuffled_pair_mean": float(random_sim.mean()),
        "alignment_gap": float(true_sim.mean() - random_sim.mean()),
    }


def process_language(tagger_model, tagger_head, embed_model, processor, device, target_lang):
    parallel = processor.load_parallel_corpus("en", target_lang)
    subset = parallel["train"].select(range(N_SAMPLES))
    en_texts = [ex["translation"]["en"] for ex in subset]
    tgt_texts = [ex["translation"][target_lang] for ex in subset]

    en_enc = processor.tokenizer(en_texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    tgt_enc = processor.tokenizer(tgt_texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(device)

    en_tags = predict_tags(tagger_model, tagger_head, en_enc["input_ids"], en_enc["attention_mask"])
    tgt_tags = predict_tags(tagger_model, tagger_head, tgt_enc["input_ids"], tgt_enc["attention_mask"])

    with torch.no_grad():
        en_embs = embed_model.get_token_embeddings(en_enc["input_ids"], en_enc["attention_mask"])
        tgt_embs = embed_model.get_token_embeddings(tgt_enc["input_ids"], tgt_enc["attention_mask"])

    results = {}
    for etype in ENTITY_TYPES:
        en_pooled = mean_pool_by_type(en_embs, en_tags, en_enc["attention_mask"], etype)
        tgt_pooled = mean_pool_by_type(tgt_embs, tgt_tags, tgt_enc["attention_mask"], etype)
        valid = ~(np.isnan(en_pooled).any(axis=1) | np.isnan(tgt_pooled).any(axis=1))
        if valid.sum() < 5:
            results[etype] = {"n": int(valid.sum()), "alignment_gap": None, "note": "too few qualifying sentence pairs"}
            continue
        results[etype] = alignment_stats(en_pooled[valid], tgt_pooled[valid])
    return results


def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MultiLingualNERPipeline(model_name="xlm-roberta-base", max_length=MAX_LEN)

    print("Loading tagger (config E: backbone + NER head)...")
    tagger_model, tagger_head = load_tagger(device)

    print("Loading BEFORE embedding model (fresh pretrained)...")
    model_before = ZeroShotAligner("xlm-roberta-base").to(device)
    model_before.eval()

    print("Loading AFTER embedding model (alignment_checkpoint.pt)...")
    model_after = ZeroShotAligner("xlm-roberta-base").to(device)
    state = torch.load(os.path.join("checkpoints", "alignment_checkpoint.pt"), map_location=device, weights_only=True)
    model_after.load_state_dict(state["model"])
    model_after.eval()

    all_results = {"before": {}, "after": {}}
    for lang in TARGET_LANGS:
        print(f"\n=== {lang} ===")
        print("  before:")
        before = process_language(tagger_model, tagger_head, model_before, processor, device, lang)
        for etype, stats in before.items():
            gap = stats.get("alignment_gap")
            n = stats.get("n")
            print(f"    {etype}: n={n} gap={gap if gap is None else round(gap, 4)}")
        all_results["before"][lang] = before

        print("  after:")
        after = process_language(tagger_model, tagger_head, model_after, processor, device, lang)
        for etype, stats in after.items():
            gap = stats.get("alignment_gap")
            n = stats.get("n")
            print(f"    {etype}: n={n} gap={gap if gap is None else round(gap, 4)}")
        all_results["after"][lang] = after

    out_path = os.path.join("results", "entity_type_alignment.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'Lang':<6}{'Type':<6}{'Before gap':>12}{'After gap':>12}{'N (after)':>12}")
    for lang in TARGET_LANGS:
        for etype in ENTITY_TYPES:
            b = all_results["before"][lang][etype].get("alignment_gap")
            a_stats = all_results["after"][lang][etype]
            a = a_stats.get("alignment_gap")
            n = a_stats.get("n")
            b_str = f"{b:.4f}" if b is not None else "n/a"
            a_str = f"{a:.4f}" if a is not None else "n/a"
            print(f"{lang:<6}{etype:<6}{b_str:>12}{a_str:>12}{n:>12}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
