"""
analyze_phonological_distance.py
Fourth hypothesis test for the zero-shot cross-lingual NER transfer gap:
phonological similarity to English, using URIEL typological features via
lang2vec (Littell et al., 2017).

This hypothesis is added specifically because Lauscher et al. (2020) report
phonological similarity as the STRONGEST single predictor of zero-shot NER
transfer quality in their much larger study (Pearson r = 0.78, Spearman
r = 0.86, for mBERT on WikiANN) - the same dataset this project uses, with
5 of this project's 8 target languages overlapping their sample. Testing
their specific strongest predictor, rather than only generic script/
fragmentation/corpus-size proxies, is the closest direct comparison this
project can make to their result.

Phonological distance is computed as 1 minus the cosine similarity between
English's and each target language's "phonology_knn" feature vector from
URIEL (28 binary/KNN-imputed phonological inventory features per language).
This is the standard "PHON" distance metric used in the typological-distance
literature (Littell et al., 2017; Lauscher et al., 2020).

Run with: python -m src.analyze_phonological_distance
(requires: pip install lang2vec)
"""

import json
import os

import numpy as np
import lang2vec.lang2vec as l2v

LANGS_F1 = {
    "tr": 0.7675, "de": 0.7435, "ar": 0.4563,
    "ko": 0.5086, "fi": 0.7578, "sw": 0.6748,
    "id": 0.4967, "ru": 0.6102,
}

# ISO 639-3 codes required by lang2vec/URIEL
ISO3 = {
    "en": "eng", "tr": "tur", "de": "deu", "ar": "ara",
    "ko": "kor", "fi": "fin", "sw": "swh", "id": "ind", "ru": "rus",
}


def cosine_distance(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return 1.0 - float(np.dot(a, b) / denom)


def pearson_r(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")


def main():
    all_iso3 = list(ISO3.values())
    features = l2v.get_features(all_iso3, "phonology_knn")

    en_vec = features[ISO3["en"]]
    results = {}
    for lang, f1 in LANGS_F1.items():
        dist = cosine_distance(en_vec, features[ISO3[lang]])
        results[lang] = {"f1": f1, "phon_distance_from_english": round(dist, 4)}
        print(f"[{lang}] phon_distance={dist:.4f}  F1={f1:.4f}")

    langs = list(LANGS_F1.keys())
    dists = [results[l]["phon_distance_from_english"] for l in langs]
    f1s = [results[l]["f1"] for l in langs]
    r = pearson_r(dists, f1s)
    print(f"\nPearson r (phonological distance vs. F1): {r:.4f}")
    print("(Lauscher et al. 2020 report r=0.78 for PHON SIMILARITY vs. NER F1 with mBERT;")
    print(" note their r is for SIMILARITY, so the equivalent DISTANCE correlation would be")
    print(" NEGATIVE at a comparable magnitude if this system replicates their finding.)")

    out = {
        "source": "lang2vec (Littell et al. 2017) URIEL phonology_knn features, cosine distance from English",
        "comparison": "Lauscher et al. (2020) report PHON similarity as the strongest NER predictor "
                       "(Pearson r=0.78, Spearman r=0.86, mBERT, WikiANN); this is a phonological "
                       "DISTANCE metric, so a replicating result would show a comparably large NEGATIVE r",
        "per_language": results,
        "pearson_r_phon_distance_vs_f1": round(r, 4),
    }
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "phonological_distance_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
