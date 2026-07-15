"""
experiment_tracker.py
Task #9: Consolidate all experiment results into a unified tracking dashboard.

Reads every results/*.json file produced by prior tasks and generates:
  1. results/plots/ablation_bar.png         -- ablation study bar chart (Task #2)
  2. results/plots/multiseed_errorbar.png   -- multi-seed error bars (Task #3)
  3. results/plots/few_shot_curve.png       -- few-shot recovery curves (Task #7)
  4. results/plots/alignment_gap.png        -- alignment gap before vs after (Task #5)
  5. results/plots/language_coverage.png    -- all 9 languages F1 bar chart, script-annotated
  6. results/plots/translate_train_vs_zero_shot.png  -- comparison bar (Task #8)
  7. results/plots/fragmentation_vs_f1.png  -- subword fragmentation hypothesis test (falsified)
  8. results/plots/corpus_size_vs_f1.png    -- pretraining corpus size hypothesis test (falsified)
  9. results/summary.csv                    -- one row per (experiment, language, metric)

Run with: python -m src.experiment_tracker
"""

import json
import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

LANG_LABELS = {"tr": "Turkish", "de": "German", "ar": "Arabic",
               "ko": "Korean", "fi": "Finnish", "sw": "Swahili", "en": "English",
               "id": "Indonesian", "ru": "Russian"}
ABLATION_LABELS = {
    "A_baseline_frozen": "A: Frozen baseline",
    "B_full_finetune_only": "B: Full fine-tune",
    "C_plus_contrastive": "C: +Contrastive",
    "D_plus_code_switch": "D: +Code-switch",
    "E_full_pipeline": "E: Full pipeline",
}
ABLATION_ORDER = ["A_baseline_frozen", "B_full_finetune_only",
                  "C_plus_contrastive", "D_plus_code_switch", "E_full_pipeline"]
COLORS = {
    "tr": "#2196F3", "de": "#4CAF50", "ar": "#F44336",
    "ko": "#FF9800", "fi": "#9C27B0", "sw": "#00BCD4",
    "id": "#8BC34A", "ru": "#795548",
}

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})


def _load(name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path) as f:
        return json.load(f)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ── 1. Ablation bar chart ──────────────────────────────────────────────────────
def plot_ablation(data):
    langs = ["tr", "de", "ar"]
    configs = ABLATION_ORDER
    n_configs = len(configs)
    n_langs = len(langs)
    x = np.arange(n_configs)
    width = 0.22
    offsets = np.linspace(-(n_langs - 1) / 2, (n_langs - 1) / 2, n_langs) * width

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, lang in enumerate(langs):
        vals = [data[c].get(lang, 0) for c in configs]
        bars = ax.bar(x + offsets[i], vals, width, label=LANG_LABELS[lang],
                      color=COLORS[lang], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([ABLATION_LABELS[c] for c in configs], rotation=15, ha="right")
    ax.set_ylabel("Zero-shot F1")
    ax.set_title("Ablation Study: Zero-Shot F1 by Configuration and Language")
    ax.set_ylim(0, 0.95)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "ablation_bar.png")


# ── 2. Multi-seed error bars ───────────────────────────────────────────────────
def plot_multiseed(data):
    summary = data["summary"]
    langs = ["tr", "de", "ar", "avg"]
    labels = [LANG_LABELS.get(l, l.upper()) for l in langs]
    means = [summary[l]["mean"] for l in langs]
    stds = [summary[l]["std"] for l in langs]
    per_seed = data["per_seed"]
    seeds = sorted(per_seed.keys())

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(langs))
    ax.bar(x, means, yerr=stds, capsize=6, color=["#2196F3", "#4CAF50", "#F44336", "#607D8B"],
           alpha=0.8, edgecolor="white", error_kw={"linewidth": 1.5})

    for j, lang in enumerate(langs):
        vals = per_seed[seeds[0]].get(lang) if lang != "avg" else None
        if lang == "avg":
            seed_vals = summary["avg"]["values"]
        else:
            seed_vals = [per_seed[s][lang] for s in seeds]
        for v in seed_vals:
            ax.plot(j, v, "o", color="black", markersize=4, alpha=0.7)

    for j, (m, s) in enumerate(zip(means, stds)):
        ax.text(j, m + s + 0.005, f"{m:.3f}±{s:.3f}", ha="center", va="bottom",
                fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1")
    ax.set_title("Config B Multi-Seed Validation (seeds 42/43/44)")
    ax.set_ylim(0, 0.9)
    fig.tight_layout()
    _save(fig, "multiseed_errorbar.png")


# ── 3. Few-shot recovery curve ─────────────────────────────────────────────────
def plot_few_shot(data):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = [0, 10, 50, 100, 500]
    for lang in ["tr", "de", "ar"]:
        vals = [data[lang][str(k)] for k in ks]
        ax.plot(ks, vals, marker="o", linewidth=2, color=COLORS[lang],
                label=LANG_LABELS[lang])
        for k, v in zip(ks, vals):
            ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7)

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("k (labeled target examples)")
    ax.set_ylabel("F1")
    ax.set_title("Few-Shot Recovery Curve (starting from zero-shot checkpoint)")
    ax.legend()
    ax.set_ylim(0.3, 0.97)
    fig.tight_layout()
    _save(fig, "few_shot_curve.png")


# ── 4. Alignment gap before vs after ──────────────────────────────────────────
def plot_alignment(data):
    langs = ["tr", "de", "ar"]
    before = [data["before"][l]["alignment_gap"] for l in langs]
    after = [data["after"][l]["alignment_gap"] for l in langs]
    x = np.arange(len(langs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - width / 2, before, width, label="Before contrastive", color="#90A4AE", alpha=0.9)
    b2 = ax.bar(x + width / 2, after, width, label="After contrastive", color="#1976D2", alpha=0.9)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS[l] for l in langs])
    ax.set_ylabel("Alignment gap (true-pair − shuffled-pair cosine sim)")
    ax.set_title("Sentence-Level Alignment Gap Before vs. After Contrastive Training")
    ax.legend()
    ax.set_ylim(0, 0.8)
    fig.tight_layout()
    _save(fig, "alignment_gap.png")


# ── 5. Language coverage bar ───────────────────────────────────────────────────
def plot_language_coverage(ablation_data, extra_data):
    # headline system (config E) + extra languages (includes id/ru, added as
    # disconfirming controls for the script hypothesis - DEVELOPMENT.md Section 4.6)
    scores = {
        "en": 0.8278,  # English val F1, DEVELOPMENT.md Section 4
        "de": ablation_data["E_full_pipeline"]["de"],
        "tr": ablation_data["E_full_pipeline"]["tr"],
        "fi": extra_data["fi"],
        "sw": extra_data["sw"],
        "ko": extra_data["ko"],
        "id": extra_data["id"],
        "ru": extra_data["ru"],
        "ar": ablation_data["E_full_pipeline"]["ar"],
    }
    # sort descending
    items = sorted(scores.items(), key=lambda x: -x[1])
    langs, vals = zip(*items)
    colors = [COLORS.get(l, "#78909C") for l in langs]
    colors = list(colors)
    en_idx = langs.index("en")
    colors[en_idx] = "#455A64"  # English (special: val F1, not zero-shot)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(langs)), vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([LANG_LABELS.get(l, l) for l in langs])
    ax.invert_yaxis()
    ax.set_xlabel("F1")
    ax.set_title("Language Coverage: Config E Zero-Shot F1 (script does NOT cleanly predict F1)")
    ax.set_xlim(0, 1.0)

    script_groups = {"Latin": {"de", "tr", "fi", "sw", "en", "id"}, "Non-Latin": {"ko", "ar", "ru"}}
    for i, (lang, val) in enumerate(zip(langs, vals)):
        script = "Latin" if lang in script_groups["Latin"] else "Non-Latin"
        ax.text(val + 0.005, i, f"{val:.3f}  [{script}]", va="center", fontsize=8)

    fig.tight_layout()
    _save(fig, "language_coverage.png")


# ── 7. Fragmentation vs. F1 and corpus size vs. F1 (falsified hypotheses) ─────
def plot_fragmentation_vs_f1(data):
    per_lang = data["per_language"]
    langs = list(per_lang.keys())
    frag = [per_lang[l]["subwords_per_word"] for l in langs]
    f1 = [per_lang[l]["f1"] for l in langs]
    r = data["pearson_r_fragmentation_vs_f1"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(frag, f1, s=80, c=[COLORS.get(l, "#78909C") for l in langs], edgecolor="white", zorder=3)
    for l, x, y in zip(langs, frag, f1):
        ax.annotate(LANG_LABELS.get(l, l), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Mean subwords per word (XLM-R tokenizer)")
    ax.set_ylabel("Zero-shot F1")
    ax.set_title(f"Subword Fragmentation vs. F1 (r = {r:.3f} - no relationship)")
    fig.tight_layout()
    _save(fig, "fragmentation_vs_f1.png")


def plot_corpus_size_vs_f1(data):
    per_lang = data["per_language"]
    langs = list(per_lang.keys())
    gib = [per_lang[l]["cc100_gib"] for l in langs]
    f1 = [per_lang[l]["f1"] for l in langs]
    r_log = data["pearson_r_log_corpus_gib_vs_f1"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(gib, f1, s=80, c=[COLORS.get(l, "#78909C") for l in langs], edgecolor="white", zorder=3)
    for l, x, y in zip(langs, gib, f1):
        ax.annotate(LANG_LABELS.get(l, l), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("XLM-R CC-100 pretraining corpus size (GiB, log scale)")
    ax.set_ylabel("Zero-shot F1")
    ax.set_title(f"Pretraining Corpus Size vs. F1 (log r = {r_log:.3f} - no relationship)")
    fig.tight_layout()
    _save(fig, "corpus_size_vs_f1.png")


# ── 6. Translate-train vs zero-shot ───────────────────────────────────────────
def plot_translate_train(data):
    langs = ["tr", "de", "ar"]
    zs = [data["zero_shot_baseline"][l] for l in langs]
    tt = [data["translate_train"][l] for l in langs]
    x = np.arange(len(langs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - width / 2, zs, width, label="Zero-shot (config E)", color="#1976D2", alpha=0.85)
    b2 = ax.bar(x + width / 2, tt, width, label="Translate-train (lexicon proxy)", color="#E53935", alpha=0.85)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.4f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS[l] for l in langs])
    ax.set_ylabel("F1")
    ax.set_title("Translate-Train (Lexicon Proxy) vs. Zero-Shot Transfer")
    ax.legend()
    ax.set_ylim(0, 0.9)
    fig.tight_layout()
    _save(fig, "translate_train_vs_zero_shot.png")


# ── Summary CSV ────────────────────────────────────────────────────────────────
def write_summary_csv(ablation, multiseed, few_shot, alignment, extra, translate_train,
                       fragmentation=None, corpus_size=None):
    rows = []

    # Ablation
    for config, lang_scores in ablation.items():
        for lang, f1 in lang_scores.items():
            rows.append({
                "experiment": "ablation",
                "config": config,
                "language": lang,
                "metric": "zero_shot_f1",
                "value": round(f1, 6),
                "notes": ABLATION_LABELS.get(config, config),
            })

    # Multi-seed
    for seed, lang_scores in multiseed["per_seed"].items():
        for lang, f1 in lang_scores.items():
            rows.append({
                "experiment": "multiseed",
                "config": f"B_seed{seed}",
                "language": lang,
                "metric": "zero_shot_f1",
                "value": round(f1, 6),
                "notes": f"Config B, seed {seed}",
            })

    # Few-shot
    for lang, k_scores in few_shot.items():
        for k, f1 in k_scores.items():
            rows.append({
                "experiment": "few_shot",
                "config": f"k={k}",
                "language": lang,
                "metric": "f1",
                "value": round(f1, 6),
                "notes": f"{k} labeled target examples",
            })

    # Alignment
    for phase, lang_metrics in alignment.items():
        for lang, metrics in lang_metrics.items():
            rows.append({
                "experiment": "alignment",
                "config": phase,
                "language": lang,
                "metric": "alignment_gap",
                "value": round(metrics["alignment_gap"], 6),
                "notes": f"true-pair minus shuffled-pair cosine sim, {phase} contrastive",
            })

    # Extra languages
    for lang, f1 in extra.items():
        rows.append({
            "experiment": "extra_languages",
            "config": "E_full_pipeline",
            "language": lang,
            "metric": "zero_shot_f1",
            "value": round(f1, 6),
            "notes": "eval-only, no retraining",
        })

    # Translate-train
    for lang, f1 in translate_train["translate_train"].items():
        rows.append({
            "experiment": "translate_train",
            "config": "lexicon_proxy",
            "language": lang,
            "metric": "f1",
            "value": round(f1, 6),
            "notes": "MUSE lexicon word-for-word substitution, not NMT",
        })
    for lang, f1 in translate_train["zero_shot_baseline"].items():
        rows.append({
            "experiment": "translate_train",
            "config": "zero_shot_baseline",
            "language": lang,
            "metric": "f1",
            "value": round(f1, 6),
            "notes": "config E reference for translate-train comparison",
        })

    # Fragmentation hypothesis test (falsified - Section 4.7)
    if fragmentation:
        for lang, metrics in fragmentation["per_language"].items():
            rows.append({
                "experiment": "fragmentation_hypothesis",
                "config": "subwords_per_word",
                "language": lang,
                "metric": "subwords_per_word",
                "value": metrics["subwords_per_word"],
                "notes": f"Pearson r vs F1 = {fragmentation['pearson_r_fragmentation_vs_f1']}",
            })

    # Corpus size hypothesis test (falsified - Section 4.7)
    if corpus_size:
        for lang, metrics in corpus_size["per_language"].items():
            rows.append({
                "experiment": "corpus_size_hypothesis",
                "config": "cc100_gib",
                "language": lang,
                "metric": "cc100_gib",
                "value": metrics["cc100_gib"],
                "notes": f"Pearson r vs F1 (log) = {corpus_size['pearson_r_log_corpus_gib_vs_f1']}",
            })

    out_path = os.path.join(RESULTS_DIR, "summary.csv")
    fieldnames = ["experiment", "config", "language", "metric", "value", "notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved: {out_path}  ({len(rows)} rows)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading result files...")
    ablation = _load("ablation_results.json")
    multiseed = _load("multiseed_results.json")
    few_shot = _load("few_shot_curve.json")
    alignment = _load("alignment_metrics.json")
    extra = _load("extra_languages_results.json")
    translate_train = _load("translate_train_results.json")
    fragmentation = _load("fragmentation_analysis.json")
    corpus_size = _load("corpus_size_analysis.json")

    print("Generating plots...")
    plot_ablation(ablation)
    plot_multiseed(multiseed)
    plot_few_shot(few_shot)
    plot_alignment(alignment)
    plot_language_coverage(ablation, extra)
    plot_translate_train(translate_train)
    plot_fragmentation_vs_f1(fragmentation)
    plot_corpus_size_vs_f1(corpus_size)

    print("Writing summary CSV...")
    write_summary_csv(ablation, multiseed, few_shot, alignment, extra, translate_train,
                       fragmentation, corpus_size)

    print("\nDone. All outputs in results/plots/ and results/summary.csv")


if __name__ == "__main__":
    main()
