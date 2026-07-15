# Zero-Shot Cross-Lingual Named Entity Recognition via Contrastive Alignment and Code-Switching Augmentation: A Development Log

**Status:** Complete
**Last updated:** 2026-07-01

---

## Abstract

We document the full development arc of a zero-shot cross-lingual Named Entity Recognition (NER) system built on XLM-RoBERTa. The system transfers entity-recognition capability from a labeled source language (English) to unlabeled target languages without any target-language supervision. Starting from a prototype that scored Turkish zero-shot F1 = 0.330 due to non-functional contrastive alignment and capacity-limited probing, we developed a revised architecture incorporating real parallel-corpus contrastive alignment (OPUS-100), full backbone fine-tuning with layer-wise learning-rate decay, and CoSDA-ML-style code-switching augmentation (MUSE bilingual lexicons). The final system scores Turkish 0.7675 / German 0.7435 / Arabic 0.4563 zero-shot F1. We then systematically investigate why these numbers look the way they do: a 4-way ablation isolates full backbone fine-tuning as the dominant lever; multi-seed validation at n=3 revises the initial conclusion that auxiliary techniques hurt; error analysis identifies Arabic ORG over-prediction as a specific failure mode; embedding visualization confirms large contrastive alignment (gap 0.558--0.636) yet no confirmed NER benefit, with the dissociation attributed to CLS-vs-token geometry. Extended zero-shot evaluation across 8 languages initially suggested script identity (Latin vs. non-Latin) as the dominant driver of the transfer gap; two further disconfirming controls (Indonesian, Russian), followed by two additional mechanistic hypothesis tests (subword fragmentation rate, XLM-R's published per-language pretraining corpus size), falsify all three candidate explanations - the transfer gap is real and large (a 1.68x range in F1 across languages) but is not reducible to any single easily-measurable per-language property tested here, a converging null result reported as a central finding rather than an inconclusive analysis. A projection-head routing probe yields a clean negative result on the alignment-to-F1 dissociation; a few-shot recovery curve shows k=10 Arabic examples (+0.227 F1) outperforms all zero-shot interventions combined; and a lexicon-substitution translate-train baseline consistently underperforms zero-shot, attributed to disfluent pseudo-translations. The project includes an experiment tracking dashboard, a config-driven CLI, a multilingual inference demo (CLI + Gradio), and 24 unit tests for the data pipeline.

---

## 1. Introduction

Named Entity Recognition in low-resource languages is constrained by the scarcity of labeled training data. Cross-lingual transfer using multilingual pretrained encoders (e.g., XLM-RoBERTa) offers a path around this constraint: a model fine-tuned on a high-resource source language (English) can, in principle, generalize to typologically distant target languages by virtue of the encoder's shared multilingual representation space (Pires et al., 2019; Wu & Dredze, 2019).

This project explores whether that transfer can be strengthened through (a) explicit contrastive alignment of source/target embeddings, and (b) data augmentation that exposes the model to target-language surface forms during training, despite the complete absence of target-language labels.

---

## 2. Initial Architecture (Prototype)

### 2.1 Design

The initial codebase consisted of three modules:
- `data_processing.py` - WikiANN loading and subword label alignment.
- `contrastive_model.py` - an XLM-R backbone (`ZeroShotAligner`) with a 2-layer MLP projection head, trained via InfoNCE/NT-Xent loss.
- `train_zero_shot.py` - orchestration of a three-phase pipeline: (1) contrastive alignment, (2) NER head fine-tuning, (3) zero-shot evaluation.

### 2.2 Defects Identified

On first read-through, the implementation contained several defects that prevented it from running:
1. `load_and_align()` did not accept the `target_languages` parameter the trainer called it with.
2. No `DataLoader` was actually constructed; the code contained an explicit placeholder comment.
3. Batch structure (`batch['source']`, `batch['target']`) was assumed by the training loop but never produced by the data pipeline.
4. The model name string `"xlm_roberta_base"` (underscores) did not match the HuggingFace model identifier `"xlm-roberta-base"` (hyphens), which would fail at model load time.
5. No evaluation metric, checkpointing, or inference path existed.

These were corrected in the first development pass: a working `PairedNERDataset`, corrected model identifier, a complete `seqeval`-based F1 evaluation loop, and checkpoint save/load were added.

### 2.3 Initial Results

With the pipeline made runnable, contrastive alignment paired English and target-language WikiANN sentences **by index** (`en[i]` matched with `tr[i]`), and the NER head was trained as a **linear probe on a frozen XLM-R backbone**.

| Phase | Metric | Value |
|---|---|---|
| Contrastive alignment (3 epochs) | InfoNCE loss | 2.7758 → 2.7727 (flat) |
| NER fine-tuning, English validation (epoch 5) | F1 | 0.3289 |
| Zero-shot evaluation, Turkish test | F1 | 0.3297 |

**Per-class English validation F1 (epoch 5):** LOC 0.31, ORG 0.22, PER 0.47.
**Per-class Turkish zero-shot F1:** LOC 0.37, ORG 0.18, PER 0.45.

### 2.4 Analysis of Initial Results

Two findings stood out:

1. **The contrastive loss did not learn.** For a batch size of 16, the InfoNCE loss at random-chance performance equals `ln(16) ≈ 2.773`. The observed loss remained at this value across all three epochs, indicating the projection head received no usable gradient signal. The root cause was pairing strategy: WikiANN sentences in different languages are independently sampled Wikipedia text, not translations of one another. Index-based pairing therefore produced no consistent semantic correspondence between "positive" pairs, and the contrastive objective degenerated to noise.

2. **NER F1 was capacity-limited, not transfer-limited.** English validation F1 (0.329) and Turkish zero-shot F1 (0.330) were nearly identical. This is informative: it suggests XLM-R's pretrained multilingual space transfers without measurable degradation, and the bottleneck was the linear-probe head's limited capacity rather than a cross-lingual transfer gap. Published results fine-tuning XLM-R end-to-end on WikiANN typically reach 80-90% F1; a frozen-backbone linear probe should not be expected to approach this.

---

## 3. Revised Architecture

Based on the above diagnosis, three changes were made.

### 3.1 Real Parallel-Corpus Contrastive Alignment

Index-paired WikiANN sentences were replaced with genuine translation pairs from **OPUS-100** (`Helsinki-NLP/opus-100`, en-tr / de-en / ar-en configurations, ~1M pairs per language). `ParallelSentenceDataset` now samples up to 50,000 real (English, target) sentence pairs per language and tokenizes both sides for InfoNCE training. This gives the contrastive objective a genuine alignment target.

### 3.2 Full Backbone Fine-Tuning with Layer-Wise LR Decay

The NER phase no longer freezes the XLM-R backbone. All parameters are updated, using a layer-wise learning-rate decay scheme (`_build_layerwise_optimizer`): the embedding layer and early transformer layers receive the lowest learning rate (`base_lr × decay^n`), later layers receive progressively higher rates, and the projection/NER heads are trained at a higher rate (1e-3) than the backbone (2e-5 base). This is standard practice for fine-tuning large pretrained encoders (Howard & Ruder, 2018) and was expected to substantially increase capacity relative to the linear-probe baseline.

### 3.3 Code-Switching Augmentation (CoSDA-ML)

Following Qin et al. (2020), a `BilingualLexicon` class downloads and caches MUSE bilingual dictionaries (`en-tr`, `en-de`, `en-ar`) and a `CodeSwitchAugmenter` randomly substitutes English tokens in WikiANN training examples with target-language translations (default substitution probability 0.3), while leaving NER tags unchanged. An augmented copy of 50% of the English training set is generated this way per run and concatenated with the original data. This exposes the NER head to target-language surface forms in context during training, without requiring any target-language labels - the technique requires only a static bilingual word list, not labeled or even parallel sentence data.

### 3.4 Multi-Language Zero-Shot Evaluation

Zero-shot evaluation was extended from a single target language (Turkish) to three typologically distinct languages: Turkish (agglutinative), German (fusional, Latin script, closely related to English), and Arabic (Semitic, different script and morphology). This tests whether transfer gains are general or specific to one language pair.

### 3.5 Integration Defects Found During First Revised Run

Running the revised pipeline end-to-end surfaced two defects not visible from static review:

1. **Contrastive loss dropped very quickly** (en-tr: 2.22 → 0.24 within one epoch; en-de and en-ar: below 0.3 from the first logged step). This is the expected behavior of InfoNCE on *genuine* translation pairs - unlike the flat 2.77 loss in Section 2.3, the model now has real signal to exploit, and small in-batch negative pools (batch size 16) make convergence fast. This is treated as a positive signal pending downstream validation in Section 4, not a defect, but is flagged because a loss this low can also indicate degenerate collapse and should be checked against final zero-shot F1 rather than trusted in isolation.

2. **`en-de` MUSE bilingual lexicon loaded 0 entries**, silently disabling German code-switching augmentation. Root cause: the MUSE dictionary format is inconsistent across language pairs - `en-tr` and `en-ar` are tab-separated, but `en-de` is space-separated. `BilingualLexicon` originally split strictly on `\t`. Fixed by splitting on arbitrary whitespace (`line.strip().split()`). Verified post-fix: 74,655 en-de word pairs loaded (previously 0).

3. **`concatenate_datasets` raised a `ValueError`** when merging the original WikiANN split with the code-switched augmented split: `Dataset.from_list()` infers a plain `int64` type for the `ner_tags` column, whereas the original WikiANN split uses a `ClassLabel` feature for the same column, and HuggingFace `datasets` refuses to concatenate datasets with mismatched feature schemas. Fixed by explicitly casting the augmented dataset to the original split's feature schema (`augmented_ds.cast(raw_train.features)`) before concatenation.

Both fixes were verified with a standalone functional test (English WikiANN + German code-switch augmentation, 10% augment ratio) before re-running the full pipeline: lexicon loaded 74,655 entries, augmented dataset concatenated to 22,000 training examples without error.

---

## 4. Results

The revised pipeline (Section 3) was run end-to-end on the full configuration: 1 epoch of
contrastive alignment per language pair on 1M-pair OPUS-100 corpora, 5 epochs of full backbone
fine-tuning on English WikiANN (30,000 examples: 20,000 original + 10,000 code-switched), and
zero-shot evaluation on Turkish, German, and Arabic test sets (no target-language labels used
at any point).

### 4.1 Headline Comparison

| Metric | Initial Prototype (Sec. 2.3) | Revised Pipeline | Δ |
|---|---|---|---|
| Contrastive InfoNCE loss (final) | 2.7727 (chance level, `ln(16)`) | 0.10–0.36 (tr/de/ar) | learned real alignment |
| English validation F1 | 0.3289 | **0.8278** | +0.499 (+152%) |
| Turkish zero-shot F1 | 0.3297 | **0.7675** | +0.438 (+133%) |
| German zero-shot F1 | not evaluated | **0.7435** | new |
| Arabic zero-shot F1 | not evaluated | **0.4563** | new |
| Average zero-shot F1 (tr/de/ar) | - | **0.6558** | - |

### 4.2 Contrastive Alignment (Phase 1)

Unlike the flat, non-learning loss in Section 2.3, InfoNCE loss on genuine OPUS-100 translation
pairs dropped sharply within the first epoch for all three language pairs (en-tr: 1.92 → 0.36;
en-de: 0.07–0.57, settling low; en-ar: 0.40 → 0.06). This confirms the original diagnosis: the
contrastive objective requires genuine positive pairs, and index-paired WikiANN sentences
provided none.

### 4.3 English NER Fine-Tuning (Phase 2)

Full backbone fine-tuning with layer-wise LR decay and code-switch augmentation reached
**F1 = 0.8278** on English validation by epoch 5 (up from 0.467 PER / 0.221 ORG / 0.306 LOC
under the frozen linear probe to 0.89 PER / 0.74 ORG / 0.85 LOC). This is now in the expected
range for full XLM-R fine-tuning on WikiANN reported in the literature (low-to-mid 80s F1),
confirming the original hypothesis that the prototype's poor results were a **capacity**
problem (frozen probe), not a fundamental limitation of the approach.

### 4.4 Zero-Shot Transfer (Phase 3)

| Language | F1 | LOC | ORG | PER |
|---|---|---|---|---|
| Turkish (tr) | 0.7675 | 0.76 | 0.65 | 0.89 |
| German (de) | 0.7435 | 0.75 | 0.61 | 0.87 |
| Arabic (ar) | 0.4563 | 0.37 | 0.38 | 0.66 |

Turkish and German transfer strongly (0.74–0.77 F1, retaining ~92–93% of English performance
despite zero target-language labels). Arabic transfers much more weakly (0.456 F1, ~55% of
English performance), with the gap concentrated in LOC and ORG rather than PER.

**Working hypothesis for the Arabic gap**: Arabic uses a non-Latin script and root-and-pattern
(templatic) morphology, both of which diverge much further from English than Turkish's
agglutinative-but-Latin-script structure or German's close typological relation to English. XLM-R
subword tokenization is also known to be less sample-efficient for Arabic than for Latin-script
languages. This is consistent with prior literature reporting larger zero-shot transfer gaps for
languages with different scripts (Pires et al., 2019; Wu & Dredze, 2019). The ablation study
(Section 5 roadmap) is needed to determine whether code-switch augmentation specifically helps
close this gap for Arabic, or whether the contrastive alignment phase contributes more.

### 4.5 Interpretation

The three interventions (real parallel-corpus alignment, full fine-tuning, code-switch
augmentation) were applied jointly, so individual contributions cannot yet be isolated from this
run alone - that is the explicit purpose of the planned 4-way ablation study (baseline /
full-finetune-only / +contrastive-only / +code-switch-only / full pipeline). What can be said
from this run: the combined approach more than doubled both source-language performance and
zero-shot transfer F1 relative to the initial prototype, validating the overall diagnosis from
Section 2.4 - the original bottleneck was capacity and a non-functional alignment signal, not an
inherent ceiling on XLM-R's cross-lingual transferability.

### 4.6 Extended Language Coverage: Disentangling Script from Morphology (Revised)

The Section 4.4 hypothesis for the Arabic gap bundled two confounded factors: non-Latin script
*and* templatic/root-and-pattern morphology, both of which diverge from English. To separate
them, the existing config E checkpoint (no retraining - zero-shot eval is pure inference) was
evaluated on five further languages chosen to decouple these factors, added across two rounds
(`src/eval_extra_languages.py`, `results/extra_languages_results.json`):

| Language | Script | Morphology / family | Zero-shot F1 |
|---|---|---|---|
| Turkish (tr) | Latin | Agglutinative, Turkic (distant from English) | 0.7675 |
| Finnish (fi) | Latin | Agglutinative, Uralic (distant from English) | 0.7578 |
| German (de) | Latin | Fusional, Germanic (close to English) | 0.7435 |
| Swahili (sw) | Latin | Agglutinative/noun-class, Bantu (very distant) | 0.6748 |
| Russian (ru) | Cyrillic (non-Latin) | Fusional, Slavic (Indo-European, moderately close) | 0.6102 |
| Korean (ko) | Hangul (non-Latin) | Agglutinative, isolate (distant) | 0.5086 |
| Indonesian (id) | Latin | Largely isolating, Austronesian (very distant) | 0.4967 |
| Arabic (ar) | Arabic (non-Latin) | Templatic, Semitic (distant) | 0.4563 |

**Round 1 (Turkish, German, Finnish, Swahili, Korean, Arabic - six languages) initially supported
a clean thesis: script, not morphological distance, is the dominant factor.** Every Latin-script
language landed in a tight 0.67–0.77 F1 band regardless of typological distance from English
(Swahili, Bantu, transferred nearly as well as German, closely related), while the two non-Latin
languages (Korean, Arabic) were the two worst performers by a wide margin. This is the finding
originally reported here and in Sections 6/7/14.

**Round 2, added specifically to stress-test that thesis, complicates it.** Indonesian was chosen
as a second "distant morphology, Latin script" control from a different family than Swahili
(Austronesian rather than Bantu); Russian was chosen as a second "non-Latin script" control from a
family *close* to English (Indo-European/Slavic) rather than a distant one - the sharpest possible
test of whether script alone, independent of family, drives the gap. Neither control behaved as
the clean thesis predicted:

- **Indonesian (Latin script) scored 0.4967** - below Korean, barely above Arabic, and nowhere
  near the 0.67–0.77 Latin-script band the first four Latin-script languages occupied. A
  Latin-script language performing like the worst non-Latin-script languages directly falsifies
  "Latin script → safe zero-shot transfer" as stated.
- **Russian (non-Latin script) scored 0.6102** - clearly better than Arabic and Korean, but also
  clearly below the Latin-script cluster's lower bound (Swahili, 0.6748). It neither confirms
  "non-Latin script → poor transfer" cleanly nor refutes it outright; it lands in between.

**Revised conclusion: script correlates with the zero-shot transfer gap across this 8-language
sample but is not sufficient to explain it on its own.** The clean binary story from Round 1 does
not survive Round 2's disconfirming controls. Section 4.7 tests two further, more mechanistic
candidate explanations (subword fragmentation, XLM-R pretraining corpus size) to see whether
either resolves the residual pattern that script alone cannot.

*(The original Round-1-only version of this section claimed script as a confirmed "decisive
finding" and is superseded by the analysis above; see the Section 15 changelog entry for this
revision and Section 14.3 for the updated headline finding.)*

### 4.7 Testing Two Alternative Explanations: Subword Fragmentation and Pretraining Corpus Size

If script identity alone doesn't explain the 8-language pattern, two more mechanistic candidates
suggest themselves from the literature: (1) XLM-R's subword vocabulary may fragment some
languages' words into more pieces than others, independent of script per se, making the
downstream classification task effectively harder; (2) XLM-R was pretrained on unequal amounts of
CC-100 text per language (Conneau et al., 2020), so lower-resource languages in pretraining might
transfer worse regardless of script. Both are cheap to test without any additional training.

**Subword fragmentation** (`src/analyze_fragmentation.py`, `results/fragmentation_analysis.json`):
mean subwords-per-word was computed for each of the 8 languages by tokenizing 2,000 WikiANN test
sentences per language with the same `xlm-roberta-base` tokenizer used throughout this project.

| Language | Subwords/word | Zero-shot F1 |
|---|---|---|
| Swahili | 1.553 | 0.6748 |
| Indonesian | 1.594 | 0.4967 |
| German | 1.609 | 0.7435 |
| Arabic | 1.715 | 0.4563 |
| Turkish | 1.751 | 0.7675 |
| Russian | 1.914 | 0.6102 |
| Finnish | 1.955 | 0.7578 |
| Korean | 2.212 | 0.5086 |

Pearson r (subwords/word vs. F1) = **−0.150** - weak and, if anything, in the direction opposite
what the hypothesis predicts (more fragmentation should mean lower F1; a coefficient this close to
zero means fragmentation explains almost none of the variance). The concrete counterexamples make
this unambiguous without needing statistical significance testing on n=8: **Finnish fragments more
than Russian and Arabic (1.955 subwords/word) yet has one of the highest F1 scores (0.7578);
Indonesian fragments about as little as Swahili (1.594 vs. 1.553) yet has one of the lowest F1
scores (0.4967) of any language tested.** Subword fragmentation does not explain the pattern.

**Pretraining corpus size** (`results/corpus_size_analysis.json`, sourced from Table 6 of Conneau
et al., 2020): CC-100 corpus size in GiB per language, as used to pretrain XLM-R.

| Language | CC-100 size (GiB) | Zero-shot F1 |
|---|---|---|
| Swahili | 1.6 | 0.6748 |
| Turkish | 20.9 | 0.7675 |
| Arabic | 28.0 | 0.4563 |
| Korean | 54.2 | 0.5086 |
| Finnish | 54.3 | 0.7578 |
| German | 66.6 | 0.7435 |
| Indonesian | 148.3 | 0.4967 |
| Russian | 278.0 | 0.6102 |

Pearson r (corpus size vs. F1) = **−0.214** (raw GiB) / **−0.235** (log GiB) - again weak, and
wrong-signed: more pretraining data weakly correlates with *worse*, not better, zero-shot F1 in
this sample. The counterexamples are sharp: **Swahili has the smallest CC-100 allocation of any
language tested (1.6 GiB - two orders of magnitude below Russian's 278 GiB) yet transfers
well (0.6748); Indonesian has the third-largest allocation (148.3 GiB) yet is one of the two worst
performers (0.4967); Russian has by far the largest allocation (278 GiB) yet is only mid-pack
(0.6102), clearly behind Turkish's 0.7675 on 20.9 GiB.** Pretraining corpus size does not explain
the pattern either.

### 4.8 Synthesis: Three Falsified Hypotheses

Three candidate explanations for the zero-shot cross-lingual NER transfer gap have now been
tested against the same 8-language sample, each chosen because it is a concrete, quantifiable,
literature-grounded mechanism rather than a vague appeal to "difficulty":

| Hypothesis | Test | Result |
|---|---|---|
| Script identity (Latin vs. non-Latin) | 8-language comparison with 2 disconfirming controls | Falsified by Indonesian (Latin, low F1); complicated by Russian (non-Latin, mid F1) |
| Subword fragmentation rate | Pearson correlation, tokenizer-measured | r = −0.150, wrong-signed, sharp counterexamples (Finnish, Indonesian) |
| XLM-R pretraining corpus size | Pearson correlation, published CC-100 sizes | r = −0.214 (−0.235 log), wrong-signed, sharp counterexamples (Swahili, Indonesian, Russian) |

None of the three explains the observed pattern. This is reported as a genuine, deliberately
sought-out negative result rather than an inconclusive analysis abandoned partway through: each
hypothesis was tested to completion with a clear, falsifiable prediction, and each failed that
prediction on concrete counterexamples rather than merely "not reaching significance." The most
honest conclusion available from this project's evidence is that **the zero-shot cross-lingual NER
transfer gap in XLM-R is real, large (a 1.68x range in F1 across the 8 languages tested), and
robust to auxiliary alignment/augmentation interventions (Section 5), but is not reducible to any
single easily-measurable per-language property tested here.** Plausible remaining candidates this
project's evidence cannot adjudicate include WikiANN's per-language silver-standard annotation
quality (the dataset is built from Wikipedia hyperlink structure, not human annotation, and is
known to vary in quality across languages) and finer-grained properties of XLM-R's per-language
representation geometry not captured by aggregate corpus size or fragmentation rate. See Section
14 for how this synthesis reframes the project's headline findings.

---

## 5. Ablation Study

### 5.1 Design

Section 4 applied all three interventions jointly, so their individual contributions could not
be isolated. To address this, five configurations were trained, each starting from a **fresh**
pretrained `xlm-roberta-base` (no config reuses another's weights):

| Config | Backbone | Contrastive alignment | Code-switch augmentation |
|---|---|---|---|
| A - baseline | frozen (linear probe) | no | no |
| B - full-finetune only | full fine-tune | no | no |
| C - +contrastive | full fine-tune | yes | no |
| D - +code-switch | full fine-tune | no | yes |
| E - full pipeline | full fine-tune | yes | yes |

E was not re-run; it reuses the completed Section 4 run, since the underlying code path is
unchanged. A–D were each trained for 5 NER epochs (1 contrastive epoch where applicable) and
evaluated zero-shot on the same Turkish/German/Arabic test sets, with no target-language labels
used at any point.

### 5.2 Results

| Config | TR | DE | AR | Avg | Δ vs. B |
|---|---|---|---|---|---|
| A - frozen baseline | 0.5434 | 0.5383 | 0.2205 | 0.4341 | −0.235 |
| B - full-finetune only | 0.7584 | **0.7531** | **0.4950** | **0.6688** | - |
| C - +contrastive | 0.7360 | 0.7447 | 0.4467 | 0.6424 | −0.026 |
| D - +code-switch | 0.7401 | 0.7428 | 0.4663 | 0.6497 | −0.019 |
| E - full pipeline | **0.7675** | 0.7435 | 0.4563 | 0.6558 | −0.013 |

*(Bold = column maximum. Note E has the single best Turkish score of any config, while B leads on German, Arabic, and average - consistent with the Section 5.3 finding that B is the strongest overall configuration even though it isn't the single best on every column.)*

### 5.3 Analysis

**Full backbone fine-tuning is the dominant lever, by a wide margin.** Going from frozen probe
(A) to full fine-tune (B) raised average zero-shot F1 by +0.235 (0.434 → 0.669) - over an order
of magnitude larger than the effect of either auxiliary technique. This confirms the Section 2.4
diagnosis: the original prototype's weak results were a capacity problem, and fixing it accounts
for nearly all of the gain reported in Section 4.

**Contrastive alignment and code-switch augmentation each slightly *hurt* zero-shot F1 relative
to full fine-tuning alone, both individually and combined.** This was not the expected outcome - 
both techniques were added on the hypothesis that they would help, particularly for Arabic. C
(−0.026 avg) and D (−0.019 avg) each underperform B, and E (both together, −0.013 avg) does not
recover to B's level either. The degradation is small and within the range plausibly explained by
single-run training noise (no multi-seed averaging yet - see Task #3), but it is consistent in
direction across all three languages for C, and across TR/DE for D.

**The Arabic gap is not closed by either auxiliary technique.** B (full fine-tune only) achieves
the *best* Arabic score of any configuration (0.4950) - better than C (0.4467), D (0.4663), and
even E (0.4563). If code-switch augmentation or contrastive alignment were specifically
compensating for script/morphology divergence (the Section 4.4 hypothesis), Arabic should have
been the language that benefited most from adding them. Instead it's the language where C hurts
most in absolute terms (−0.048 vs. B). This points away from "the auxiliary techniques help
Arabic but not enough" and toward **the Arabic gap being a property of XLM-R's pretrained
representation/tokenization for Arabic that fine-tuning recipe changes don't address at this
scale** - not a deficiency the current toolkit (parallel-corpus alignment, lexicon-based
code-switching) is well-suited to fix.

**Caveats (resolved in 5.4).** These were single-seed results; the small negative deltas (−0.013
to −0.026 avg) needed multi-seed confirmation before being treated as a real effect rather than
noise. Section 5.4 reports that confirmation.

### 5.4 Multi-Seed Validation

Config B (full-finetune-only) was re-run for 3 seeds (42, 43, 44), each from a fresh pretrained
backbone, to establish the run-to-run variance against which the Section 5.2/5.3 deltas should be
judged (`src/run_multiseed.py`, `results/multiseed_results.json`).

| Language | Mean | Std | Per-seed values |
|---|---|---|---|
| Turkish | 0.7561 | 0.0112 | 0.7409, 0.7601, 0.7674 |
| German | 0.7544 | 0.0041 | 0.7501, 0.7599, 0.7534 |
| Arabic | 0.4828 | 0.0291 | 0.4901, 0.5143, 0.4441 |
| **Average** | **0.6645** | **0.0099** | 0.6603, 0.6781, 0.6549 |

**Revised conclusion: most of the Section 5.2/5.3 deltas are within noise, not a real effect of
removing the auxiliary techniques.** The single-seed value originally reported for B (avg 0.6688)
falls well inside this 3-seed distribution (mean 0.6645 ± 0.0099) - it was simply one sample from
this spread, not a specially strong run. Judged against this noise floor:

- **B vs. E (full pipeline, avg 0.6558):** the gap (−0.013) is just over 1 std below B's mean - 
  plausibly noise. The earlier claim that "full-finetune-alone outperforms the full pipeline"
  does **not** hold up with proper uncertainty quantification; E's single value is consistent
  with having been drawn from the same distribution as B's seeds.
- **B vs. D (+code-switch, avg 0.6497):** gap (−0.019) is ~1.9 std below B's mean - still not
  clearly distinguishable from noise with only n=3.
- **B vs. C (+contrastive, avg 0.6424):** gap (−0.026) is ~2.2 std below B's mean - the most
  plausible candidate for a real (negative) effect of the three, though n=3 is too small to claim
  this with confidence; it would need its own multi-seed run to confirm.

**Arabic is the noisiest language by a wide margin** (std 0.029, vs. 0.011 for Turkish and 0.004
for German - roughly 3-7x higher relative variance). This is itself a finding worth noting: it
suggests XLM-R's zero-shot Arabic representations are less stable under English-only fine-tuning
than Turkish or German, independent of which auxiliary technique is used. It also means any
single-seed claim about Arabic specifically (e.g. "B has the best Arabic score") is on
particularly weak footing - the 3-seed Arabic range for B alone (0.444–0.514) already spans most
of the gap between the originally reported B (0.495) and C (0.447) values.

**Revised interpretation of Section 5.3:** the *first* finding (full fine-tuning is the dominant
lever, A → B) stands - it is roughly 24x larger than B's own multi-seed std (0.235 vs. 0.0099) and
is not in doubt. The *second* finding (contrastive/code-switch hurt performance) should be
downgraded from "established" to "suggestive, dominated by C, not yet confirmed" given it is the
same order of magnitude as seed noise. This is exactly the failure mode multi-seed validation
exists to catch, and is reported here rather than left as the more dramatic but unconfirmed
single-seed claim.

---

## 6. Error Analysis

### 6.1 Setup

Token-level predictions from the config E (full pipeline) checkpoint were collected on the full
tr/de/ar test sets and collapsed from BIO tags to entity type (O/PER/ORG/LOC) for a 4x4 confusion
matrix per language (`src/error_analysis.py`, `results/confusion_<lang>.png`,
`results/error_analysis.json`). Qualitative examples were drawn by re-tokenizing raw test
sentences and surfacing the first 8 per language containing at least one entity-type mismatch
(`results/error_analysis.md`).

### 6.2 Per-Class Recall

| Language | O | PER | ORG | LOC |
|---|---|---|---|---|
| Turkish | 0.932 | 0.922 | 0.897 | 0.841 |
| German | 0.953 | 0.915 | 0.846 | 0.829 |
| Arabic | 0.768 | 0.686 | 0.902 | 0.590 |

Turkish and German show a similar, modest pattern: recall is highest for O and PER, lowest for
LOC, and all four classes sit within a ~10-point band (0.83–0.95). Arabic breaks this pattern
sharply - ORG recall (0.902) is comparable to or *better* than tr/de, but O, PER, and especially
LOC recall collapse (0.768, 0.686, 0.590).

### 6.3 The Arabic Failure Mode Is Specific, Not Uniform: Over-Prediction of ORG

The confusion matrices show this isn't a general degradation - it's concentrated in one
direction. Comparing the rate at which each true class is mispredicted as ORG:

| True class -> predicted ORG | Turkish | German | Arabic |
|---|---|---|---|
| O -> ORG | 4.2% | 3.2% | **16.3%** |
| PER -> ORG | 5.2% | 5.0% | **26.0%** |
| LOC -> ORG | 11.1% | 11.4% | **37.8%** |

In Arabic, the model over-predicts ORG at 2-4x the rate seen in Turkish/German for every other
class, and this single failure mode accounts for the bulk of the recall collapse in O, PER, and
LOC. (ORG's own recall is *unaffected* - 0.902, in line with tr/de - because ORG is the class
absorbing the confusion, not losing to it.) This refines the Section 4.4 hypothesis: the Arabic
zero-shot gap is not a uniform "harder language" effect but a specific learned bias toward the
ORG class when the model is uncertain on Arabic input, plausibly because Arabic PER/LOC surface
patterns (e.g. multi-word names, definite-article prefixes) overlap more with whatever cues the
English-only fine-tuning taught it to associate with ORG than do Turkish or German equivalents.
This is now a concrete, falsifiable target for follow-up rather than a vague script/morphology
hypothesis.

### 6.4 Qualitative Findings

- **A recurring corpus artifact, not a model failure**: several Turkish and Arabic examples begin
  with a Wikipedia redirect-page marker (`YÖNLENDİRME`, `تحويل` - both mean "redirect") as the
  first token of the sentence, immediately followed by the redirect target (often itself an
  entity). The model frequently mistags the marker token itself as `B-ORG` or `B-LOC` (e.g. error
  analysis tr examples 3, 7, 8; ar examples 2, 3, 5, 8). This is a WikiANN data quirk - redirect
  boilerplate occasionally leaking into the NER-labeled sentence - rather than a genuine
  cross-lingual transfer failure, and inflates the apparent error rate slightly across all three
  languages.
- **Boundary errors dominate over type confusion** for tr/de: most mismatches are I-tag
  continuation errors (e.g. correctly identifying the start of a multi-word entity but extending
  or truncating its span by one token - German example 8, Turkish example 5) rather than
  confusing one entity type for a completely different one.
- **Arabic shows genuine type confusion, not just boundary errors**: example 3 (`ويلفريد أورباين
  إيلفيس إندزانغا`, a person's full name) is entirely mistagged as I-ORG; example 6 tags a
  political title phrase as ORG instead of O. This is consistent with the systematic ORG
  over-prediction in 6.3 rather than imprecise span boundaries.

---

## 7. Embedding Alignment Visualization

### 7.1 Setup

A potential gap in Section 4.2 was that contrastive alignment was judged only by InfoNCE loss
going down - a loss can drop without the underlying embedding space actually becoming more
aligned, since loss is relative to in-batch negatives rather than an absolute measure. To check
this directly, 200 real OPUS-100 (English, target) sentence pairs per language were embedded with
two model states: a **fresh pretrained XLM-R** ("before") and the **headline run's
`alignment_checkpoint.pt`** ("after", end of Phase 1, before any NER fine-tuning) (`src/visualize_alignment.py`).

For each state, two things were measured per language:
- **Raw cosine similarity** of true (en, target) pairs - but raw similarity alone is a weak
  signal, since pretrained LM sentence embeddings are known to be anisotropic (clustered in a
  narrow cone), which inflates cosine similarity for *any* pair of sentences, matched or not.
- **Alignment gap** = true-pair similarity − shuffled-pair similarity (same embeddings,
  mismatched pairing). This isolates the part of the similarity that's actually attributable to
  translation correspondence, controlling for the anisotropy baseline.

### 7.2 Results

| Language | Before: true-pair sim | Before: gap | After: true-pair sim | After: gap |
|---|---|---|---|---|
| Turkish | 0.992 | 0.005 | 0.767 | **0.629** |
| German | 0.993 | 0.007 | 0.795 | **0.636** |
| Arabic | 0.977 | 0.005 | 0.784 | **0.558** |

(`results/alignment_metrics.json`, t-SNE plot: `results/embedding_alignment_tsne.png`)

### 7.3 Analysis

**Contrastive training produced genuine alignment, not just a falling loss number.** Before
training, the gap between true and shuffled pairs is ~0.005-0.007 for all three languages - 
essentially indistinguishable from chance, confirming the anisotropy concern: raw similarities
of 0.97-0.99 were measuring "all sentences look similar in this space," not translation
correspondence. After contrastive training, raw similarity actually *drops* (0.77-0.80) because
training spreads embeddings out across the space, but the gap jumps to 0.56-0.64 - true pairs are
now reliably ~60 percentage points closer than mismatched pairs. This is strong, direct evidence
that Phase 1 is doing what it's designed to do, independent of the InfoNCE loss curve reported in
Section 4.2.

**Arabic alignment is real but measurably weaker than Turkish/German** (gap 0.558 vs. 0.629/0.636
 - about 0.07-0.08 lower, a similar relative gap to other Arabic-specific weaknesses found
elsewhere in this project). This is consistent with, and adds a third independent line of
evidence for, the Arabic-specific pattern documented in Sections 5 and 6.

**The more important finding is the dissociation between this result and Sections 5/5.4**:
contrastive alignment demonstrably succeeds at its own stated objective (pulling true translation
pairs together in embedding space, confirmed here with a large, unambiguous effect size) - yet
the multi-seed-validated ablation in Section 5.4 found no confirmed improvement in downstream
zero-shot NER F1 from adding it (config C's −0.026 avg F1 vs. B remains the most plausible
estimate of its net effect, though still not significant at n=3). Put together, these results
suggest the auxiliary representation-alignment objective and the downstream token-classification
objective are not as connected as the original design assumed: aligning **sentence-level** (CLS)
embeddings doesn't necessarily transfer to better **token-level** representations, since NER
fine-tuning and evaluation never use the projection head this metric is computed from (see the
note in `_build_layerwise_optimizer`, Section 3.2) - only the shared backbone benefits indirectly,
and evidently not enough to move the F1 needle outside of noise. This is a more precise statement
of the project's central empirical finding than either Section 5 or Section 7 could provide alone.

### 7.4 Projection-Head Routing Probe (Task #14)

Section 7.3 proposed an explanation for the dissociation: NER training/eval never touch the
projection head, so the demonstrated sentence-level alignment can't directly inform token-level
NER representations. This is directly testable: config F repeats config C's recipe (contrastive
alignment + full fine-tune, no code-switch) with one change - `route_through_projection=True`
forces NER training/eval to classify `projection_head(token_embeddings)` instead of raw backbone
token embeddings (`src/run_probe.py`, `get_projected_token_embeddings()` in
`contrastive_model.py`).

| Config | TR | DE | AR | Avg | Δ vs. C |
|---|---|---|---|---|---|
| C - contrastive, not routed | 0.7360 | 0.7447 | 0.4467 | 0.6424 | - |
| F - contrastive, routed through projection head | 0.7382 | 0.7464 | 0.4384 | 0.6410 | −0.0014 |

**The probe does not support the Section 7.3 explanation.** F and C are essentially identical
(avg delta −0.0014, an order of magnitude smaller than config B's multi-seed std of 0.0099 from
Section 5.4 - nowhere near distinguishable from noise). Forcing NER to use the contrastively
aligned projection head neither closes nor worsens the gap; it does nothing measurable either
way. This rules out the most direct version of the "the projection head isn't reachable" theory:
making it reachable doesn't help.

A more likely explanation, given this result: the projection head's alignment is a property of
the **pooled [CLS] vector geometry** specifically (a 2-layer MLP trained on whole-sentence
representations), and applying that same transformation independently to every token's vector
doesn't carry the same meaning - there is no guarantee that what aligns a sentence's summary
vector also aligns the individual token vectors that make it up, especially for entity-bearing
tokens that are a small fraction of a sentence's content. The benefit of contrastive training
that *does* persist (if any - recall this is itself unconfirmed at n=3) most likely flows
through small shared-backbone weight adjustments from Phase 1's gradient updates, not through any
mechanism this probe could activate by routing through the projection head. Properly resolving
this would need representation-similarity analysis (e.g. CKA between backbone layers before/after
contrastive training) rather than another architectural variant - noted as a natural follow-up
but out of scope here.

---

## 8. Few-Shot Recovery Curve

### 8.1 Setup

Sections 5-7 establish *why* Arabic transfers poorly at the zero-shot algorithmic level. This
section asks a different, more practical question: if a deployment can label even a small amount
of target-language data, how much of the gap closes immediately? Starting from the config E
checkpoint, the model was fine-tuned independently (each run starting fresh from the zero-shot
checkpoint, not cumulative) on k = 10/50/100/500 labeled examples per target language, then
evaluated on the full test set (`src/few_shot_curve.py`, `results/few_shot_curve.json`). k=0 is
the already-measured zero-shot baseline (Section 4.4).

### 8.2 Results

| k (labeled examples) | Turkish | German | Arabic |
|---|---|---|---|
| 0 (zero-shot) | 0.7675 | 0.7435 | 0.4563 |
| 10 | 0.7600 | 0.7514 | **0.6828** |
| 50 | 0.8192 | 0.7616 | 0.7264 |
| 100 | 0.8282 | 0.7932 | 0.7469 |
| 500 | 0.8606 | 0.8106 | 0.7868 |

*(Bold marks the single largest single-step jump in the table - Arabic's +0.227 from k=0 to k=10,
discussed in 8.3 - not a column maximum; every language's own k=500 value is its highest.)*

### 8.3 Analysis

**A handful of labeled Arabic examples closes most of the gap that no zero-shot technique in this
project could.** Just 10 labeled Arabic examples raise F1 from 0.456 to 0.683 - a +0.227 jump,
recovering roughly three-quarters of the entire 0.31-point gap to Turkish/German in a single small
fine-tuning pass. By k=500, Arabic (0.787) is within 0.07-0.08 of Turkish (0.861) and German
(0.811) - the original gap has shrunk by more than 75%. For comparison, the best result from any
zero-shot intervention tested anywhere in this project (contrastive alignment, code-switching,
projection-head routing, multi-seed-averaged) was on the order of ±0.01-0.03 F1, and most of that
was not statistically distinguishable from noise. **500 labeled examples (about 2.5% of the
20,000-example English training set) outperforms every zero-shot algorithmic intervention
combined, by an order of magnitude.**

**Turkish shows a small dip at k=10 (0.7675 → 0.7600) before recovering.** This is consistent with
instability from fine-tuning on a very small batch (effectively 1-2 gradient steps per epoch at
batch size 8) rather than a real regression - Turkish was already close to its German/English
ceiling zero-shot, so there's less room for k=10 to help and more relative exposure to sample
noise from such a small set. German improves steadily and roughly linearly across all four k
values, never dipping.

**Practical implication for Section 4.6's script-based explanation**: the Arabic gap being
"intrinsic to XLM-R's pretrained representation for non-Latin scripts" doesn't mean it's
unfixable - it means the fix is supervision, not unsupervised algorithmic cleverness. A small
amount of target-language signal apparently lets the existing (already fine-tuned) backbone and
NER head recalibrate to Arabic's surface patterns far more effectively than any attempt to
pre-align the representation space without labels. This is a genuinely useful, actionable
conclusion for anyone deciding how to deploy this kind of system on a low-resource target
language: don't invest further in unsupervised alignment tricks: invest in labeling a few hundred
examples instead.

---

## 9. Translate-Train Baseline

### 9.1 Setup

Translate-train is the standard alternative to zero-shot transfer in the cross-lingual NER
literature: instead of training on English and transferring zero-shot (config E), translate the
labeled training data into the target language and train directly on that. A full implementation
needs a trained NMT system plus word-alignment-based label projection - out of scope for the
compute/dependency budget here. Instead, a **lexicon-substitution proxy** was built
(`src/translate_train_baseline.py`, `load_ner_dataset_translated()` in `data_processing.py`):
every word in the entire English training set is substituted via the same MUSE bilingual
lexicons used for code-switching (Section 3.3), but at `switch_prob=1.0` (every translatable
word, not 30%) and not mixed with the original English data. Labels stay aligned automatically
since substitution is per-token and in place. The model is then trained from scratch (matching
config B/E's recipe: full backbone fine-tune, layer-wise LR decay, 5 epochs) on this fully
pseudo-translated set and evaluated on the real target-language test set.

**This is explicitly a simplification, not a true translate-train baseline.** A static
word-for-word dictionary captures lexical substitution only - no word reordering, no
morphological adaptation (critically relevant for agglutinative Turkish or templatic Arabic,
where a word-for-word swap doesn't produce a grammatical sentence), and any word missing from the
lexicon is left in English, producing patchy, code-mixed-looking pseudo-target-language text
rather than fluent target language.

### 9.2 Results

| Language | Zero-shot (E) | Translate-train (lexicon) | Δ |
|---|---|---|---|
| Turkish | 0.7675 | 0.7344 | −0.0331 |
| German | 0.7435 | 0.7218 | −0.0217 |
| Arabic | 0.4563 | 0.4390 | −0.0173 |

### 9.3 Analysis

**Translate-train (this lexicon-substitution version) underperforms zero-shot transfer for all
three languages.** The deltas are consistently negative (−0.017 to −0.033), larger in magnitude
than the Section 5.4 noise floor (config B's multi-seed std of 0.0099) and consistent in
direction across all three typologically different languages - a weaker form of evidence than a
proper multi-seed run on this baseline (not done here, flagged as a caveat, same discipline as
Section 5.4), but the sign agreement across tr/de/ar makes "this is just noise in a random
direction" a less likely explanation than for a single-language single-seed result.

This is plausibly explained by the documented limitation rather than reflecting on translate-train
as a strategy in general: **training on disfluent, partially-untranslated, morphologically
incorrect pseudo-target-language text is worse signal than training on clean, fluent English and
relying on XLM-R's pretrained cross-lingual alignment to transfer it zero-shot.** This is
consistent with the project's recurring pattern (Sections 5, 7.4): naive or static interventions
(word-level lexicon substitution here; non-functional index-pairing in Section 2; the projection
head routing in 7.4) tend to underperform the simpler, well-executed baseline, while the genuine
lever that worked throughout this project was always full backbone fine-tuning capacity, not
auxiliary data engineering. A true NMT-based translate-train baseline - producing fluent,
grammatical target-language sentences - would very plausibly close or reverse this gap, since the
literature generally reports translate-train as competitive with or better than zero-shot
transfer; this result should be read as "naive lexicon substitution is worse than zero-shot," not
as "translate-train as a general strategy is worse than zero-shot."

---

## 10. Experiment Tracking and Visualization Dashboard

### 10.1 Motivation

With eight experimental tracks complete (ablation, multi-seed validation, error analysis, embedding alignment, extended language coverage, projection-head probe, few-shot curve, translate-train baseline), the project had accumulated eight separate result JSON files with no unified view across them. Task #9 addresses this by consolidating all results into a single tracked artifact set: a family of publication-ready plots and a normalized summary CSV.

### 10.2 Implementation

**`src/experiment_tracker.py`** is a standalone script that reads all `results/*.json` files and produces:

| Output | Description |
|--------|-------------|
| `results/plots/ablation_bar.png` | Grouped bar chart: all 5 ablation configs × 3 languages |
| `results/plots/multiseed_errorbar.png` | Mean ± std bar chart for config B, seeds 42/43/44, with individual seed points overlaid |
| `results/plots/few_shot_curve.png` | F1 vs. k curves on symlog x-axis for tr/de/ar (k=0/10/50/100/500) |
| `results/plots/alignment_gap.png` | Before vs. after contrastive alignment gap (true-pair − shuffled-pair cosine sim) |
| `results/plots/language_coverage.png` | Horizontal bar chart across all 7 languages, annotated with script category |
| `results/plots/translate_train_vs_zero_shot.png` | Side-by-side zero-shot vs. lexicon-proxy translate-train comparison |
| `results/summary.csv` | 54-row normalized table: one row per (experiment, config, language, metric, value) |

The tracker uses only `matplotlib` and `numpy` - no additional dependencies beyond what the main pipeline already requires. The script is completely idempotent; re-running overwrites outputs deterministically from the fixed JSON sources.

### 10.3 Summary CSV Schema

The `results/summary.csv` file provides a flat, analysis-ready representation of all experimental findings:

```
experiment, config, language, metric, value, notes
ablation, A_baseline_frozen, tr, zero_shot_f1, 0.543370, ...
ablation, B_full_finetune_only, tr, zero_shot_f1, 0.758362, ...
multiseed, B_seed42, tr, zero_shot_f1, 0.740856, Config B, seed 42
few_shot, k=10, ar, f1, 0.682829, 10 labeled target examples
alignment, after, ar, alignment_gap, 0.557880, ...
...
```

This schema is compatible with `pandas.read_csv()` and makes it straightforward to reproduce any reported aggregate (e.g., average zero-shot F1 per ablation config, few-shot gain per language) without re-running any training.

### 10.4 Key Visual Findings

The language coverage chart (`language_coverage.png`) most clearly encapsulates the project's central empirical finding: all six Latin-script languages tested cluster tightly between 0.67 and 0.83 F1, while the two non-Latin-script languages (Arabic, Korean) fall 0.15-0.35 points below the cluster. The few-shot curve plot (`few_shot_curve.png`) shows the complementary finding: Arabic's gap collapses rapidly with minimal supervision, indicating the XLM-R representation space is not fundamentally broken for Arabic - it simply requires a small amount of labeled signal to calibrate.

---

## 11. Configuration-Driven CLI

### 11.1 Motivation

Prior to this change, `python -m src.train_zero_shot` read its hyperparameters from a hardcoded Python dict inside the `__main__` block, which `configs/config.yaml` (written earlier) duplicated without being wired to anything. Repeating experiments with different settings required manual code edits, making it impossible to run comparison sweeps without touching source files.

### 11.2 Implementation

The `__main__` block of `src/train_zero_shot.py` was replaced with an `argparse`-based CLI that:

1. Reads `configs/config.yaml` as the default hyperparameter source.
2. Accepts `--config PATH` to substitute a different YAML file.
3. Exposes every scalar hyperparameter as an override flag (e.g., `--ner-epochs 10`, `--switch-prob 0.5`).
4. Accepts `--target-langs tr de ar ko fi` to override the language list without editing YAML.
5. Accepts `--skip-contrastive` to re-run only Phase 2+3 from an existing checkpoint.
6. Accepts `--seed INT` for reproducible runs.

CLI overrides shadow the YAML value only for the keys that are explicitly passed; anything not overridden falls through to `config.yaml`. This means the YAML file remains the canonical source of truth for a given experiment configuration.

Example invocations:

```bash
# Full pipeline, default config
python -m src.train_zero_shot

# Ablation sweep: change one variable without touching YAML
python -m src.train_zero_shot --ner-epochs 10 --seed 44

# Transfer to new language pair, skip realignment
python -m src.train_zero_shot --target-langs fi sw ko --skip-contrastive

# Point to a custom config for a different experiment
python -m src.train_zero_shot --config configs/config_large.yaml
```

The `yaml` dependency was already present in the environment (PyYAML is a transitive dependency of `datasets`). No new packages were added.

---

## 12. Inference Demo

### 12.1 Design

`src/demo.py` provides a self-contained inference interface that loads a trained checkpoint and tags arbitrary text in any language - no retraining, no labels needed. It is the primary entry point for demonstrating the system to an external audience.

Three operating modes are supported:

| Mode | Invocation | Use case |
|------|-----------|----------|
| Single-shot CLI | `--text "..."` | Quick tagging in a terminal |
| Interactive CLI | `--interactive` | Exploratory multi-line session with ANSI-colored entity highlights |
| Web UI | `--web` | Gradio-based visual demo (optional dep: `pip install gradio`) |

### 12.2 Tokenization and Label Alignment

Inference uses XLM-R's `▁` word-start convention to reconstruct word-level predictions from subword tokens. The first subword token's predicted label is taken as the word label, matching the subword alignment convention used in `data_processing.py` during training. Special tokens (`[CLS]`, `[SEP]`) and zero-offset mappings are excluded before reconstruction.

Entity spans are accumulated from consecutive B-/I- tags with matching entity types and presented as a structured entity list below the token-colored output.

### 12.3 Usage Examples

```bash
# Tag a sentence, with ANSI color codes
python -m src.demo --text "Angela Merkel visited Berlin last Tuesday."

# Interactive session (Ctrl-D to quit)
python -m src.demo --interactive

# Gradio web UI on localhost
python -m src.demo --web

# Different checkpoint or config
python -m src.demo --checkpoint checkpoints/ner_checkpoint_B.pt --text "Kerem İstanbul'da."

# CI / logging (no ANSI)
python -m src.demo --no-color --text "Samsung opened a new office in Seoul."
```

### 12.4 Output Format

CLI output for an English sentence:
```
[B-PER]Angela [I-PER]Merkel visited [B-LOC]Berlin last Tuesday .

Entities found:
  Angela Merkel  [PER]
  Berlin         [LOC]
```

The Gradio mode renders HTML with colored `<mark>` spans and superscript entity-type labels.

---

## 13. Unit Tests

### 13.1 Coverage

`tests/test_data_pipeline.py` contains 24 unit tests covering the four most critical components of the data pipeline:

| Test class | What it tests | n |
|---|---|---|
| `TestSubwordLabelAlignment` | `tokenize_and_align()`: first-subword labeling, continuation masking, IOB2 span preservation, output length | 6 |
| `TestCodeSwitchAugmenter` | Label immutability under augmentation, switch_prob=0/1 boundary cases, output length invariant, capitalization transfer | 5 |
| `TestBilingualLexiconParsing` | Tab-separated, space-separated, and mixed-separator formats; malformed-line skipping; case normalization; multi-translation accumulation | 6 |
| `TestOpus100ConfigNames` | Alphabetical config-name ordering for all 5 language pairs used in the project plus symmetry property | 6 |
| `TestDatasetCastRoundTrip` | `Dataset.from_list()` int64 → ClassLabel cast before `concatenate_datasets()` | 1 |

All tests run without network access, GPU, or model downloads. The `TestSubwordLabelAlignment` suite instantiates the real `xlm-roberta-base` tokenizer (cached locally after initial download) to test against actual subword behavior.

### 13.2 Test Design Rationale

Tests are focused on behavior that is hard to catch from end-to-end training runs:

- **Subword alignment**: a silent label-offset bug would not crash training but would train on shifted targets; 6 tests establish the invariant across single-subword, multi-subword, and special-token cases.
- **Lexicon parsing**: the space-vs-tab separator bug was a real production failure (0 German code-switching entries); the regression tests document both separator types explicitly.
- **Dataset cast**: the `ClassLabel` / `int64` mismatch causes a hard crash only at concatenation time, which is late in the pipeline; the test catches it at the unit level.
- **OPUS-100 naming**: alphabetical config name ordering is a non-obvious external API contract; a wrong guess (`tr-en` instead of `en-tr`) raises a `datasets` error at download time with a confusing message.

### 13.3 Running Tests

```bash
# All tests
pytest tests/ -v

# Single class
pytest tests/test_data_pipeline.py::TestSubwordLabelAlignment -v

# With coverage (requires pytest-cov)
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 14. Final Results and Discussion

This section synthesizes findings from Sections 4--13 into a unified view. It is intended as the primary reference for understanding what the project found, why, and what it means.

### 14.1 Main Results Table

**Table 1. Zero-shot F1 across all configurations and languages (WikiANN test set).**

| Config | Description | Turkish | German | Arabic | Avg (tr/de/ar) |
|--------|-------------|---------|--------|--------|----------------|
| Prototype | Frozen probe, broken contrastive | 0.330 | -- | -- | -- |
| A | Frozen linear probe (revised backbone) | 0.543 | 0.538 | 0.221 | 0.434 |
| B | Full fine-tune only (no auxiliary) | 0.758 | 0.753 | 0.495 | 0.669 |
| C | Full fine-tune + contrastive | 0.736 | 0.745 | 0.447 | 0.642 |
| D | Full fine-tune + code-switch | 0.740 | 0.743 | 0.466 | 0.650 |
| **E** | **Full pipeline (B+C+D)** | **0.768** | **0.744** | **0.456** | **0.656** |
| B (multi-seed) | Config B, seeds 42/43/44 | 0.756 ± 0.011 | 0.754 ± 0.004 | 0.483 ± 0.029 | 0.664 ± 0.010 |

*(Bold on the E row marks it as the flagship/headline configuration used throughout Sections
4, 6–13, not a per-column maximum - see Section 5.2 for the config with the actual column
maximum on each metric, which is B for German, Arabic, and average, and E only for Turkish.)*

**Table 2. Extended language coverage (config E checkpoint, zero-shot eval only).**

| Language | Script | Typological family | F1 |
|---|---|---|---|
| English (val) | Latin | Germanic | 0.828 |
| Turkish | Latin | Turkic | 0.768 |
| Finnish | Latin | Uralic | 0.758 |
| German | Latin | Germanic | 0.744 |
| Swahili | Latin | Bantu | 0.675 |
| Russian | Cyrillic | Slavic (Indo-European) | 0.610 |
| Korean | Hangul | Koreanic | 0.509 |
| Indonesian | Latin | Austronesian | 0.497 |
| Arabic | Arabic | Semitic | 0.456 |

Indonesian and Russian (last two rows) were added specifically as disconfirming controls for the
script hypothesis below (Sections 4.6–4.8) - see Finding 2 for why they change the conclusion.

**Table 3. Few-shot recovery (starting from config E checkpoint).**

| k | Turkish F1 | German F1 | Arabic F1 |
|---|---|---|---|
| 0 (zero-shot) | 0.768 | 0.744 | 0.456 |
| 10 | 0.760 | 0.751 | **0.683** (+0.227) |
| 50 | 0.819 | 0.762 | 0.726 |
| 100 | 0.828 | 0.793 | 0.747 |
| 500 | 0.861 | 0.811 | 0.787 |

### 14.2 Finding 1: Full Backbone Fine-Tuning Is the Dominant Lever

The frozen linear probe (config A, avg 0.434) establishes that XLM-R already transfers cross-lingually without any task-specific training -- the gap to the full system is almost entirely capacity, not representation quality. Unlocking full backbone fine-tuning (config B) recovers that capacity (avg 0.669), a +0.235 gain that is ~24x larger than the multi-seed noise floor (std 0.010) and therefore robustly real. The contrastive and code-switch additions add ≤0.03 on top of B, within ~3x the noise floor -- plausible but unconfirmed at n=3. The most defensible claim after multi-seed validation: this is a full backbone fine-tuning problem, not a cross-lingual alignment problem.

### 14.3 Finding 2 (Revised): The Transfer Gap Correlates with Script but Resists Three Explanatory Hypotheses

The first six languages tested (en/de/fi/tr/sw + the original three) initially suggested a clean story: every Latin-script language landed in a tight 0.67--0.83 band regardless of typological distance from English, while the two non-Latin-script languages (Korean, Arabic) were the worst performers by a wide margin. Two additional controls, deliberately chosen to stress-test this claim (Section 4.6), broke it: **Indonesian (Latin script, Austronesian) scored 0.497 - below Korean and barely above Arabic**, directly falsifying "Latin script implies safe transfer." **Russian (Cyrillic, but Indo-European) scored 0.610 - clearly better than Arabic/Korean but clearly below the Latin cluster**, landing in between rather than confirming either side.

Two further mechanistic hypotheses were then tested against all 8 languages (Section 4.7) to see if either could explain what script alone could not: **subword fragmentation rate** (Pearson r = −0.150 against F1, wrong-signed, contradicted by Finnish's high fragmentation/high F1 and Indonesian's low fragmentation/low F1) and **XLM-R's published per-language pretraining corpus size** (Pearson r = −0.214 raw / −0.235 log, wrong-signed, contradicted by Swahili's tiny 1.6 GiB allocation transferring well and Russian's enormous 278 GiB allocation transferring only middlingly). Neither hypothesis survived its own disconfirming evidence.

**Revised conclusion: the zero-shot transfer gap is real, large (1.68x range across 8 languages), and correlates loosely with script - but is not reducible to script, subword fragmentation, or pretraining corpus size individually.** This is now reported as a converging null result across three independently falsified hypotheses (Section 4.8) rather than a single confirmed mechanism. The error analysis (Arabic's ORG over-prediction, Section 6) and the embedding alignment finding (Arabic's smaller contrastive gap, Section 7) remain valid, specific observations about Arabic - but they should not be read as validating a general "script explains it" theory, since that theory did not survive testing against Indonesian and Russian. The most defensible remaining hypothesis, unresolved by this project's evidence, is that WikiANN's per-language annotation quality (a silver-standard, Wikipedia-hyperlink-derived dataset known to vary in reliability across languages) or some other per-language property not captured by script, fragmentation, or corpus size drives the residual pattern.

### 14.4 Finding 3: The Arabic Gap Is Addressable, But Needs Supervision

Every zero-shot intervention tried in this project -- contrastive alignment, code-switch augmentation, projection-head routing, lexicon-proxy translate-train -- moves Arabic F1 by ≤0.03, within the noise floor of the multi-seed estimate. k=10 labeled Arabic examples (+0.227, from 0.456 to 0.683) outperforms the sum of all zero-shot techniques by an order of magnitude. By k=500 examples (2.5% of the English training set), Arabic (0.787) is within 0.07--0.08 of German (0.811) and Turkish (0.861). The gap is not unfixable -- it is not addressable by unsupervised alignment engineering at this scale. For practitioners: if the target language is non-Latin-script, budget for a small labeled set. For researchers: the Arabic ORG over-prediction pattern (Section 6) and the reduced contrastive alignment gap (Section 7) suggest the representations themselves are misaligned for Arabic, not just the classifier; the right intervention is likely something that acts on the token-level representation geometry, not sentence-level contrastive loss.

### 14.5 Finding 4: Contrastive Alignment Works at Its Own Objective But Doesn't Transfer to NER

The embedding visualization (Section 7) confirms that Phase 1 contrastive training is not wasted: the true-pair vs. shuffled-pair cosine similarity gap jumps from ~0.006 (chance) to 0.56--0.64 after contrastive training, a large and clearly real effect. Yet the ablation (Section 5) shows no confirmed downstream NER benefit. The projection-head routing probe (Section 7.4) rules out the most direct resolution (making the head reachable by NER training). The most likely explanation: the projection head aligns pooled sentence-level CLS representations, but NER classification operates on per-token states with no access to that geometry. Any benefit to the backbone's token representations via shared-weight updates is apparently too small to clear the 0.010 multi-seed noise floor. This is a well-known pattern in the contrastive learning literature (sentence-level alignment not automatically transferring to token-level tasks) and is worth flagging for any future cross-lingual NER work that uses contrastive pre-alignment.

### 14.6 Finding 5: Lexicon-Substitution Translate-Train Is Worse Than Zero-Shot Transfer

The translate-train baseline (Section 9) consistently underperforms zero-shot for all three languages (tr −0.033, de −0.022, ar −0.017). This is attributed to the static lexicon substitution producing partially-translated, morphologically incorrect, word-order-preserved pseudo-target text -- worse signal than clean English plus XLM-R's pretrained cross-lingual transfer. This does not generalize to NMT-based translate-train (which the literature typically finds competitive with or superior to zero-shot), and is reported as a result specific to the lexicon-substitution simplification, not as a claim about the strategy.

### 14.7 Key Negative Results

This project deliberately documents negative results alongside positive ones:

1. **Contrastive alignment does not improve NER F1** (Sections 5, 7, 7.4) -- even with the alignment gap confirmed large and real.
2. **Code-switch augmentation does not significantly improve NER F1** (Section 5.4) -- single-seed negative deltas do not survive multi-seed variance analysis.
3. **Projection-head routing closes no gap** (Section 7.4) -- clean null result, delta −0.0014 vs. noise floor 0.010.
4. **Lexicon-substitute translate-train underperforms zero-shot** (Section 9) -- attributed to the approximation, not the strategy.
5. **The initial "full-finetune-alone beats the full pipeline" ablation conclusion did not survive multi-seed testing** (Section 5.4) -- this is the most important methodological lesson of the project: single-seed ablation claims in the ≤0.03 delta range require multi-seed validation before reporting.
6. **Three candidate explanations for the zero-shot transfer gap were tested and all three failed** (Sections 4.6–4.8): script identity was falsified by Indonesian and complicated by Russian; subword fragmentation rate correlated at r = −0.150 (wrong-signed) with sharp counterexamples; XLM-R's pretraining corpus size correlated at r = −0.214/−0.235 (wrong-signed) with sharp counterexamples. This revises the originally reported "script drives the gap" finding (Section 4.6) from confirmed to falsified, and is the project's most consequential negative result -- it converts what looked like the headline positive finding into a rigorously-tested null result instead.

### 14.8 Figure Index

All figures generated by `python -m src.experiment_tracker` in `results/plots/`:

| File | Content | Key takeaway |
|------|---------|--------------|
| `ablation_bar.png` | Zero-shot F1 per config × language | Full fine-tuning dominates; auxiliary techniques within noise |
| `multiseed_errorbar.png` | Config B mean ± std, individual seeds | Arabic variance 3-7x higher than tr/de |
| `few_shot_curve.png` | F1 vs. k labeled examples | Arabic k=10 jump the most striking single data point in the project |
| `alignment_gap.png` | Before vs. after contrastive gap | Gap large and real; dissociation with NER F1 is the puzzle |
| `language_coverage.png` | All 9 languages (English val + 8 zero-shot targets) by F1, with script annotation | Latin-script cluster is NOT clean once Indonesian/Russian are added -- see Section 4.6 |
| `translate_train_vs_zero_shot.png` | Lexicon proxy vs. config E | Proxy underperforms uniformly |
| `fragmentation_vs_f1.png` | Subwords/word vs. F1 scatter, all 8 languages | r = −0.150, no visible trend -- fragmentation does not explain the gap |
| `corpus_size_vs_f1.png` | CC-100 GiB (log scale) vs. F1 scatter, all 8 languages | r = −0.235 (log), no visible trend -- pretraining volume does not explain the gap |

### 14.9 Codebase Overview

| File | Role |
|------|------|
| `src/contrastive_model.py` | `ZeroShotAligner`, `InfoNCELoss`, `NERClassifier` |
| `src/data_processing.py` | `MultiLingualNERPipeline`, `ParallelSentenceDataset`, `BilingualLexicon`, `CodeSwitchAugmenter` |
| `src/train_zero_shot.py` | `ZeroShotNERTrainer`, CLI entry point (argparse + YAML) |
| `src/run_ablation.py` | 4-way ablation study runner |
| `src/run_multiseed.py` | Multi-seed validation for config B |
| `src/error_analysis.py` | Confusion matrices + qualitative examples |
| `src/visualize_alignment.py` | Embedding alignment gap visualization |
| `src/eval_extra_languages.py` | Extended language evaluation (ko/fi/sw/id/ru) |
| `src/run_probe.py` | Projection-head routing probe (config F) |
| `src/few_shot_curve.py` | Few-shot recovery curve (k=0/10/50/100/500) |
| `src/translate_train_baseline.py` | Lexicon-substitution translate-train baseline |
| `src/analyze_fragmentation.py` | Subword fragmentation vs. F1 hypothesis test |
| `src/experiment_tracker.py` | Unified plot generation + summary CSV |
| `src/demo.py` | Multilingual inference demo (CLI + Gradio) |
| `tests/test_data_pipeline.py` | 24 unit tests, no network/GPU required |
| `configs/config.yaml` | Canonical hyperparameter defaults |

### 14.10 Limitations and Future Directions

1. **Single source language**: Training uses English-only WikiANN. Extending to multiple source languages (or a mixture) would test whether the pattern generalizes beyond English-centric transfer.
2. **No proper translate-train baseline**: The lexicon-substitution proxy is not a fair comparison to NMT-based translate-train. A proper baseline (e.g., using mBART or NLLB for sentence-level translation) remains outstanding.
3. **Full CKA analysis absent (partially addressed 2026-07-01, see Section 17.2)**: a lighter mean-pooled token-level alignment-gap measurement now shows the contrastive objective does shift token representations, at roughly half the magnitude of the sentence-level shift, and that this real shift still does not move NER F1 - a more precise account than Sections 7.3-7.4 could give alone. A full CKA comparison across backbone layers (rather than a single mean-pooled cosine gap) would still sharpen this further and remains open.
4. **n=3 multi-seed validation**: 3 seeds is sufficient to show that small deltas are noise-dominated but not sufficient to confirm or rule out effects in the 0.02-0.03 range. Larger seed sets (n=5-10) would sharpen the ablation conclusions. The extended 8-language evaluation and Section 17 analyses are single-seed throughout.
5. **Arabic sub-analysis (partially addressed 2026-07-01, see Section 17.3)**: The ORG over-prediction pattern (Section 6) is a specific, falsifiable claim that has not been traced to a root cause. The entity-type-level factor analysis shows this is not Arabic-specific - ORG (and LOC) correlate weakly with every tested factor across all 8 languages, while PER correlates strongly with script (r=0.824) - suggesting Arabic's ORG bias is an instance of a broader, entity-type-level pattern rather than an Arabic-specific idiosyncrasy. The root mechanism (why ORG/LOC resist these factors while PER doesn't) remains unidentified.

---

## 15. Extended Paper Analyses: Phonological Distance, Entity-Type Breakdown, Token-Level Alignment, and Their Bridge

These three analyses were added to `PAPER.md` (not this dev log's own conclusions, which they extend rather than replace) after the user asked for the paper to add a "unique perspective" beyond replicating Lauscher et al. (2020). Fact-checking Lauscher et al.'s tasks first revealed they *do* test NER, on WikiANN (identical dataset), across 12 languages with 5 overlapping this project's 8 - correcting an earlier factual error in `PAPER.md`'s Related Work section - and that their strongest NER-specific predictor is phonological similarity, not script or corpus size. That discovery directly motivated analysis 1 below.

### 15.1 Phonological Distance (`src/analyze_phonological_distance.py`)

Using `lang2vec` (Littell et al., 2017) URIEL `phonology_knn` features, computed cosine distance from English for all 8 target languages and correlated against zero-shot F1:

| Language | Phon. distance | F1 |
|---|---|---|
| Swahili | 0.0909 | 0.6748 |
| Indonesian | 0.0909 | 0.4967 |
| Finnish | 0.1296 | 0.7578 |
| Russian | 0.1419 | 0.6102 |
| Turkish | 0.1818 | 0.7675 |
| German | 0.1942 | 0.7435 |
| Korean | 0.2538 | 0.5086 |
| Arabic | 0.2994 | 0.4563 |

Pearson r = -0.387 (`results/phonological_distance_analysis.json`) - correctly signed and the strongest of the 4 factors tested (script, fragmentation, corpus size, phonological distance), but far short of Lauscher et al.'s reported r≈0.78 (equivalent ≈-0.78 for distance). Sharp counterexample: Indonesian and Swahili are phonologically equidistant from English (0.0909, identical to 4 decimals) yet differ by 0.178 F1. A partial, directionally-consistent, but substantially weaker replication of Lauscher et al.'s specific NER finding.

### 15.2 Token-Level Representation Alignment (`src/analyze_token_alignment.py`)

Extends Section 7's CLS-level alignment-gap methodology to mean-pooled RAW backbone token embeddings (`get_token_embeddings()`, what NER actually reads, bypassing the projection head) for the same OPUS-100 pairs, before vs. after Phase 1 contrastive training:

| Language | Token-level gap (before) | Token-level gap (after) | CLS-level gap (after, Section 7) |
|---|---|---|---|
| Turkish | 0.008 | 0.311 | 0.629 |
| German | 0.008 | 0.362 | 0.636 |
| Arabic | 0.004 | 0.363 | 0.558 |

(`results/token_alignment_metrics.json`) **Token representations DO shift substantially** (chance-level before, 0.31-0.36 after) - refuting the strongest form of "alignment never reaches tokens." But the shift is **roughly half the CLS-level magnitude**, a real measurable attenuation. **Arabic's ranking reverses**: smallest CLS-level gap of the three but largest token-level gap. Refines the Section 7.4 explanation with a direct measurement rather than only the indirect projection-head-routing probe: alignment does reach tokens, at reduced strength, and even that real shift doesn't move NER F1 (Section 5's ablation).

### 15.3 Entity-Type-Level Factor Analysis (`src/eval_per_entity_type.py`)

Re-evaluated config E checkpoint on all 8 languages capturing per-entity-type F1 (PER/ORG/LOC), then recomputed all 4 hypotheses' correlations per entity type instead of only aggregate:

| Entity type | Script (Latin=1) | Fragmentation | Corpus size | Phon. distance |
|---|---|---|---|---|
| PER | r=0.824 | r=-0.517 | r=-0.515 | r=-0.403 |
| ORG | r=0.356 | r=0.036 | r=0.258 | r=-0.275 |
| LOC | r=0.476 | r=0.139 | r=-0.202 | r=-0.224 |
| Aggregate (micro) | r=0.664 | r=-0.150 | r=-0.214 | r=-0.387 |

(`results/per_entity_type_analysis.json`) **PER transfer correlates strongly with script (r=0.824)** - stronger than any aggregate-level correlation found in Sections 4.6-4.8/15.1. **ORG and LOC correlate weakly with everything** (all |r|<0.5). The aggregate "no single factor explains it" result is not uniform - it is substantially an ORG/LOC phenomenon. Connects to Section 6: Arabic's ORG over-prediction bias may be one instance of a broader pattern (ORG being the entity type most resistant to all 4 factors, across all 8 languages), not an Arabic-specific idiosyncrasy.

### 15.4 Entity-Type-Conditioned Token Alignment (`src/analyze_entity_type_alignment.py`)

Added after the user asked what else could be analyzed further, once Sections 15.1-15.3 were complete. Bridges 15.2 (token-level alignment, averaged uniformly across all tokens) and 15.3 (PER correlates with script, ORG/LOC don't) by testing whether the token-level alignment gap itself differs by entity type - the natural hypothesis connecting the two. Method: the fully-trained config E model (backbone + NER head) is used PURELY as a labeling tool to predict a PER/ORG/LOC/O tag for every subword position on both sides of 1,000 OPUS-100 sentence pairs per language (tr/de/ar); the "before"/"after" alignment-gap embeddings come from separate models (fresh pretrained / `alignment_checkpoint.pt`), exactly as in 15.2, decoupling the tagging model from the measured model.

| Language | PER gap (n) | ORG gap (n) | LOC gap (n) | O gap (n) |
|---|---|---|---|---|
| Turkish | 0.303 (110) | 0.340 (570) | 0.313 (26) | 0.320 (623) |
| German | 0.348 (74) | 0.374 (595) | 0.320 (38) | 0.377 (593) |
| Arabic | 0.273 (51) | 0.356 (601) | 0.256 (31) | 0.334 (530) |

(`results/entity_type_alignment.json`) **The bridging hypothesis is refuted.** PER does NOT show a larger alignment gap than ORG in any of the 3 languages - if anything ORG shows the largest or near-largest gap in all three. Whatever makes PER transfer more predictably than ORG/LOC (15.3), it is not that PER-relevant token representations are more strongly aligned by contrastive training. This deepens the Section 7/15.2 dissociation: representation-level alignment and downstream classification quality are decoupled per entity type too, not just in aggregate - ORG-tagged tokens show real, substantial alignment yet ORG remains the entity type least explained by any tested factor and most prone to over-prediction errors. Incidental observation (not statistically validated, no gold labels on OPUS-100): the tagger predicts ORG far more often than PER/LOC on this out-of-domain text, uniformly across all 3 languages (570-601 vs. 26-110 sentences) - consistent with, though not proof of, the broader ORG over-prediction pattern from Section 6, and notably NOT concentrated in Arabic the way the gold-labeled error rate is.

### 15.5 Positioning

These 4 analyses are presented in `PAPER.md` as the paper's genuine additive content beyond replicating Lauscher et al.: (1) directly testing their own strongest identified NER predictor rather than only generic proxies, made possible by the task/dataset/language overlap; (2) changing the unit of analysis from aggregate F1 to entity type, revealing structure the aggregate view conceals; (3) measuring inside the model's representations directly rather than only correlating outputs against external factors; (4) combining (2) and (3) to test a specific bridging hypothesis, which turns out to be false - itself a genuine finding, since it rules out the most obvious mechanistic explanation connecting the two most novel results in the paper. None of the four were performed by Lauscher et al. or, to our knowledge, elsewhere in the cited literature for this specific system. While integrating 15.4, also caught and fixed a real bug in `PAPER.md`: Sections 5.6/6.3 had referenced "Section 6's error analysis" as a self-reference within the paper, but PAPER.md's own Section 6 is about zero-shot interventions/few-shot, not error analysis - the Arabic ORG over-prediction finding was never actually presented in PAPER.md itself, only in DEVELOPMENT.md. Fixed by inlining the actual numbers (O→ORG 16.3%, PER→ORG 26.0%, LOC→ORG 37.8%) at first mention in PAPER.md Section 5.6 rather than leaving a dangling cross-reference to the wrong section.

---

## 16. Changelog

| Date | Change |
|---|---|
| 2026-06-29 | Initial codebase analyzed: identified non-functional data pipeline (signature mismatches, missing DataLoader, no evaluation/checkpointing). |
| 2026-06-29 | Fixed data pipeline: `PairedNERDataset`, corrected `xlm-roberta-base` identifier, `seqeval`-based evaluation, checkpoint save/load. Added `requirements.txt` entries, `configs/config.yaml`, `check.py` sanity script. |
| 2026-06-29 | First successful end-to-end run. Identified `wikiann` dataset script incompatibility with `datasets>=5.0.0`; repointed to `unimelb-nlp/wikiann`. |
| 2026-06-29 | Diagnosed non-learning contrastive phase (flat loss at chance level, `ln(16)`) and capacity-limited linear-probe NER head (English F1 0.329, Turkish zero-shot F1 0.330). |
| 2026-06-30 | Implemented real parallel-corpus contrastive alignment via OPUS-100, full backbone fine-tuning with layer-wise LR decay, and CoSDA-ML code-switching augmentation via MUSE bilingual lexicons. Extended zero-shot evaluation to Turkish, German, and Arabic. |
| 2026-06-30 | First run of revised pipeline: contrastive alignment converged on all three languages (en-tr loss 2.22→0.24; en-de, en-ar <0.3) using real OPUS-100 pairs. Run then failed in Phase 2 with two bugs: (1) `en-de` MUSE lexicon parsed 0 entries due to space- vs tab-separated format inconsistency, silently disabling German code-switching; (2) `concatenate_datasets` rejected merging original/augmented WikiANN splits due to a `ner_tags` feature-schema mismatch (`ClassLabel` vs inferred `int64`). Both fixed (whitespace-tolerant lexicon parsing; explicit feature cast before concatenation) and verified via standalone functional test. |
| 2026-06-30 | Full pipeline completed successfully end-to-end. Results: English val F1 0.8278 (up from 0.3289), Turkish zero-shot F1 0.7675, German zero-shot F1 0.7435, Arabic zero-shot F1 0.4563 (up from single-language baseline of 0.3297 on Turkish only). Average zero-shot F1 across tr/de/ar: 0.6558. Confirmed original diagnosis: prototype's weak results were caused by non-functional contrastive signal and frozen-probe capacity limits, not an inherent ceiling on cross-lingual transfer. Arabic transfers notably weaker than Turkish/German (hypothesis: script + templatic morphology divergence); flagged for ablation investigation. Set up full project roadmap as tracked tasks (ablation study, multi-seed trials, error analysis, embedding viz, more languages, few-shot curve, translate-train baseline, experiment tracking, CLI, demo, tests) and a project-local `MEMORY.md` for context continuity across chat compaction. |
| 2026-06-30 | Built 4-way ablation study (`src/run_ablation.py`): added `freeze_backbone`, `use_code_switch`, `checkpoint_name` parameters to `ner_finetune()` and `checkpoint_name` to `contrastive_train()`/`zero_shot_eval_multi()` so each lever can be toggled independently without checkpoint collisions. Code review caught and fixed a bug where `contrastive_train()` would have silently overwritten the headline run's `alignment_checkpoint.pt`. Ran configs A (frozen baseline), B (full-finetune only), C (+contrastive), D (+code-switch); reused E (full pipeline) from the completed run. **Result: full backbone fine-tuning (B) is responsible for nearly all of the gain (+0.235 avg F1 over frozen baseline), while contrastive alignment and code-switch augmentation each slightly underperform full-finetune-alone, individually and combined** (avg F1: B 0.669 > E 0.656 > D 0.650 > C 0.642 > A 0.434). Notably, B also achieves the best Arabic score of any config (0.495), so neither auxiliary technique closes the Arabic transfer gap - pointing toward the gap being intrinsic to XLM-R's pretrained Arabic representation rather than something the alignment/augmentation recipe can fix at this scale. Full analysis in Section 5; flagged as single-seed and provisional pending Task #3 (multi-seed trials). |
| 2026-06-30 | Added `set_seed()` (`src/train_zero_shot.py`) and ran config B for 3 seeds (`src/run_multiseed.py`, Section 5.4) to test whether the ablation's small negative deltas for C/D/E vs. B were real or noise. **Result: B's own seed-to-seed std (avg 0.0099) is the same order of magnitude as most of the ablation deltas (−0.013 to −0.026), so the "full-finetune-alone beats the full pipeline" claim does not hold up** - B vs. E (−0.013) and B vs. D (−0.019) are not clearly distinguishable from noise; B vs. C (−0.026, ~2.2 std) remains the most plausible real effect but is unconfirmed at n=3. Arabic showed 3-7x higher seed variance than Turkish/German (std 0.029 vs. 0.011/0.004), itself noted as a finding. The dominant-lever conclusion (full fine-tuning >> auxiliary techniques) is unaffected and stands at ~24x the observed noise floor. This is exactly the failure mode multi-seed validation is meant to catch; reported honestly rather than keeping the more dramatic single-seed claim. |
| 2026-06-30 | Built error analysis tooling (`src/error_analysis.py`): per-language 4x4 entity-type confusion matrices + qualitative mismatched examples on the config E checkpoint (Section 6). **Result: Arabic's weak zero-shot performance is driven by a specific, identifiable failure mode - systematic over-prediction of ORG** - true O/PER/LOC tokens are mispredicted as ORG at 2-4x the rate seen in Turkish/German (O→ORG 16.3% vs. 3-4%; PER→ORG 26.0% vs. ~5%; LOC→ORG 37.8% vs. ~11%), while ORG's own recall (0.902) is on par with tr/de. This refines the Section 4.4 script/morphology hypothesis into a concrete, falsifiable claim. Also identified a WikiANN corpus artifact (Wikipedia redirect-page markers `YÖNLENDİRME`/`تحويل` mistagged as entities) inflating error counts slightly across languages, and noted tr/de errors are mostly span-boundary mistakes while Arabic shows genuine type confusion. |
| 2026-06-30 | Built embedding alignment visualization (`src/visualize_alignment.py`, Section 7): t-SNE projection + cosine-similarity alignment gap (true-pair vs. shuffled-pair similarity, controlling for embedding-space anisotropy) comparing a fresh pretrained backbone against the headline run's `alignment_checkpoint.pt`. **Result: contrastive alignment is real and large** - the true/shuffled gap jumps from ~0.005-0.007 (chance level) before training to 0.558-0.636 after, confirming Phase 1 does pull true translation pairs together, independent of the raw InfoNCE loss curve. Arabic's gap (0.558) is measurably smaller than Turkish/German (0.629/0.636), a third independent line of evidence for the Arabic-specific pattern. **Most important finding: this result is dissociated from the Section 5.4 ablation** - contrastive alignment succeeds at its own sentence-level objective but this does not translate into a confirmed downstream NER F1 improvement, most plausibly because NER fine-tuning/eval never use the projection head this alignment is measured on (only the shared backbone benefits indirectly, and evidently not enough to clear the multi-seed noise floor). |
| 2026-06-30 | Extended zero-shot evaluation to Korean, Finnish, Swahili (`src/eval_extra_languages.py`, Section 4.6) using the existing config E checkpoint - pure inference, no retraining. Results: Finnish 0.7578, Swahili 0.6748, Korean 0.5086. **Decisive finding: script, not morphological distance from English, drives the Arabic/Korean gap.** Every Latin-script language tested (German, Finnish, Turkish, Swahili) lands in a tight 0.67-0.77 band regardless of typological distance - Swahili (Bantu, very distant from English) transfers almost as well as German (closely related), ruling out morphology as the dominant factor. The two non-Latin-script languages (Korean/Hangul, Arabic) are the two worst performers by a wide margin. This converges with the Section 6 (ORG over-prediction) and Section 7 (weaker contrastive alignment) findings on the same root cause: XLM-R's representations are specifically weaker for non-Latin scripts. Disentangling script from morphology via Swahili as a clean control was the key move here. Added new Task #14 (not in original roadmap): probe whether routing the projection head into NER training closes the Section 7 alignment-to-F1 gap - requires a new training run, deferred until after this task. |
| 2026-06-30 | Built and ran the Task #14 projection-head routing probe (`src/run_probe.py`, `get_projected_token_embeddings()` in `contrastive_model.py`, Section 7.4): config F repeats config C's recipe (contrastive + full fine-tune, no code-switch) but forces NER training/eval to classify `projection_head(token_embeddings)` instead of raw backbone embeddings. **Result: no effect.** F vs. C avg delta is −0.0014, an order of magnitude smaller than config B's multi-seed std (0.0099, Section 5.4) - not distinguishable from noise in either direction. This rules out the Section 7.3 hypothesis in its most direct form: making the projection head reachable by NER training does not close the alignment-to-F1 gap. Revised explanation offered (Section 7.4): the projection head aligns pooled [CLS]-vector geometry, which doesn't necessarily transfer to individual token vectors when the same transformation is applied per-token; any real benefit from contrastive training more likely flows through small shared-backbone weight changes, not the projection head itself. Proper resolution would need representation-similarity analysis (e.g. CKA) rather than another architectural variant - noted as a natural follow-up, out of scope for this task. |
| 2026-06-30 | Built few-shot recovery curve (`src/few_shot_curve.py`, `few_shot_finetune()` in `train_zero_shot.py`, Section 8): fine-tuned the config E checkpoint independently on k=10/50/100/500 labeled target-language examples per language, evaluated on the full test set. **Result: a handful of labeled Arabic examples closes most of the gap that no zero-shot technique in this project could.** k=10 alone raises Arabic F1 from 0.456 to 0.683 (+0.227, recovering ~75% of the gap to tr/de); by k=500, Arabic (0.787) is within 0.07-0.08 of Turkish (0.861) and German (0.811). This single fine-tuning pass on 500 examples (2.5% of the English training set) outperforms every zero-shot intervention tested anywhere in this project (contrastive alignment, code-switching, projection-head routing - all ≤±0.03 F1, mostly within noise) by an order of magnitude. Reframes the Section 4.6 script-based explanation: the gap is intrinsic to XLM-R's pretrained representation, but that means it needs supervision to fix, not further unsupervised alignment engineering - a directly actionable conclusion for deployment decisions. |
| 2026-06-30 | Built translate-train baseline (`src/translate_train_baseline.py`, `load_ner_dataset_translated()`/`translate_train_finetune()`, Section 9): a lexicon-substitution proxy (every English word substituted via MUSE dictionaries at switch_prob=1.0, no mixing with real English) standing in for a full NMT-based translate-train pipeline, which was out of scope here. **Result: this baseline underperforms zero-shot transfer for all three languages** (tr −0.033, de −0.022, ar −0.017), consistently negative across all three typologically different languages despite being single-seed. Attributed to the documented limitation: static word-for-word substitution produces disfluent, partially-untranslated, morphologically incorrect pseudo-target-language text, which is worse training signal than clean English relying on XLM-R's pretrained cross-lingual transfer. Explicitly NOT interpreted as "translate-train as a strategy underperforms zero-shot" - the literature generally finds the opposite for properly-implemented (NMT-based) translate-train; this result is specific to the lexicon-substitution simplification. Completes Tier 2 of the project roadmap. |
| 2026-07-01 | Built experiment tracking dashboard (`src/experiment_tracker.py`, Section 10): reads all 8 result JSON files and generates 6 publication-ready plots (`results/plots/`) plus a 54-row normalized summary CSV (`results/summary.csv`). No new training; purely a consolidation and visualization layer. The language coverage and few-shot curve plots most clearly communicate the project's two central findings - script-driven transfer gap, and the gap's rapid collapse under minimal supervision. |
| 2026-07-01 | Replaced hardcoded `__main__` config dict in `train_zero_shot.py` with an argparse CLI wired to `configs/config.yaml` (Task #10, Section 11). All hyperparameters now overridable via flags (`--ner-epochs`, `--seed`, `--target-langs`, etc.); YAML remains the canonical default. No class-level changes; `--help` verified clean. |
| 2026-07-01 | Built inference demo (`src/demo.py`, Section 12): CLI (single-shot `--text` and `--interactive`) + optional Gradio web UI (`--web`). Loads `ner_checkpoint.pt` and tags arbitrary multilingual text with word-level IOB2 predictions. ANSI-colored entity output in terminal; HTML `<mark>` spans in web mode. Checkpoint-agnostic via `--checkpoint` flag; compatible with any config via `--config`. |
| 2026-07-01 | Added 24 unit tests (`tests/test_data_pipeline.py`, Section 13, Task #12): covers subword label alignment (6), code-switch augmenter (5), bilingual lexicon parsing - both separator formats + edge cases (6), OPUS-100 config name ordering (6), and Dataset cast round-trip (1). All 24 pass; no network or GPU required. |
| 2026-07-01 | Wrote final results and discussion section (Section 14, Task #13): synthesizes all 13 prior tasks into unified findings tables, 5 numbered findings, figure index, codebase overview, and limitations. Updated Abstract to cover full project scope. Marked project Status: Complete. |
| 2026-07-01 | Fixed document structure: resolved duplicate section numbers (two "Section 8"s, two "Section 10"s) caused by appending new sections without renumbering; moved Changelog and References to the true end of the document (now Sections 15 and 16) and reordered Changelog rows into correct chronological sequence (the translate-train baseline row and the four Tier-3 rows were previously out of date order). Corrected a bolding error in the Section 5.2 ablation table where the column-maximum highlighting for Turkish and German was transposed between configs B and E. |
| 2026-07-01 | Independent audit pass (via subagent) over the full document for numeric consistency, cross-reference correctness, table bolding conventions, and markdown syntax. Found and fixed a 4th-decimal numeric mismatch (Section 7.4's config C average F1 was stated as 0.6425, recomputed from `ablation_results.json` as 0.6424 - matching Section 5.2). Added clarifying notes to three tables (Sections 4.6, 8.2, 14.1 Table 1) whose bold highlighting marks something other than a column maximum (newly added languages, largest single-step jump, and the flagship config respectively), since Section 5.2 nearby uses bold strictly for column maxima and the inconsistent convention could otherwise read as another error. No other numeric, cross-reference, or markdown-syntax issues found. |
| 2026-07-01 | Stress-tested the Section 4.6 "script drives the transfer gap" finding with two additional disconfirming controls (`src/eval_extra_languages.py` extended to 5 languages): Indonesian (Latin script, Austronesian) and Russian (Cyrillic script, Indo-European). **Result: the clean script thesis does not survive.** Indonesian (0.4967) scored below Korean and near Arabic despite Latin script, directly falsifying "Latin script implies safe transfer"; Russian (0.6102) landed between the Latin cluster and the Arabic/Korean cluster rather than confirming either side. |
| 2026-07-01 | Tested two further mechanistic hypotheses for the residual pattern (`src/analyze_fragmentation.py`, new Section 4.7): subword fragmentation rate (Pearson r = −0.150 vs. F1, wrong-signed, contradicted by Finnish's high-fragmentation/high-F1 and Indonesian's low-fragmentation/low-F1) and XLM-R's published per-language CC-100 pretraining corpus size from Conneau et al. (2020) Table 6 (Pearson r = −0.214 raw / −0.235 log, wrong-signed, contradicted by Swahili's 1.6 GiB allocation transferring well and Russia's 278 GiB allocation transferring only mid-pack). **Neither hypothesis explains the pattern.** Added Section 4.8 synthesizing all three falsified hypotheses as a converging null result. Revised Section 14.3 (previously "Finding 2: Script Drives the Gap") to reflect the falsification, added item 6 to the Section 14.7 negative-results list, updated Table 2 in Section 14.1, and rewrote the Abstract. Regenerated `results/plots/language_coverage.png` with all 9 languages and added two new plots (`fragmentation_vs_f1.png`, `corpus_size_vs_f1.png`) to `src/experiment_tracker.py`. New result files: `results/fragmentation_analysis.json`, `results/corpus_size_analysis.json`. This is the most consequential revision in the project: the original "decisive finding" (Section 4.6, 2026-06-30) is now explicitly superseded and reframed as the first of three falsified hypotheses, motivated by the user's request to "explore if it holds or not" before drafting an academic paper around it. |
| 2026-07-01 | Drafted a standalone academic paper (`PAPER.md`, Task #17) framed around the Section 4.6–4.8 falsification study as its central contribution, with the ablation/multi-seed results and few-shot recovery curve as supporting evidence. Structure: Abstract, Introduction, Related Work (Rahimi et al. 2019, Rust et al. 2021, and Muller et al. 2021 added as new citations specific to cross-lingual NER transfer and script/tokenizer effects), Method, two results sections (system validation; transfer-gap hypothesis testing), a section on the gap being addressable via few-shot supervision despite being unexplained, Discussion, Limitations, Conclusion. All numeric values cross-checked directly against source JSON files before finalizing (`ablation_results.json`, `multiseed_results.json`, `fragmentation_analysis.json`, `corpus_size_analysis.json`, `few_shot_curve.json`) - no discrepancies found. `PAPER.md` is a separate document from `DEVELOPMENT.md`; the latter remains the complete chronological dev log and is referenced from the paper's closing note for full reproducibility detail. |
| 2026-07-01 | User asked whether the paper's "no single factor explains it" finding is actually unique; honest answer was no - Lauscher et al. (2020, "From Zero to Hero: On the Limitations of Zero-Shot Language Transfer with Multilingual Transformers", EMNLP, pp. 4483-4499) already establishes the same general shape of result (no single factor predicts transfer quality; few-shot supervision is surprisingly effective) at much larger scale with proper statistical power, which this project's n=8 Pearson correlations do not have. Citation details verified via WebSearch before citing (author list, venue, page numbers) rather than relying on recalled/possibly-hallucinated details. Per user request, re-framed `PAPER.md` explicitly as an independent, small-scale replication supporting Lauscher et al. rather than implying novel discovery: new title, rewritten Abstract/Introduction/Related Work/Discussion/Conclusion all state the replication relationship up front, added Lauscher et al. as a dedicated Related Work paragraph and to References. This is a positioning change, not a numerical one - no results were altered, only how the paper frames its own contribution. |
| 2026-07-01 | User asked to add a genuine "unique perspective" to the paper beyond replication (Section 15, Tasks #18-22). While researching, discovered Lauscher et al. (2020) actually DOES test NER on WikiANN (contradicting the just-added Related Work claim that it didn't) across 12 languages, 5 overlapping this project's 8, with phonological similarity as their strongest NER predictor - fixed the factual error immediately. Then executed 3 new analyses: (1) phonological distance via `lang2vec`/URIEL (`src/analyze_phonological_distance.py`), the strongest of 4 factors tested (r=-0.387) but still far below Lauscher et al.'s reported effect, with a sharp Indonesia/Swahili tied-distance counterexample; (2) token-level (not just CLS-level) representation alignment measurement (`src/analyze_token_alignment.py`), showing contrastive alignment DOES shift token representations substantially (refuting "zero effect on tokens") but at roughly half the CLS-level magnitude, with Arabic's rank reversing between the two levels; (3) entity-type-level factor analysis (`src/eval_per_entity_type.py`), the most novel finding - PER correlates strongly with script (r=0.824, stronger than any aggregate correlation) while ORG/LOC correlate weakly with all 4 factors, showing the aggregate null result is substantially an ORG/LOC phenomenon. All 3 numerically cross-checked against source JSON before writing into `PAPER.md` (new Sections 5.5-5.7, 6.2) and this log (Section 15). Added `lang2vec` to `requirements.txt`. Updated Limitations items 3 and 5 to reflect partial resolution. These 3 analyses are the paper's actual unique/additive content, distinct from the Lauscher-replication framing established in the previous entry - both are true simultaneously: the paper replicates known broader claims AND contributes 3 specific analyses not present in the replicated study. |
| 2026-07-01 | User asked what could be analyzed further among the open discussion threads (Task #23, Section 15.4). Recommended and executed the natural bridge between the two newest analyses: does the token-level alignment gap (15.2) differ by entity type, which would explain why PER correlates with script but ORG/LOC don't (15.3)? Built `src/analyze_entity_type_alignment.py`: uses the fully-trained config E model purely as a labeling tool to predict PER/ORG/LOC/O tags on both sides of 1,000 OPUS-100 pairs per language, keeping the tagging model decoupled from the before/after embedding models (exactly as in 15.2). **Result: the hypothesis is refuted.** ORG shows the largest or near-largest token-level alignment gap in all 3 languages, not PER - the opposite of what would explain the entity-type correlation pattern. This deepens rather than resolves the representation-vs-classification-quality dissociation established in Section 7/15.2, now shown to hold per entity type too. Incidentally discovered the tagger over-predicts ORG on out-of-domain OPUS-100 text uniformly across all 3 languages (no gold labels available to validate as an error rate, flagged as a side observation). While integrating this into `PAPER.md`, also caught and fixed a real cross-reference bug: Sections 5.6/6.3 had referred to "Section 6's error analysis" as if it were a section within PAPER.md itself, but PAPER.md's own Section 6 covers zero-shot interventions/few-shot, not error analysis - the Arabic ORG over-prediction finding had never actually been presented in PAPER.md, only in DEVELOPMENT.md. Fixed by inlining the real numbers (O→ORG 16.3%, PER→ORG 26.0%, LOC→ORG 37.8%) at first mention rather than leaving a dangling wrong self-reference. Updated Abstract, Introduction, Discussion, Limitations (new item 8), and Conclusion to reflect a 4th additive analysis, not 3. |

---

## 17. References

- Pires, T., Schlinger, E., & Garrette, D. (2019). How Multilingual is Multilingual BERT? *ACL*.
- Wu, S., & Dredze, M. (2019). Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT. *EMNLP*.
- Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). *ACL*.
- Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. *EMNLP*.
- Qin, L., et al. (2020). CoSDA-ML: Multi-Lingual Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP. *IJCAI*.
- Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. *ACL*.
- Pan, X., et al. (2017). Cross-lingual Name Tagging and Linking for 282 Languages (WikiANN). *ACL*.
- Zhang, B., et al. (2020). Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation (OPUS-100). *ACL*.
- Lauscher, A., Ravishankar, V., Vulić, I., & Glavaš, G. (2020). From Zero to Hero: On the Limitations of Zero-Shot Language Transfer with Multilingual Transformers. *EMNLP*, pp. 4483-4499.
- Littell, P., Mortensen, D. R., Lin, K., Kairis, K., Turner, C., & Levin, L. (2017). URIEL and lang2vec: Representing Languages as Typological, Geographical, and Phylogenetic Vectors. *EACL*.
- Rahimi, A., Li, Y., & Cohn, T. (2019). Massively Multilingual Transfer for NER. *ACL*.
- Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models. *ACL*.
- Muller, B., Sagot, B., & Seddah, D. (2021). When Being Unseen from mBERT is just the Beginning: Handling New Languages With Multilingual Language Models. *NAACL*.
- Artetxe, M., Labaka, G., & Agirre, E. (2020). Translation Artifacts in Cross-lingual Transfer Learning. *EMNLP*.
