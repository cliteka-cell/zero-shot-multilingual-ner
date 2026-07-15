# No Single Factor Explains the Gap: Independent Evidence for Lauscher et al. (2020) from a Zero-Shot Cross-Lingual NER System

**Author:** Kerem
**Status:** Preprint draft

---

## Abstract

Lauscher et al. (2020) showed, across many languages and tasks, that zero-shot cross-lingual transfer with multilingual transformers does not reduce cleanly to any single easily-measured factor, and that inexpensive few-shot target-language supervision is surprisingly effective at closing transfer gaps that zero-shot methods cannot. For Named Entity Recognition (NER) specifically, evaluated on WikiANN, they identify phonological similarity to the source language as the strongest single predictor. This paper reports independent, smaller-scale evidence for both broader claims, obtained from a self-contained system (contrastive sentence alignment, full backbone fine-tuning, code-switching augmentation) built and evaluated separately, on the same task and dataset and a partially overlapping language sample. We first test script identity (Latin versus non-Latin), the most commonly cited single-factor explanation, using a design intended to isolate it from typological/morphological distance. Six languages support it cleanly; two additional controls chosen specifically to try to break this result - Indonesian (Latin script, distant family) and Russian (non-Latin script, family close to English) - do break it. Two further hypotheses, subword fragmentation rate and pretraining corpus size, also fail (Pearson r = -0.150 and -0.214 to -0.235). We then test Lauscher et al.'s own strongest predictor, phonological distance (via URIEL/lang2vec features): it correlates better than the other three (r = -0.387, correctly signed) but still falls well short of their reported r ≈ -0.78 equivalent, and has its own sharp counterexample (Indonesian and Swahili are phonologically equidistant from English yet differ by 0.18 F1). Breaking the analysis down by entity type reveals a pattern none of the four aggregate-level factors capture: PER (person) transfer correlates strongly with script (r = 0.824), while ORG and LOC correlate weakly with every factor tested (all |r| < 0.5) - the aggregate "no single factor explains it" result is substantially an ORG/LOC phenomenon, not a uniform one. We measure token-level (not just sentence-level) representation shift from contrastive training directly: it is real and non-zero (unlike a fresh pretrained baseline) but roughly half the magnitude of the sentence-level shift, and does not produce a measurable NER F1 gain regardless. We then test the natural hypothesis bridging these last two findings - that PER's stronger factor correlation is explained by PER-tagged tokens being more strongly aligned by contrastive training than ORG/LOC-tagged tokens - and refute it: ORG shows the largest or near-largest token-level alignment gap in all three languages tested, not PER, showing that representation-level alignment and downstream transfer quality are decoupled per entity type, not just in aggregate. Separately, ten labeled target-language examples close roughly three-quarters of the largest zero-shot gap observed, an order of magnitude larger than the effect of every zero-shot intervention tested combined. We frame the paper's contribution accordingly: independent replication of Lauscher et al.'s broader claims, a direct test of their specific strongest NER predictor on overlapping languages, and four analyses - entity-type decomposition, token-level representation measurement, their entity-type-conditioned combination, and phonological distance via URIEL - that go beyond what either study measured on its own.

---

## 1. Introduction

Cross-lingual transfer with multilingual pretrained encoders - mBERT, XLM-RoBERTa (Conneau et al., 2020) - allows a model fine-tuned only on labeled English data to perform reasonably on other languages with zero target-language supervision. This capability is valuable precisely because labeled data is scarce for most of the world's languages. But the transfer is well documented to be uneven: some target languages retain most of the source-language performance, others degrade sharply (Pires et al., 2019; Wu & Dredze, 2019; Rahimi et al., 2019).

**This paper is explicitly a small-scale, independently-obtained replication of a specific pair of claims from Lauscher et al. (2020).** Testing many languages and tasks against multiple candidate predictors of transfer quality (pretraining corpus size, typological distance, and others), Lauscher et al. found that no single easily-measured factor cleanly predicts zero-shot cross-lingual transfer performance, and separately, that inexpensive few-shot fine-tuning on a handful of target-language examples is surprisingly effective at closing gaps that zero-shot methods cannot. We did not set out to replicate this study; we built a zero-shot cross-lingual NER system for other reasons, arrived at both conclusions independently while investigating our own system's results, and only recognized the connection to Lauscher et al. afterward. We report that connection explicitly here rather than presenting either finding as new, because we think independently-obtained corroborating evidence - on a task (NER) and language sample distinct from the original study, using a self-built system rather than reused published results - has value that a citation alone does not convey.

The single-factor hypothesis we tested most thoroughly is script: non-Latin-script languages are underrepresented in multilingual subword vocabularies and pretraining corpora relative to their number of speakers, and this is thought to degrade downstream performance for those languages specifically (Rust et al., 2021; Muller et al., 2021). We used a specific experimental design intended to disentangle script from a commonly confounded variable, morphological/typological distance from English: if a Latin-script language that is morphologically very distant from English (e.g. Swahili, Bantu) transfers about as well as a Latin-script language close to English (e.g. German), while non-Latin-script languages transfer poorly regardless of their morphological similarity to English (e.g. Korean, an isolating agglutinative language, and Arabic, a Semitic language), that is reasonably strong evidence that script, not morphology, is doing the explanatory work.

Six languages tested this way supported exactly that conclusion, and at that point in the project's development the finding looked like a positive, standalone result worth reporting on its own. We then did something a smaller project might not have budget to do: we treated the finding as a hypothesis still open to falsification, and deliberately added two more controls chosen specifically to break it if it were wrong. It broke - directly reproducing, in miniature and on a different task, the "no single factor explains it" shape of Lauscher et al.'s much larger study. This paper reports both halves honestly - the initial supporting evidence and the disconfirming follow-up - along with two further mechanistic hypotheses tested and also rejected, and positions the whole arc as independent support for an existing finding rather than a novel one.

Beyond replication, we push four analyses further than either the original literature or this project's own earlier work: (1) because our task, dataset, and much of our language sample overlap directly with Lauscher et al.'s NER evaluation, we test their own strongest identified predictor - phonological distance - rather than only generic proxies, giving the closest possible direct comparison (Section 5.5); (2) we break the four-factor analysis down by entity type (PER/ORG/LOC) rather than only aggregate F1, and find a pattern the aggregate view hides entirely (Section 5.6); (3) we directly measure whether contrastive alignment shifts token-level backbone representations - the representations NER classification actually reads - rather than relying only on an indirect probe, resolving a limitation flagged but not addressed in this project's earlier development log (Section 6.2); (4) we then connect (2) and (3) directly, testing whether the token-level alignment gap itself differs by entity type - the natural bridging hypothesis between them - and find it does not, refuting a specific mechanistic explanation we went in expecting to confirm (Section 6.3). None of these four analyses were performed by Lauscher et al., and we present them as this paper's genuine additive content on top of the replication.

---

## 2. Related Work

**The finding this paper replicates.** Lauscher et al. (2020) is the anchoring prior work for this study, and the overlap is closer than a first read suggests: among their five evaluated tasks (POS tagging, dependency parsing, NER, NLI, QA), NER is evaluated on WikiANN (Rahimi et al., 2019) - the identical dataset this paper uses - across 12 languages, five of which (Arabic, Finnish, Korean, Russian, Turkish) also appear in this paper's eight-language sample. For NER specifically, they report phonological similarity to the source language as the strongest single predictor of transfer quality (Pearson r = 0.78, Spearman r = 0.86, for mBERT), and note that pretraining corpus size correlates more weakly with NER performance than with higher-level semantic tasks (NLI, QA) - a pattern our own corpus-size result (Section 5.4) is directly consistent with. Beyond the task-specific finding, their broader claim - that transfer quality is not well predicted by any single easily-available factor considered in isolation - and their finding that inexpensive few-shot fine-tuning is "surprisingly effective across the board" at closing gaps zero-shot methods cannot, are what we independently replicate in Sections 5 and 6. Because the task, dataset, and much of the language sample overlap directly with our own, Section 5.5 tests their specific strongest NER predictor, phonological similarity, rather than only the generic script/fragmentation/corpus-size proxies - the closest direct comparison this paper can make to their result.

**Cross-lingual transfer with multilingual encoders.** mBERT and XLM-R transfer non-trivially to languages absent or underrepresented in fine-tuning data (Pires et al., 2019; Wu & Dredze, 2019). Rahimi et al. (2019) established the massively multilingual NER transfer setup on WikiANN (Pan et al., 2017) that this paper's evaluation protocol follows.

**Script and vocabulary allocation.** Multilingual subword vocabularies allocate disproportionately few tokens to non-Latin scripts, and Rust et al. (2021) show that per-language tokenizer quality - how many subwords a language's words fragment into - correlates with monolingual model performance for that language, motivating our subword-fragmentation hypothesis. Muller et al. (2021) study mBERT specifically on scripts poorly represented in pretraining and report degraded performance, motivating the script hypothesis directly. Both are candidate single factors of exactly the kind Lauscher et al. (2020) find, in aggregate, insufficient - our Section 5 results are consistent with that conclusion for this project's specific system and task.

**Typological distance features.** Littell et al. (2017) introduce the URIEL typological database and the `lang2vec` toolkit that exposes it, providing per-language feature vectors (genetic, phonological, syntactic, geographic) usable to compute typological distance between any language pair. Lauscher et al. (2020) use these features to construct their PHON predictor for NER; we use the same toolkit and feature set (`phonology_knn`) in Section 5.5 to compute an equivalent distance metric independently.

**Contrastive cross-lingual alignment.** SimCSE-style contrastive objectives (Gao et al., 2021) and cross-lingual sentence alignment more broadly are used to pull translation-equivalent representations together; we use an InfoNCE objective on real OPUS-100 parallel sentences (Zhang et al., 2020) for this purpose and separately investigate whether it improves the downstream token-level task.

**Data augmentation for cross-lingual transfer.** CoSDA-ML (Qin et al., 2020) code-switches source-language training data with target-language lexical items via bilingual dictionaries; we use this as one lever in our ablation.

**Translate-train.** Training on machine-translated target-language data is a standard alternative to zero-shot transfer (Artetxe et al., 2020, discuss translation artifacts in this setting); we test a simplified lexicon-substitution proxy for it as a secondary experiment.

This paper's contribution relative to this literature is not a new technique and not a new empirical claim at the field level - Section 5's conclusion and Section 6's conclusion are both already established, at larger scale and with more statistical power, by Lauscher et al. (2020). The contribution is independent replication on a different task, system, and language sample, obtained via a deliberately adversarial test of a single-factor hypothesis (script) that a smaller or less careful project might have stopped investigating as soon as it looked confirmed.

---

## 3. Method

### 3.1 Architecture

The system uses an XLM-RoBERTa-base backbone shared between two objectives: a 2-layer MLP projection head trained with an InfoNCE contrastive loss on sentence pairs, and a token-classification head trained on IOB2-tagged NER data. Both heads read from the same backbone; the NER head reads directly from backbone token representations rather than through the projection head unless otherwise specified (see Section 6).

### 3.2 Training Procedure

Training proceeds in two phases. **Phase 1 (contrastive alignment)**: the backbone is trained with a bidirectional InfoNCE loss on up to 50,000 real English–target parallel sentence pairs per language pair, drawn from OPUS-100 (Zhang et al., 2020). **Phase 2 (NER fine-tuning)**: the full backbone (not a frozen probe) is fine-tuned on labeled English WikiANN data with layer-wise learning-rate decay (Howard & Ruder, 2018), augmented with code-switched copies of 50% of training examples using CoSDA-ML-style lexical substitution (Qin et al., 2020) via MUSE bilingual dictionaries. No target-language labels are used at any point in either phase.

### 3.3 Evaluation Protocol

Evaluation is zero-shot span-level F1 (seqeval) on the WikiANN test split for each target language, with three entity types (PER, ORG, LOC) in IOB2 format. No target-language data is used for training or model selection.

---

## 4. Experiment 1: Building and Validating the System

### 4.1 Headline Results

The full system (contrastive alignment + full fine-tuning + code-switching) reaches English validation F1 = 0.8278 and zero-shot F1 of 0.7675 (Turkish), 0.7435 (German), 0.4563 (Arabic) - average 0.6558 across these three original target languages.

### 4.2 Ablation: Isolating the Dominant Lever

A 4-way ablation (frozen probe A; full fine-tune only B; +contrastive C; +code-switch D; full pipeline E, each trained fresh) isolates which component drives the gain:

| Config | TR | DE | AR | Avg |
|---|---|---|---|---|
| A - frozen probe | 0.543 | 0.538 | 0.221 | 0.434 |
| B - full fine-tune only | 0.758 | 0.753 | 0.495 | 0.669 |
| C - +contrastive | 0.736 | 0.745 | 0.447 | 0.642 |
| D - +code-switch | 0.740 | 0.743 | 0.466 | 0.650 |
| E - full pipeline | 0.768 | 0.744 | 0.456 | 0.656 |

Full backbone fine-tuning alone (A→B) accounts for a +0.235 average-F1 gain. Adding contrastive alignment or code-switching on top of B does not clearly help - both configurations that add exactly one auxiliary technique to B score slightly *lower* than B alone.

### 4.3 Multi-Seed Validation

Because the C/D/E-vs-B deltas above (−0.013 to −0.026) are small, we re-ran config B for three seeds to establish a noise floor before treating any of them as real: mean 0.6645, std 0.0099. Under this noise floor, the B-vs-D and B-vs-E deltas are not distinguishable from seed variance; the B-vs-C delta (−0.026, ≈2.2 standard deviations) is the most plausible candidate for a real effect but remains unconfirmed at n = 3. We report this explicitly because our first-pass, single-seed reading of the ablation overstated the case that auxiliary techniques hurt performance - a claim that did not survive proper uncertainty quantification. **Conclusion: full fine-tuning capacity, not cross-lingual alignment engineering, is the dominant, robustly established lever in this system.**

---

## 5. Experiment 2: Explaining the Transfer Gap

### 5.1 Eight Languages

Beyond the original three (Turkish, German, Arabic), we zero-shot-evaluated the existing checkpoint (no retraining) on five further languages, chosen in two rounds:

**Round 1** (chosen to test script vs. morphological distance as confounded explanations): Finnish (Latin script, Uralic - distant family, but Latin script), Swahili (Latin script, Bantu - very distant family, Latin script), Korean (Hangul, isolating/agglutinative - distant family, non-Latin script).

**Round 2** (chosen specifically to try to break the conclusion Round 1 supported): Indonesian (Latin script, Austronesian - a *second* distant-family-but-Latin-script control, from a different family than Swahili, to check the Round-1 result wasn't Swahili-specific), Russian (Cyrillic script, Slavic/Indo-European - a *close*-family-but-non-Latin-script control, the sharpest test of whether script matters independent of family).

| Language | Script | Family (distance from English) | Zero-shot F1 |
|---|---|---|---|
| Turkish | Latin | Turkic (distant) | 0.7675 |
| Finnish | Latin | Uralic (distant) | 0.7578 |
| German | Latin | Germanic (close) | 0.7435 |
| Swahili | Latin | Bantu (very distant) | 0.6748 |
| Russian | Cyrillic | Slavic/Indo-European (moderate) | 0.6102 |
| Korean | Hangul | Koreanic (distant) | 0.5086 |
| Indonesian | Latin | Austronesian (very distant) | 0.4967 |
| Arabic | Arabic | Semitic (distant) | 0.4563 |

### 5.2 Hypothesis 1: Script Identity

Round 1 alone (six languages) supports script as the dominant factor unambiguously: every Latin-script language falls in a 0.67–0.77 band spanning close (German) to very distant (Swahili) morphological relatedness to English, while both non-Latin-script languages (Korean, Arabic) fall well below that band. This is the result we set out to report as the paper's central finding before Round 2.

**Round 2 falsifies it.** Indonesian - Latin script - scores 0.4967, below Korean and barely above Arabic, landing inside the "non-Latin" cluster despite its script. Russian - non-Latin script - scores 0.6102, clearly better than Korean/Arabic but clearly below the Latin cluster's floor (Swahili, 0.6748), landing in neither cluster cleanly. A single Latin-script counterexample performing as poorly as the worst non-Latin languages is sufficient to falsify "Latin script implies safe zero-shot transfer" as a general claim; a single non-Latin counterexample performing substantially better than other non-Latin languages weakens "non-Latin script implies poor transfer" from a law to, at best, a loose tendency.

### 5.3 Hypothesis 2: Subword Fragmentation Rate

A more mechanistic version of the script hypothesis is that script correlates with, but is not identical to, subword fragmentation: XLM-R's shared vocabulary may simply split some languages' words into more pieces than others, independent of script per se, making downstream classification harder. We measured mean subwords-per-word for each of the eight languages using the XLM-R tokenizer over 2,000 WikiANN test sentences per language.

| Language | Subwords/word | F1 |
|---|---|---|
| Swahili | 1.553 | 0.6748 |
| Indonesian | 1.594 | 0.4967 |
| German | 1.609 | 0.7435 |
| Arabic | 1.715 | 0.4563 |
| Turkish | 1.751 | 0.7675 |
| Russian | 1.914 | 0.6102 |
| Finnish | 1.955 | 0.7578 |
| Korean | 2.212 | 0.5086 |

Pearson r = −0.150 - weak, and wrong-signed if fragmentation were the driver (more fragmentation should predict *lower* F1; a correlation this close to zero indicates fragmentation explains almost none of the variance). The concrete counterexamples make this unambiguous without further significance testing on n = 8: Finnish fragments more than Russian or Arabic (1.955 subwords/word) yet has one of the highest F1 scores (0.7578); Indonesian fragments about as little as Swahili (1.594 vs. 1.553) yet has one of the lowest F1 scores (0.4967) of any language tested.

### 5.4 Hypothesis 3: Pretraining Corpus Size

XLM-R was pretrained on unequal amounts of CC-100 text per language (Conneau et al., 2020). Lower-resource languages in pretraining might transfer worse regardless of script or fragmentation.

| Language | CC-100 size (GiB) | F1 |
|---|---|---|
| Swahili | 1.6 | 0.6748 |
| Turkish | 20.9 | 0.7675 |
| Arabic | 28.0 | 0.4563 |
| Korean | 54.2 | 0.5086 |
| Finnish | 54.3 | 0.7578 |
| German | 66.6 | 0.7435 |
| Indonesian | 148.3 | 0.4967 |
| Russian | 278.0 | 0.6102 |

Pearson r = −0.214 (raw GiB) / −0.235 (log GiB) - again weak, and wrong-signed: more pretraining data weakly correlates with *worse*, not better, zero-shot F1 in this sample. Swahili has the smallest CC-100 allocation of any language tested (1.6 GiB, two orders of magnitude below Russian's 278 GiB) yet transfers well; Indonesian has the third-largest allocation (148.3 GiB) yet is one of the two worst performers; Russian has by far the largest allocation (278 GiB) yet is only mid-pack, clearly behind Turkish's 0.7675 achieved with 20.9 GiB.

### 5.5 Hypothesis 4: Phonological Similarity (Lauscher et al.'s Strongest NER Predictor)

Because this study's task (NER), dataset (WikiANN), and five of eight target languages overlap directly with Lauscher et al. (2020)'s NER evaluation, we can test their own strongest identified predictor rather than only generic proxies. Following their approach, we use `lang2vec` (Littell et al., 2017) to obtain each language's URIEL phonological feature vector (`phonology_knn`, 28 dimensions) and compute phonological distance from English as one minus cosine similarity.

| Language | Phonological distance from English | F1 |
|---|---|---|
| Swahili | 0.0909 | 0.6748 |
| Indonesian | 0.0909 | 0.4967 |
| Finnish | 0.1296 | 0.7578 |
| Russian | 0.1419 | 0.6102 |
| Turkish | 0.1818 | 0.7675 |
| German | 0.1942 | 0.7435 |
| Korean | 0.2538 | 0.5086 |
| Arabic | 0.2994 | 0.4563 |

Pearson r = −0.387 - correctly signed (greater phonological distance predicts lower F1, as Lauscher et al.'s result would suggest) and the strongest correlation of the four hypotheses tested, roughly 1.6-2.6× the magnitude of the fragmentation and corpus-size correlations. It is still far short of Lauscher et al.'s reported r ≈ 0.78 (equivalently ≈ −0.78 for distance) in their much larger study, and it has a sharp counterexample of its own: **Indonesian and Swahili are phonologically equidistant from English by this metric (0.0909, identical to four decimal places) yet differ by 0.178 F1** - the single largest same-distance F1 gap in the sample. Phonological distance is the best-performing of the four factors tested here, but it does not explain this specific, large discrepancy any better than script, fragmentation, or corpus size did.

We read this as a partial, directionally-consistent but substantially weaker replication of Lauscher et al.'s specific NER finding. Plausible reasons for the gap in effect size include our much smaller sample (n = 8 vs. their larger language set, which affords more statistical power and a wider range of phonological distances to fit against), the single-checkpoint zero-shot evaluation here (no multi-seed averaging for the five extended languages, unlike the original three), and genuine differences between our system (contrastive-aligned, code-switch-augmented, fully fine-tuned XLM-R) and mBERT, the model Lauscher et al. report this specific correlation for.

### 5.6 Entity-Type-Level Refinement: The Aggregate Null Result Is Driven by ORG and LOC

All four hypotheses above were tested against aggregate (micro-averaged) F1. WikiANN NER has three entity types - PER, ORG, LOC - and a prior error analysis on this checkpoint (documented in the accompanying development log, not reproduced in full here) already shows Arabic's weak aggregate score is driven by a specific, non-uniform failure rather than uniform difficulty across entity types: true O/PER/LOC tokens are mispredicted as ORG at 2-4x the rate seen in Turkish/German (O→ORG 16.3% vs. 3-4%; PER→ORG 26.0% vs. ~5%; LOC→ORG 37.8% vs. ~11%), while ORG's own recall (0.902) is on par with those languages. We re-evaluated the same checkpoint on all eight languages capturing per-entity-type F1, then recomputed each of the four hypotheses' Pearson correlations separately per entity type.

| Entity type | Script (Latin=1) | Fragmentation | Corpus size | Phon. distance |
|---|---|---|---|---|
| PER | **r = 0.824** | r = −0.517 | r = −0.515 | r = −0.403 |
| ORG | r = 0.356 | r = 0.036 | r = 0.258 | r = −0.275 |
| LOC | r = 0.476 | r = 0.139 | r = −0.202 | r = −0.224 |
| Aggregate (micro) | r = 0.664 | r = −0.150 | r = −0.214 | r = −0.387 |

**PER transfer correlates strongly with script identity (r = 0.824) - clearly stronger than any of the four aggregate-level correlations in Section 5.2-5.5.** ORG and LOC, by contrast, correlate weakly with all four factors (every |r| < 0.5, several near zero). This decomposition shows the aggregate "no single factor explains it" result is not uniform across entity types: person-name recognition is reasonably well predicted by script alone, while organization and location recognition are not well predicted by anything measured in this study. This refines, rather than reverses, Sections 5.2-5.5's conclusion - the transfer gap remains real and largely unexplained in aggregate, but the unexplained variance concentrates specifically in ORG and LOC, which is itself a more precise and actionable finding than "the gap is unexplained" taken as a monolithic claim. It also connects to the Arabic-specific over-prediction finding noted above: if ORG recognition is poorly predicted by any of these factors across all eight languages, not just Arabic, then Arabic's ORG over-prediction bias may be an instance of a broader pattern - ORG being the entity type most resistant to whatever mechanism drives cross-lingual transfer in this system - rather than an Arabic-specific idiosyncrasy.

### 5.7 Synthesis: Four Hypotheses, One Partial Success, and an Entity-Type Refinement

| Hypothesis | Test | Outcome |
|---|---|---|
| Script identity | 8-language comparison, 2 disconfirming controls | Falsified (Indonesian); complicated (Russian) |
| Subword fragmentation | Pearson correlation, tokenizer-measured | r = −0.150, wrong-signed, sharp counterexamples |
| Pretraining corpus size | Pearson correlation, published CC-100 sizes | r = −0.214/−0.235, wrong-signed, sharp counterexamples |
| Phonological distance | Pearson correlation, URIEL/lang2vec features | r = −0.387, correctly signed, best of the four, still far below Lauscher et al.'s reported effect, own sharp counterexample |

None of the four fully explains the observed aggregate pattern; phonological distance comes closest but does not close the gap to Lauscher et al.'s reported effect size, and Section 5.6 shows the unexplained aggregate variance is concentrated specifically in ORG and LOC rather than distributed uniformly. We emphasize that each hypothesis was tested to completion with a clear, falsifiable, quantitative prediction - not abandoned partway or judged only by failing to reach significance on a small sample. The transfer gap is real (a 1.68× range in F1 between the best and worst of the eight languages tested) and is not fully attributable, individually, to script, fragmentation, pretraining volume, or phonological distance - though the last comes closer than the first three, consistent with (if weaker than) Lauscher et al.'s NER-specific finding. Plausible remaining candidates this study cannot adjudicate include WikiANN's per-language silver-standard annotation quality (the dataset is built automatically from Wikipedia hyperlink structure rather than human annotation, and is known in the literature to vary in reliability across languages) and entity-type-specific properties of XLM-R's representation geometry not captured by any of the four aggregate-level measures tested here.

---

## 6. The Gap Is Addressable Despite Being Unexplained

### 6.1 Zero-Shot Algorithmic Interventions Fail

Beyond the hypotheses above, we tested whether specific interventions could close the Arabic gap (the largest gap among the original three languages) without target-language labels: contrastive alignment (Section 4.2, config C vs. B: −0.026 avg), code-switching (config D vs. B: −0.019 avg), and routing NER classification through the contrastively-trained projection head instead of raw backbone features (a direct test of whether the demonstrated sentence-level alignment - a true/shuffled-pair cosine similarity gap of 0.558–0.636, confirmed via embedding visualization, up from ~0.005–0.007 before training - can be made to help token-level classification by making it reachable; config F vs. C: −0.0014, an order of magnitude below the multi-seed noise floor). None moved Arabic F1 by more than ±0.03, and most of that range is not distinguishable from noise.

### 6.2 Why Doesn't Alignment Help? A Direct Token-Level Measurement

Section 6.1's routing probe tests one specific mechanism for the dissociation (whether NER training can *reach* the projection head) and finds it doesn't matter - but that probe never measures whether the raw backbone token representations NER actually classifies (via `get_token_embeddings()`, bypassing the projection head entirely, which is how the main pipeline runs) shift at all from contrastive training. We measure this directly: mean-pooled raw backbone token embeddings for the same OPUS-100 sentence pairs used in Section 6.1's alignment-gap measurement, comparing a fresh pretrained backbone against the Phase-1 checkpoint, using the identical true-pair-vs-shuffled-pair gap methodology applied at the sentence level in Section 6.1 but computed here on token-level (not CLS-level, not projection-head-routed) representations.

| Language | Token-level gap (before) | Token-level gap (after) | Sentence-level (CLS+projection) gap (after) |
|---|---|---|---|
| Turkish | 0.008 | 0.311 | 0.629 |
| German | 0.008 | 0.362 | 0.636 |
| Arabic | 0.004 | 0.363 | 0.558 |

Two findings emerge. First, **token representations do shift substantially from contrastive training** - from near-zero (0.004-0.008, chance level, matching the pretrained-baseline sentence-level gap) to 0.31-0.36 - which directly refutes the strongest version of "contrastive alignment never reaches token representations at all." Second, that shift is **roughly half the magnitude of the sentence-level shift** (0.31-0.36 vs. 0.56-0.64), a real, measurable attenuation between what the contrastive objective directly optimizes (pooled CLS vectors through the projection head) and what happens to the token-level states several layers of pooling and a separate head removed from that objective. Third, and unexpectedly, **the per-language ranking does not transfer**: Arabic has the smallest sentence-level gap of the three languages but the largest token-level gap, essentially tied with German - whatever makes Arabic's sentence-level alignment weaker does not straightforwardly weaken its token-level alignment too.

The practical conclusion is more precise than Section 6.1 alone could support: it is not that contrastive alignment fails to reach token representations - it demonstrably does, at roughly half strength - but that even this real, substantial, non-zero token-level shift is insufficient to move NER F1 outside the multi-seed noise floor (Section 4.3). Whatever the contrastive objective changes about token representations, it is either not the specific structure NER classification depends on, or not changed by enough to matter at this system's scale. This is a more precise, directly measured account of the dissociation than either Section 6.1's probe or Section 5.7's synthesis could offer on their own, though it stops short of a full explanation - a complete account would need something closer to CKA representation-similarity analysis across all backbone layers, which we did not perform (Section 8).

### 6.3 Does the Alignment Gap Differ by Entity Type? A Refuted Hypothesis

Section 5.6 shows PER transfer correlates strongly with script (r = 0.824) while ORG and LOC do not (all |r| < 0.5). Section 6.2 measures the token-level alignment gap only as a single average across all tokens. The natural bridging hypothesis: if PER-tagged token representations are pulled together by contrastive training more strongly than ORG/LOC-tagged representations, that would help explain why PER transfers more predictably. We test this directly, using the fully-trained config E model purely as a labeling tool (predicting a PER/ORG/LOC/O tag for every subword position on both sides of 1,000 OPUS-100 sentence pairs per language) and computing the same true-pair-vs-shuffled-pair alignment gap as Section 6.2, separately for each predicted entity type.

| Language | PER gap (n) | ORG gap (n) | LOC gap (n) | O gap (n) |
|---|---|---|---|---|
| Turkish | 0.303 (110) | 0.340 (570) | 0.313 (26) | 0.320 (623) |
| German | 0.348 (74) | 0.374 (595) | 0.320 (38) | 0.377 (593) |
| Arabic | 0.273 (51) | 0.356 (601) | 0.256 (31) | 0.334 (530) |

**The hypothesis is refuted.** PER does not show a larger alignment gap than ORG in any of the three languages - if anything, ORG shows the largest or near-largest gap in all three (0.340, 0.374, 0.356, versus PER's 0.303, 0.348, 0.273). Whatever makes PER transfer more predictably than ORG/LOC (Section 5.6), it is not that PER-relevant token representations are more strongly aligned by contrastive training - by this measure, ORG representations are pulled together at least as strongly, sometimes more so, yet ORG is precisely the entity type Section 5.6 shows is *not* well predicted by any tested factor and prior error analysis (noted in Section 5.6) shows is subject to systematic over-prediction on gold-labeled WikiANN data. This deepens rather than resolves the dissociation documented throughout this section: representation-level alignment and downstream classification quality are decoupled not just in aggregate, but per entity type too - a real alignment signal on ORG-tagged tokens coexists with ORG being the hardest entity type to predict transfer quality for and the one most prone to being predicted where it shouldn't be.

One incidental observation from this analysis: the tagger predicts ORG far more often than PER or LOC on this out-of-domain (non-WikiANN) text, consistently across all three languages (570-601 ORG-containing sentences vs. 26-110 for PER/LOC, out of 1,000 sampled). Because OPUS-100 has no gold entity labels, this cannot be validated as an error rate the way the WikiANN-based over-prediction analysis above could - but it is at least consistent with, and not contradicted by, that broader over-prediction pattern, and it is notable that this volume skew appears uniform across languages rather than concentrated in Arabic, unlike the *error* rate the gold-labeled analysis measures. We report it as a suggestive side observation, not a confirmed finding, given the absence of gold labels and the small PER/LOC sample sizes in some cells (as few as n = 26 for Turkish LOC).

### 6.4 Few-Shot Supervision Succeeds

In contrast, fine-tuning the same checkpoint on a small number of labeled Arabic examples closes most of the gap immediately:

| k (labeled examples) | Turkish | German | Arabic |
|---|---|---|---|
| 0 (zero-shot) | 0.7675 | 0.7435 | 0.4563 |
| 10 | 0.7600 | 0.7514 | 0.6828 |
| 50 | 0.8192 | 0.7616 | 0.7264 |
| 100 | 0.8282 | 0.7932 | 0.7469 |
| 500 | 0.8606 | 0.8106 | 0.7868 |

Ten labeled Arabic examples raise F1 from 0.456 to 0.683 (+0.227), recovering roughly three-quarters of the gap to Turkish/German in a single small fine-tuning pass - an order of magnitude larger than the effect of every zero-shot intervention tested combined. By 500 examples (2.5% of the English training set), Arabic (0.787) is within 0.07–0.08 of Turkish and German. **The practical implication is independent of Section 5's null result**: not knowing *why* a language transfers poorly does not prevent fixing it economically - a small labeled set closes the gap regardless of mechanism. This is the second of the two Lauscher et al. (2020) findings this paper independently replicates: they report few-shot fine-tuning as "surprisingly effective across the board" across their much larger study, and our single-task, single-system result is directly consistent with that.

---

## 7. Discussion

The central methodological point of this paper is what happened between Sections 5.2 and 5.5: a hypothesis that looked confirmed on six languages, using a deliberately controlled comparison (Swahili isolating script from morphology), did not survive two additional controls chosen specifically to stress-test it. This is exactly what adding disconfirming evidence is supposed to do, and exactly why an initially clean six-language result should not have been reported as final without first trying to break it. Two further mechanistic hypotheses, chosen because they are quantifiable and literature-motivated rather than post-hoc, also failed on their own terms with sharp counterexamples rather than merely falling short of significance.

**We frame this as independent corroboration of Lauscher et al. (2020) rather than as an original discovery, and we think that framing matters.** It would have been easy to write Section 5 as if "no single factor explains the transfer gap" were a new finding of this project - the six-language version of the result, before Round 2, genuinely looked that way from inside the project. It is not new: Lauscher et al. already established the general shape of this result across many languages and several tasks, with far more statistical power than eight languages and three pairwise Pearson correlations can offer. What this project adds is a second, independently-obtained data point for the same conclusion, arrived at on a different task (NER rather than the tasks in the original study), a different system (built from scratch here, not a reused published checkpoint), and a language sample selected without reference to the original study's results. Two independent systems reaching the same conclusion through different paths is modest but real evidence that the conclusion is not an artifact of one particular study's setup - which is a different and, we would argue, still useful kind of contribution from proposing something new.

A reader who assumed "it's obviously the script" or "it's obviously the pretraining data" now has two independent studies' worth of evidence against both, not one. That evidence was cheap to obtain in this project's case: all extended-language evaluations are pure inference on an existing checkpoint, and the fragmentation and corpus-size analyses require no training at all - which suggests replication of this kind is worth doing more often precisely because it does not require repeating the expensive part of the original study.

**The four analyses in Sections 5.5, 5.6, 6.2, and 6.3 go beyond replication, and it is worth being precise about what makes them different from Sections 5.2-5.4.** Testing script, fragmentation, and corpus size answers "does this specific proxy explain the gap" and gets three no's. Testing phonological distance (5.5) is still in that family, but it is not an arbitrary fourth proxy - it is the specific variable an existing, much larger study already identified as the best available answer for this exact task and dataset, so a partial, directionally-consistent replication (r = −0.387, correctly signed, but well short of their effect size) is more informative than either a clean confirmation or a clean rejection would have been: it suggests the effect is real but this study's sample and setup capture only part of it. The entity-type breakdown (5.6) is a different kind of move entirely - it does not test a new candidate cause, it changes the unit of analysis, and in doing so shows the aggregate null result was concealing a real, specific structure (PER predictable, ORG/LOC not) that no single-factor aggregate test, ours or Lauscher et al.'s, would surface. The token-level measurement (6.2) is different again - it does not test whether some external property predicts the gap, it opens up the model itself and asks whether a specific internal mechanism (contrastive alignment) does what it is supposed to do at the representation level NER actually uses, independent of any question about which languages transfer well. The entity-type-conditioned alignment measurement (6.3) combines the previous two moves and, in doing so, produces this paper's clearest negative result: the natural hypothesis connecting "PER transfers more predictably" (5.6) to "alignment reaches tokens at reduced strength" (6.2) - that PER-tagged representations are more strongly aligned - is directly testable, and directly false. We went in expecting confirmation and got a refutation instead, which is itself informative: it rules out the most obvious mechanistic bridge between our two most novel findings and leaves the actual connection, if one exists, unidentified. These four moves - testing the literature's own best predictor directly, changing the unit of analysis, measuring inside the model rather than only its outputs, and combining the two to test a specific bridging hypothesis - are the parts of this paper that are not already in Lauscher et al. or elsewhere in the cited literature, as far as we are aware.

---

## 8. Limitations

1. **Eight languages is still a small sample** for correlational claims (Pearson r on n = 8 is not statistically powerful); the wrong-signed correlations and sharp counterexamples are more persuasive than the r values themselves, but a larger language sample would strengthen or further complicate the picture. This limitation applies with extra force to Section 5.6's entity-type breakdown, where each correlation is computed on the same eight data points split three ways by entity type rather than gaining new observations.
2. **WikiANN is a silver-standard dataset** derived automatically from Wikipedia hyperlink structure, not human-annotated, and its per-language annotation quality is not independently verified in this study - it remains a live candidate explanation for the residual pattern that this paper cannot rule in or out.
3. **Single source language.** All fine-tuning uses English-only data; the pattern might look different with a different or multiple source languages.
4. **n = 3 seeds** for the multi-seed ablation is sufficient to establish a noise floor but not sufficient to confirm or rule out effects in the 0.02–0.03 range with high confidence. The extended eight-language evaluation (Sections 5.1-5.6) and the token-level analysis (Section 6.2) use single-seed checkpoints throughout, unlike the original three-language multi-seed result.
5. **Section 6.2's token-level measurement is a mean-pooled cosine-similarity gap, not full CKA representation-similarity analysis.** It establishes that *some* token-level shift occurs and roughly how large it is relative to the sentence-level shift, but mean-pooling discards per-token and per-layer structure that a proper CKA comparison across backbone layers would preserve; it cannot say *which* tokens, dimensions, or layers shifted, only that the aggregate did.
6. **Section 5.5's phonological distance metric uses a single URIEL feature set** (`phonology_knn`) and a single distance formula (cosine); Lauscher et al. (2020) may use a different specific formulation, and other URIEL feature families (genetic, syntactic, geographic distance) were not tested here and might correlate differently.
7. **Section 5.6's entity-type correlations are exploratory**, computed after observing the aggregate null result rather than pre-registered; the strong PER-script correlation (r = 0.824) should be treated as a hypothesis for future confirmation on a larger sample, not a confirmed effect.
8. **Section 6.3's entity-type labels are model-predicted, not gold.** No gold parallel entity-annotated corpus exists across these language pairs, so entity types on both sides of each OPUS-100 pair are predicted by the same zero-shot NER system being evaluated - a defensible exploratory technique given the alternative (no data at all), but not independent verification, and least reliable for Arabic, where the tagger's own zero-shot F1 is lowest. Sample sizes also vary substantially by entity type and language (as few as n = 26 for Turkish LOC, versus n = 570-601 for ORG), so the ORG estimates are considerably more stable than the LOC estimates.

---

## 9. Conclusion

We built and validated a zero-shot cross-lingual NER system, established that full backbone fine-tuning - not auxiliary contrastive alignment or code-switching - is its dominant performance lever, and then set out to explain the substantial, real cross-language variance in its zero-shot transfer (0.456–0.768 F1 across three original languages, extending to 0.497–0.768 across eight). An initial six-language sample supported script identity as the explanation; two additional controls, deliberately chosen to try to break that conclusion, did break it. Two further hypotheses - subword fragmentation and pretraining corpus size - were tested to the same standard and also failed. Separately, we show the gap - whatever its cause - collapses rapidly under minimal target-language supervision.

Both outcomes are independent replications of Lauscher et al. (2020), obtained on a different task, a self-built system, and a language sample chosen without reference to their results. We report them as replication, not discovery. Four further analyses go beyond that replication. Testing Lauscher et al.'s own strongest NER predictor, phonological distance, on our overlapping task, dataset, and languages shows a real but substantially weaker effect (r = −0.387 vs. their reported ≈−0.78) than their larger study found. Decomposing the four-factor analysis by entity type shows the aggregate null result is concentrated in ORG and LOC while PER transfer is well predicted by script alone (r = 0.824). Directly measuring token-level representation shift shows contrastive alignment does change the representations NER actually reads, substantially, just at roughly half the strength of its effect on sentence-level representations - and that even this real shift does not move NER F1. And testing the natural hypothesis connecting the previous two findings - that PER's stronger correlation reflects PER-tagged representations being more strongly aligned - directly refutes it: ORG shows the largest token-level alignment gap in every language tested, not PER. None of these four analyses were performed by the study we are replicating, and we present them as this paper's own additive content rather than as further replication.

For anyone deciding how to deploy a system like this on a low-resource target language today, the actionable conclusion is the one both studies converge on - do not invest further in unsupervised alignment engineering to close a script-correlated gap whose root cause remains unidentified; invest in a small labeled set instead. For anyone investigating *why* such gaps exist, the entity-type and token-level results suggest two more specific directions than "try another aggregate-level proxy" - look separately at ORG/LOC rather than NER as a whole, and look inside the model's representations rather than only at its outputs - while the refuted bridging hypothesis (Section 6.3) is a reminder that even a well-motivated mechanistic story connecting two real findings needs to be tested directly rather than assumed, since here it turned out to be wrong.

---

## References

- Artetxe, M., Labaka, G., & Agirre, E. (2020). Translation Artifacts in Cross-lingual Transfer Learning. *EMNLP*.
- Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). *ACL*.
- Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. *EMNLP*.
- Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. *ACL*.
- Lauscher, A., Ravishankar, V., Vulić, I., & Glavaš, G. (2020). From Zero to Hero: On the Limitations of Zero-Shot Language Transfer with Multilingual Transformers. *EMNLP*, pp. 4483-4499.
- Littell, P., Mortensen, D. R., Lin, K., Kairis, K., Turner, C., & Levin, L. (2017). URIEL and lang2vec: Representing Languages as Typological, Geographical, and Phylogenetic Vectors. *EACL*.
- Muller, B., Sagot, B., & Seddah, D. (2021). When Being Unseen from mBERT is just the Beginning: Handling New Languages With Multilingual Language Models. *NAACL*.
- Pan, X., et al. (2017). Cross-lingual Name Tagging and Linking for 282 Languages (WikiANN). *ACL*.
- Pires, T., Schlinger, E., & Garrette, D. (2019). How Multilingual is Multilingual BERT? *ACL*.
- Qin, L., et al. (2020). CoSDA-ML: Multi-Lingual Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP. *IJCAI*.
- Rahimi, A., Li, Y., & Cohn, T. (2019). Massively Multilingual Transfer for NER. *ACL*.
- Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models. *ACL*.
- Wu, S., & Dredze, M. (2019). Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT. *EMNLP*.
- Zhang, B., et al. (2020). Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation (OPUS-100). *ACL*.

---

*Full development history, additional experiments (error analysis, embedding alignment visualization, translate-train baseline), engineering artifacts (inference demo, unit tests, experiment tracking dashboard), and reproducibility details are documented in `DEVELOPMENT.md` in this repository.*
