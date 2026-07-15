"""
Zero-Shot Multilingual NER Data Processing Pipeline.

Prepares data for three pipeline phases:
  - Real parallel-corpus contrastive alignment (opus-100)
  - Code-switching augmented NER fine-tuning (CoSDA-ML style, MUSE bilingual lexicons)
  - Multi-language zero-shot NER evaluation (WikiANN)
"""

import os
import random
import urllib.request

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from torch.utils.data import Dataset


LABEL2ID = {
    "O": 0, "B-PER": 1, "I-PER": 2,
    "B-ORG": 3, "I-ORG": 4,
    "B-LOC": 5, "I-LOC": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MUSE_DICT_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/{src}-{tgt}.txt"


class MultiLingualNERPipeline:
    def __init__(self, model_name="xlm-roberta-base", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.label2id = LABEL2ID
        self.id2label = ID2LABEL

    # ------------------------------------------------------------------
    # NER (WikiANN) loading + subword label alignment
    # ------------------------------------------------------------------
    def tokenize_and_align(self, examples):
        tokenized = self.tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        all_labels = []
        for i, label_ids in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned.append(-100)
                elif word_idx != prev_word_idx:
                    aligned.append(label_ids[word_idx])
                else:
                    aligned.append(-100)
                prev_word_idx = word_idx
            all_labels.append(aligned)
        tokenized["labels"] = all_labels
        return tokenized

    def load_ner_dataset(self, lang):
        """Load WikiANN for a single language, tokenized and label-aligned."""
        dataset = load_dataset("unimelb-nlp/wikiann", lang)
        tokenized = dataset.map(self.tokenize_and_align, batched=True, remove_columns=dataset["train"].column_names)
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        print(f"WikiANN [{lang}] loaded: {len(tokenized['train'])} train / {len(tokenized['test'])} test")
        return tokenized

    def load_ner_dataset_augmented(self, lang, augmenter, augment_ratio=0.5):
        """
        Load WikiANN train split for `lang`, mix in code-switched copies of a
        fraction of examples (CoSDA-ML style), then tokenize + align all of it.
        Validation/test splits are returned unaugmented.
        """
        dataset = load_dataset("unimelb-nlp/wikiann", lang)
        raw_train = dataset["train"]

        n_augment = int(len(raw_train) * augment_ratio)
        idx = random.sample(range(len(raw_train)), n_augment)
        augmented_examples = [augmenter.augment(raw_train[i]) for i in idx]

        from datasets import Dataset as HFDataset
        augmented_ds = HFDataset.from_list(augmented_examples)
        # from_list() infers plain int64 for ner_tags; cast back to the original
        # ClassLabel feature schema so concatenate_datasets doesn't reject it.
        augmented_ds = augmented_ds.cast(raw_train.features)
        combined_train = concatenate_datasets([raw_train, augmented_ds])

        tokenized_train = combined_train.map(
            self.tokenize_and_align, batched=True, remove_columns=combined_train.column_names
        )
        tokenized_val = dataset["validation"].map(
            self.tokenize_and_align, batched=True, remove_columns=dataset["validation"].column_names
        )
        tokenized_test = dataset["test"].map(
            self.tokenize_and_align, batched=True, remove_columns=dataset["test"].column_names
        )

        for split in (tokenized_train, tokenized_val, tokenized_test):
            split.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        print(
            f"WikiANN [{lang}] + code-switch augmentation loaded: "
            f"{len(tokenized_train)} train ({n_augment} augmented) / {len(tokenized_val)} val / {len(tokenized_test)} test"
        )
        return {"train": tokenized_train, "validation": tokenized_val, "test": tokenized_test}

    def load_ner_dataset_translated(self, source_lang, augmenter):
        """
        Builds a translate-train set: EVERY English training example is passed
        through `augmenter` (expected switch_prob=1.0, single-language lexicon)
        so the entire split becomes word-level-substituted pseudo-target-language
        text, labels preserved exactly (substitution is per-token and in place,
        so alignment is trivial - no MT word-reordering/alignment problem to
        solve). This is a simplified proxy for true MT-based translate-train:
        it captures lexical substitution but not word order, morphological
        adaptation, or fluency, since the bilingual lexicon is a static
        word-for-word dictionary, not a trained translation model. Unlike
        load_ner_dataset_augmented(), there is no mixing with the original
        English data - the returned train split is fully translated.
        """
        dataset = load_dataset("unimelb-nlp/wikiann", source_lang)
        raw_train = dataset["train"]
        translated_examples = [augmenter.augment(raw_train[i]) for i in range(len(raw_train))]

        from datasets import Dataset as HFDataset
        translated_ds = HFDataset.from_list(translated_examples)
        translated_ds = translated_ds.cast(raw_train.features)

        tokenized_train = translated_ds.map(
            self.tokenize_and_align, batched=True, remove_columns=translated_ds.column_names
        )
        tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        print(f"Translate-train (lexicon substitution) built: {len(tokenized_train)} pseudo-translated examples")
        return tokenized_train

    # ------------------------------------------------------------------
    # Real parallel-corpus contrastive alignment data (opus-100)
    # ------------------------------------------------------------------
    def load_parallel_corpus(self, source_lang="en", target_lang="tr"):
        """Loads real translation pairs from opus-100 (config names are alphabetical)."""
        config = "-".join(sorted([source_lang, target_lang]))
        dataset = load_dataset("Helsinki-NLP/opus-100", config)
        print(f"Parallel corpus opus-100 [{config}] loaded: {len(dataset['train'])} train pairs")
        return dataset


class ParallelSentenceDataset(Dataset):
    """
    Yields paired (source, target) tokenized batches from REAL translation
    pairs (opus-100), giving the contrastive loss genuine alignment signal
    (unlike pairing unrelated sentences by index).
    """
    def __init__(self, hf_split, source_lang, target_lang, tokenizer, max_length=128, max_examples=50000):
        n = min(len(hf_split), max_examples)
        self.data = hf_split.select(range(n))
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tokenize(self, text):
        return self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]["translation"]
        src_enc = self._tokenize(pair[self.source_lang])
        tgt_enc = self._tokenize(pair[self.target_lang])
        return {
            "source": {k: v.squeeze(0) for k, v in src_enc.items()},
            "target": {k: v.squeeze(0) for k, v in tgt_enc.items()},
        }


class BilingualLexicon:
    """Downloads and caches a MUSE en-X bilingual dictionary for word-level translation."""
    def __init__(self, target_lang, source_lang="en", cache_dir="data"):
        self.target_lang = target_lang
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"muse_{source_lang}-{target_lang}.txt")

        if not os.path.exists(cache_path):
            url = MUSE_DICT_URL.format(src=source_lang, tgt=target_lang)
            print(f"Downloading bilingual lexicon: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(cache_path, "wb") as f:
                f.write(data)

        self.word_map = {}
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                # MUSE dictionaries are inconsistently tab- or space-separated
                # depending on language pair, so split on any whitespace.
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                src_word, tgt_word = parts
                self.word_map.setdefault(src_word.lower(), []).append(tgt_word)

        print(f"Bilingual lexicon en-{target_lang} loaded: {len(self.word_map)} source words")

    def translate(self, word):
        candidates = self.word_map.get(word.lower())
        if not candidates:
            return None
        return random.choice(candidates)


class CodeSwitchAugmenter:
    """
    CoSDA-ML style augmentation: randomly substitutes English words in a
    WikiANN example with target-language translations, leaving NER labels
    untouched. Trains the NER head to handle code-mixed / target-language
    tokens in context without needing any labeled target-language data.
    """
    def __init__(self, lexicons, switch_prob=0.3):
        """lexicons: dict[lang_code] -> BilingualLexicon"""
        self.lexicons = lexicons
        self.switch_prob = switch_prob
        self.langs = list(lexicons.keys())

    def augment(self, example):
        tokens = list(example["tokens"])
        lang = random.choice(self.langs)
        lexicon = self.lexicons[lang]

        new_tokens = []
        for tok in tokens:
            if random.random() < self.switch_prob:
                translation = lexicon.translate(tok)
                if translation is not None:
                    if tok[:1].isupper():
                        translation = translation.capitalize()
                    new_tokens.append(translation)
                    continue
            new_tokens.append(tok)

        return {
            "tokens": new_tokens,
            "ner_tags": example["ner_tags"],
            "langs": example.get("langs", ["en"] * len(new_tokens)),
            "spans": example.get("spans", []),
        }
