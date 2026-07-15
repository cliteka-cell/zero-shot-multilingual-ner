"""
train_zero_shot.py
Orchestrates the three-phase zero-shot NER pipeline:

  Phase 1 - Contrastive alignment on REAL parallel sentences (opus-100):
            pulls XLM-R embeddings of true English/target translation pairs
            together via InfoNCE loss.
  Phase 2 - NER fine-tuning: FULL backbone fine-tuning (layer-wise LR decay)
            on English WikiANN, augmented with CoSDA-ML style code-switching
            so the model sees target-language tokens in context during
            training without using any target-language labels.
  Phase 3 - Zero-shot evaluation across MULTIPLE target languages, no
            target-language labels used at any point.
"""

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contrastive_model import ZeroShotAligner, InfoNCELoss, NERClassifier
from src.data_processing import (
    MultiLingualNERPipeline,
    ParallelSentenceDataset,
    BilingualLexicon,
    CodeSwitchAugmenter,
)


def set_seed(seed):
    """Seeds python/numpy/torch RNGs (data shuffling, augmentation sampling,
    weight init, dropout) so a run is reproducible for a given seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ZeroShotNERTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.processor = MultiLingualNERPipeline(
            model_name=config["base_model"],
            max_length=config.get("max_seq_length", 128),
        )
        self.model = ZeroShotAligner(config["base_model"]).to(self.device)
        self.criterion = InfoNCELoss(temperature=config.get("temp", 0.07))
        self.num_labels = 7
        self.id2label = self.processor.id2label
        self.ner_head = None
        # When True, NER training/eval route token embeddings through the (contrastively
        # trained) projection head instead of bypassing it - see get_ner_embeddings().
        self.route_through_projection = False

    def get_ner_embeddings(self, input_ids, attention_mask):
        if self.route_through_projection:
            return self.model.get_projected_token_embeddings(input_ids, attention_mask)
        return self.model.get_token_embeddings(input_ids, attention_mask)

    # ------------------------------------------------------------------
    # Phase 1: Contrastive alignment on real parallel sentences
    # ------------------------------------------------------------------
    def contrastive_train(self, source_lang="en", target_langs=("tr",), checkpoint_name="alignment_checkpoint.pt"):
        print(f"\n=== Phase 1: Contrastive Alignment on real parallel data ({source_lang} -> {target_langs}) ===")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.get("lr", 2e-5))
        epochs = self.config.get("contrastive_epochs", 3)

        for target_lang in target_langs:
            parallel = self.processor.load_parallel_corpus(source_lang, target_lang)
            ds = ParallelSentenceDataset(
                parallel["train"], source_lang, target_lang,
                self.processor.tokenizer,
                max_length=self.config.get("max_seq_length", 128),
                max_examples=self.config.get("max_parallel_examples", 50000),
            )
            loader = DataLoader(ds, batch_size=self.config.get("batch_size", 16), shuffle=True, num_workers=0)
            log_every = max(1, len(loader) // 5)

            for epoch in range(1, epochs + 1):
                self.model.train()
                total_loss = 0.0
                for step, batch in enumerate(loader):
                    src_tokens = {k: v.to(self.device) for k, v in batch["source"].items()}
                    tgt_tokens = {k: v.to(self.device) for k, v in batch["target"].items()}

                    emb_src = self.model(src_tokens)
                    emb_tgt = self.model(tgt_tokens)
                    loss = self.criterion(emb_src, emb_tgt)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    if (step + 1) % log_every == 0:
                        print(f"  [{target_lang}] Epoch {epoch} | Step {step+1}/{len(loader)} | Loss: {loss.item():.4f}")

                print(f"[{target_lang}] Epoch {epoch} | Avg Loss: {total_loss/len(loader):.4f}")

        self._save_checkpoint(checkpoint_name)

    # ------------------------------------------------------------------
    # Phase 2: Full NER fine-tuning on English + code-switch augmentation
    # ------------------------------------------------------------------
    def ner_finetune(self, lang="en", augment_langs=("tr", "de", "ar"), freeze_backbone=False,
                      use_code_switch=True, checkpoint_name="ner_checkpoint.pt",
                      route_through_projection=False):
        self.route_through_projection = route_through_projection
        mode = "FROZEN linear probe" if freeze_backbone else "full backbone fine-tune"
        cs_desc = f"code-switch augmentation: {augment_langs}" if use_code_switch else "no code-switch augmentation"
        proj_desc = "routed through projection head" if route_through_projection else "raw backbone embeddings"
        print(f"\n=== Phase 2: {mode} on {lang} ({cs_desc}, {proj_desc}) ===")

        if use_code_switch:
            lexicons = {l: BilingualLexicon(target_lang=l, source_lang=lang) for l in augment_langs}
            augmenter = CodeSwitchAugmenter(lexicons, switch_prob=self.config.get("switch_prob", 0.3))
            dataset = self.processor.load_ner_dataset_augmented(
                lang, augmenter, augment_ratio=self.config.get("augment_ratio", 0.5)
            )
        else:
            dataset = self.processor.load_ner_dataset(lang)

        train_loader = DataLoader(dataset["train"], batch_size=self.config.get("batch_size", 16), shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset["validation"], batch_size=32, num_workers=0)

        self.ner_head = NERClassifier(
            hidden_size=self.model.backbone.config.hidden_size,
            num_labels=self.num_labels,
        ).to(self.device)

        for p in self.model.backbone.parameters():
            p.requires_grad = not freeze_backbone

        if freeze_backbone:
            # Linear probe: only the NER head is trained (mirrors the original prototype).
            optimizer = torch.optim.AdamW(
                self.ner_head.parameters(), lr=self.config.get("head_lr", 1e-3), weight_decay=0.01
            )
        else:
            optimizer = self._build_layerwise_optimizer(
                base_lr=self.config.get("ner_lr", 2e-5),
                head_lr=self.config.get("head_lr", 1e-3),
                layer_decay=self.config.get("layer_decay", 0.95),
            )
        epochs = self.config.get("ner_epochs", 5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        best_f1 = 0.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.ner_head.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                token_embs = self.get_ner_embeddings(input_ids, attention_mask)
                logits = self.ner_head(token_embs)
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.ner_head.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            val_f1 = self._evaluate(val_loader, label=f"EN val epoch {epoch}")
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                self._save_checkpoint(checkpoint_name, include_head=True)

        print(f"Best EN val F1: {best_f1:.4f}")
        for p in self.model.backbone.parameters():
            p.requires_grad = True

    def _build_layerwise_optimizer(self, base_lr=2e-5, head_lr=1e-3, layer_decay=0.95, weight_decay=0.01):
        """Lower LR for embeddings/early layers, higher LR for later layers and heads."""
        backbone = self.model.backbone
        num_layers = backbone.config.num_hidden_layers
        groups = []

        groups.append({
            "params": backbone.embeddings.parameters(),
            "lr": base_lr * (layer_decay ** num_layers),
            "weight_decay": weight_decay,
        })
        for i, layer in enumerate(backbone.encoder.layer):
            lr = base_lr * (layer_decay ** (num_layers - i - 1))
            groups.append({"params": layer.parameters(), "lr": lr, "weight_decay": weight_decay})

        # projection_head is only included when NER training actually routes through it
        # (self.route_through_projection) - otherwise it's unused in this forward path,
        # and including it would only apply decoupled weight decay to dead parameters.
        if self.route_through_projection:
            groups.append({"params": self.model.projection_head.parameters(), "lr": head_lr, "weight_decay": weight_decay})
        groups.append({"params": self.ner_head.parameters(), "lr": head_lr, "weight_decay": weight_decay})

        return torch.optim.AdamW(groups)

    # ------------------------------------------------------------------
    # Phase 3: Zero-shot evaluation across multiple target languages
    # ------------------------------------------------------------------
    def zero_shot_eval_multi(self, target_langs=("tr", "de", "ar"), checkpoint_name="ner_checkpoint.pt"):
        print(f"\n=== Phase 3: Zero-Shot Evaluation across {target_langs} ===")
        self._load_checkpoint(checkpoint_name, include_head=True)

        results = {}
        for lang in target_langs:
            dataset = self.processor.load_ner_dataset(lang)
            test_loader = DataLoader(dataset["test"], batch_size=32, num_workers=0)
            f1 = self._evaluate(test_loader, label=f"{lang} test (zero-shot)")
            results[lang] = f1

        avg = sum(results.values()) / len(results)
        print("\n--- Zero-Shot Summary ---")
        for lang, f1 in results.items():
            print(f"  {lang}: {f1:.4f}")
        print(f"  Average: {avg:.4f}")
        return results

    # ------------------------------------------------------------------
    # Few-shot recovery: continue fine-tuning on a small labeled target-language
    # subset (Task #7) - NOT part of the main zero-shot pipeline. Assumes a
    # checkpoint (e.g. the config E headline run) is already loaded.
    # ------------------------------------------------------------------
    def few_shot_finetune(self, lang, k_examples, epochs=15, lr=1e-5, batch_size=8):
        print(f"\n=== Few-shot: {lang}, k={k_examples} labeled examples, {epochs} epochs, lr={lr} ===")
        dataset = self.processor.load_ner_dataset(lang)
        few_shot_train = dataset["train"].select(range(k_examples))
        train_loader = DataLoader(
            few_shot_train, batch_size=min(batch_size, k_examples), shuffle=True, num_workers=0
        )
        test_loader = DataLoader(dataset["test"], batch_size=32, num_workers=0)

        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.ner_head.parameters()), lr=lr
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        for epoch in range(epochs):
            self.model.train()
            self.ner_head.train()
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                token_embs = self.get_ner_embeddings(input_ids, attention_mask)
                logits = self.ner_head(token_embs)
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self._evaluate(test_loader, label=f"{lang} few-shot k={k_examples}")

    # ------------------------------------------------------------------
    # Translate-train baseline (Task #8): trains on the FULL English set after
    # word-level lexicon substitution into the target language (not mixed with
    # real English, unlike code-switch augmentation), then evaluates on the
    # real target-language test set. Standard alternative to zero-shot transfer
    # in the cross-lingual NER literature - this is a simplified
    # lexicon-substitution proxy for it, not full MT-based translate-train (see
    # load_ner_dataset_translated() docstring for the caveat).
    # ------------------------------------------------------------------
    def translate_train_finetune(self, target_lang, epochs=5, checkpoint_name=None):
        print(f"\n=== Translate-train (lexicon substitution): full fine-tune on pseudo-{target_lang} data ===")
        lexicon = {target_lang: BilingualLexicon(target_lang=target_lang, source_lang="en")}
        augmenter = CodeSwitchAugmenter(lexicon, switch_prob=1.0)
        train_ds = self.processor.load_ner_dataset_translated("en", augmenter)
        train_loader = DataLoader(train_ds, batch_size=self.config.get("batch_size", 16), shuffle=True, num_workers=0)

        self.ner_head = NERClassifier(
            hidden_size=self.model.backbone.config.hidden_size,
            num_labels=self.num_labels,
        ).to(self.device)
        for p in self.model.backbone.parameters():
            p.requires_grad = True

        optimizer = self._build_layerwise_optimizer(
            base_lr=self.config.get("ner_lr", 2e-5),
            head_lr=self.config.get("head_lr", 1e-3),
            layer_decay=self.config.get("layer_decay", 0.95),
        )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.ner_head.train()
            total_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                token_embs = self.get_ner_embeddings(input_ids, attention_mask)
                logits = self.ner_head(token_embs)
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.ner_head.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")

        if checkpoint_name:
            self._save_checkpoint(checkpoint_name, include_head=True)

        dataset = self.processor.load_ner_dataset(target_lang)
        test_loader = DataLoader(dataset["test"], batch_size=32, num_workers=0)
        return self._evaluate(test_loader, label=f"{target_lang} translate-train test")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _evaluate(self, loader, label="eval"):
        from seqeval.metrics import f1_score, classification_report
        self.model.eval()
        self.ner_head.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                token_embs = self.get_ner_embeddings(input_ids, attention_mask)
                logits = self.ner_head(token_embs)
                preds = logits.argmax(dim=-1)

                for pred_seq, label_seq in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                    pred_tags = [self.id2label[p] for p, l in zip(pred_seq, label_seq) if l != -100]
                    true_tags = [self.id2label[l] for l in label_seq if l != -100]
                    all_preds.append(pred_tags)
                    all_labels.append(true_tags)

        f1 = f1_score(all_labels, all_preds)
        print(f"  [{label}] F1: {f1:.4f}")
        print(classification_report(all_labels, all_preds))
        return f1

    def _save_checkpoint(self, filename, include_head=False):
        os.makedirs("checkpoints", exist_ok=True)
        state = {"model": self.model.state_dict()}
        if include_head and self.ner_head is not None:
            state["ner_head"] = self.ner_head.state_dict()
        path = os.path.join("checkpoints", filename)
        torch.save(state, path)
        print(f"Saved: {path}")

    def _load_checkpoint(self, filename, include_head=False):
        path = os.path.join("checkpoints", filename)
        if not os.path.exists(path):
            print(f"No checkpoint at {path}, skipping.")
            return
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"])
        if include_head and "ner_head" in state:
            if self.ner_head is None:
                self.ner_head = NERClassifier(
                    hidden_size=self.model.backbone.config.hidden_size,
                    num_labels=self.num_labels,
                ).to(self.device)
            self.ner_head.load_state_dict(state["ner_head"])
        print(f"Loaded: {path}")

    def run(self, target_langs=("tr", "de", "ar"), skip_contrastive=False):
        """Full pipeline: align (real parallel data) -> full fine-tune on EN (+ code-switch) -> multi-lang zero-shot eval."""
        if not skip_contrastive:
            self.contrastive_train(target_langs=target_langs)
        self.ner_finetune(lang="en", augment_langs=target_langs)
        return self.zero_shot_eval_multi(target_langs=target_langs)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Zero-shot cross-lingual NER via contrastive alignment + code-switching."
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--target-langs", nargs="+", default=None, metavar="LANG",
        help="Target languages for zero-shot eval (overrides config.target_langs)"
    )
    parser.add_argument("--skip-contrastive", action="store_true",
                        help="Skip Phase 1 contrastive alignment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Scalar hyperparameter overrides - each maps onto a config dict key
    _HP = [
        ("--base-model",             str,   "base_model"),
        ("--batch-size",             int,   "batch_size"),
        ("--lr",                     float, "lr"),
        ("--ner-lr",                 float, "ner_lr"),
        ("--head-lr",                float, "head_lr"),
        ("--layer-decay",            float, "layer_decay"),
        ("--contrastive-epochs",     int,   "contrastive_epochs"),
        ("--ner-epochs",             int,   "ner_epochs"),
        ("--max-parallel-examples",  int,   "max_parallel_examples"),
        ("--switch-prob",            float, "switch_prob"),
        ("--augment-ratio",          float, "augment_ratio"),
        ("--temp",                   float, "temp"),
        ("--max-seq-length",         int,   "max_seq_length"),
    ]
    for flag, typ, key in _HP:
        parser.add_argument(flag, type=typ, default=None,
                            dest=key, metavar=key.upper(),
                            help=f"Override config.{key}")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides (only keys that were explicitly passed)
    for _flag, _typ, key in _HP:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    target_langs = args.target_langs or config.get("target_langs", ["tr", "de", "ar"])

    set_seed(args.seed)
    trainer = ZeroShotNERTrainer(config)
    trainer.run(target_langs=tuple(target_langs), skip_contrastive=args.skip_contrastive)
