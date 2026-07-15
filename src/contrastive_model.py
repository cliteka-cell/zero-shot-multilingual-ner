"""
contrastive_model.py
Core Contrastive Learning Module for Zero-Shot NER Alignment.

SimCSE-style InfoNCE loss pulls embeddings of source/target language
sentences together in a shared vector space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class InfoNCELoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy).
    Pulls matched pairs together and pushes unmatched pairs apart.
    Bidirectional: loss computed in both src->tgt and tgt->src directions.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings1, embeddings2):
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)
        labels = torch.arange(len(embeddings1), device=embeddings1.device)
        logits = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss


class ZeroShotAligner(nn.Module):
    """
    XLM-RoBERTa backbone with a projection head for cross-lingual alignment.
    The projection head is trained via contrastive loss; the backbone provides
    pretrained multilingual representations.
    """
    def __init__(self, model_name="xlm-roberta-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, tokens):
        """Returns normalized projected [CLS] embeddings for contrastive training."""
        outputs = self.backbone(**tokens)
        cls = outputs.last_hidden_state[:, 0, :]
        projected = self.projection_head(cls)
        return F.normalize(projected, dim=1)

    def get_token_embeddings(self, input_ids, attention_mask):
        """Returns full sequence token embeddings for NER head training/inference."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # [batch, seq_len, hidden_size]

    def get_projected_token_embeddings(self, input_ids, attention_mask):
        """Routes raw token embeddings through the projection head (nn.Linear applies
        to the last dim regardless of leading dims, so this works token-wise the same
        way forward() applies it to a single [CLS] vector). Lets NER training/eval
        actually use whatever the contrastive phase learned in the projection head,
        instead of bypassing it via get_token_embeddings()."""
        token_embs = self.get_token_embeddings(input_ids, attention_mask)
        return self.projection_head(token_embs)


class NERClassifier(nn.Module):
    """Linear NER head trained on English, applied zero-shot to target language."""
    def __init__(self, hidden_size=768, num_labels=7, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, token_embeddings):
        return self.classifier(self.dropout(token_embeddings))
