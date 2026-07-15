# Zero-Shot Multi-Lingual Named Entity Recognition

## Abstract
This project implements a **Zero-Shot Transfer Learning** framework for Named Entity Recognition (NER), addressing the challenge of extracting entity boundaries in low-resource languages (e.g., Turkish, Korean) without labeled data. 

By leveraging **Multilingual Transformer Embeddings** and **Contrastive Loss functions (InfoNCE)**, the model aligns English source entities with their semantic targets in a shared latent space, achieving transfer without explicit target-language supervision.

## Methodology
- **Architecture**: XLM-RoBERTa Backbone for universal token representation.
- **Alignment Strategy**: SimCSE-style contrastive learning to force translation alignment.
- **Optimization**: InfoNCE Loss to maximize cosine similarity between "Anchor" (English) and "Target" (Unlabeled) text clusters.

## Project Structure
- `src/data_processing.py`: Handles WikiANN data loading and Cross-lingual token alignment.
- `src/contrastive_model.py`: Implements the projection head and InfoNCE loss logic.
- `src/train_zero_shot.py`: The primary training orchestrator.
