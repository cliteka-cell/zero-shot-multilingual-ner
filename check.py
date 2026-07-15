"""
Quick sanity check: verifies imports and a single forward pass without
downloading the full dataset. Run with: python check.py
"""
import torch
from src.contrastive_model import ZeroShotAligner, InfoNCELoss, NERClassifier
from src.data_processing import MultiLingualNERPipeline, LABEL2ID

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer and model...")
    processor = MultiLingualNERPipeline(max_length=32)
    model = ZeroShotAligner().to(device)
    criterion = InfoNCELoss(temperature=0.07)
    ner_head = NERClassifier(hidden_size=model.backbone.config.hidden_size, num_labels=len(LABEL2ID)).to(device)

    # Contrastive forward pass
    texts_en = [["Apple", "is", "a", "company"], ["Paris", "is", "beautiful"]]
    texts_tr = [["Apple", "bir", "şirkettir"], ["Paris", "güzeldir"]]
    enc_en = processor.tokenizer(texts_en, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
    enc_tr = processor.tokenizer(texts_tr, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)

    model.train()
    emb_en = model(enc_en)
    emb_tr = model(enc_tr)
    loss = criterion(emb_en, emb_tr)
    print(f"Contrastive loss: {loss.item():.4f}")

    # NER head forward pass
    model.eval()
    with torch.no_grad():
        token_embs = model.get_token_embeddings(enc_en["input_ids"], enc_en["attention_mask"])
        logits = ner_head(token_embs)
    print(f"NER logits shape: {logits.shape}  (expected [2, seq_len, 7])")

    print("\nAll checks passed.")

if __name__ == "__main__":
    main()
