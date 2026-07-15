"""
demo.py
Task #11: Inference demo for zero-shot cross-lingual NER.

Loads the trained checkpoint (checkpoints/ner_checkpoint.pt) and tags
arbitrary text in any language - no retraining, no labels needed.

Modes
-----
CLI (default):
    python -m src.demo --text "Angela Merkel visited Berlin last Tuesday."
    python -m src.demo --text "Kerem İstanbul'da çalışıyor."
    python -m src.demo --interactive

Web UI (requires: pip install gradio):
    python -m src.demo --web

Options
-------
--checkpoint PATH   NER checkpoint to load  (default: checkpoints/ner_checkpoint.pt)
--config PATH       YAML config             (default: configs/config.yaml)
--text TEXT         Text to tag (CLI mode)
--interactive       Read lines from stdin until EOF (Ctrl-D / Ctrl-Z)
--web               Launch Gradio web demo
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contrastive_model import ZeroShotAligner, NERClassifier
from src.data_processing import MultiLingualNERPipeline

# IOB2 label set (must match training)
LABEL2COLOR = {
    "O":       "",
    "B-PER":   "\033[92m",   # green
    "I-PER":   "\033[92m",
    "B-ORG":   "\033[94m",   # blue
    "I-ORG":   "\033[94m",
    "B-LOC":   "\033[93m",   # yellow
    "I-LOC":   "\033[93m",
}
RESET = "\033[0m"

NUM_LABELS = 7


def _load_model(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    processor = MultiLingualNERPipeline(
        model_name=config["base_model"],
        max_length=config.get("max_seq_length", 128),
    )
    model = ZeroShotAligner(config["base_model"]).to(device)
    head = NERClassifier(
        hidden_size=model.backbone.config.hidden_size,
        num_labels=NUM_LABELS,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run the full pipeline first: python -m src.train_zero_shot"
        )

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    if "ner_head" not in state:
        raise KeyError(
            "Checkpoint does not contain a NER head. "
            "Make sure to use a checkpoint saved after Phase 2 (ner_checkpoint.pt)."
        )
    head.load_state_dict(state["ner_head"])
    model.eval()
    head.eval()

    return model, head, processor


def tag_text(text, model, head, processor, device):
    """
    Tokenize `text`, run the model, and return a list of (word, label) pairs
    where each label is the IOB2 tag for that word's first subword token.
    """
    tokenizer = processor.tokenizer
    id2label = processor.id2label

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=processor.max_length,
        return_offsets_mapping=True,
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        token_embs = model.get_token_embeddings(input_ids, attention_mask)
        logits = head(token_embs)
        pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()

    # Reconstruct word-level predictions: take the label of the first subword
    # per word (same convention as subword label alignment in data_processing.py).
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    word_preds = []
    current_word = ""
    current_label = "O"
    first_subword = True

    for token, pred_id, (start, end) in zip(tokens, pred_ids, offset_mapping):
        if token in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
            continue
        if start == 0 and end == 0:
            continue

        label = id2label[pred_id]
        is_continuation = token.startswith("▁") is False and not first_subword

        if token.startswith("▁") or first_subword:
            # New word (XLM-R uses '▁' as word-start marker)
            if current_word:
                word_preds.append((current_word.lstrip("▁"), current_label))
            current_word = token
            current_label = label
            first_subword = False
        else:
            current_word += token

    if current_word:
        word_preds.append((current_word.lstrip("▁"), current_label))

    return word_preds


def render_cli(word_preds, use_color=True):
    """Format word_preds as a colored token string + aligned entity list."""
    lines = []
    colored_tokens = []
    entities = []
    current_entity_words = []
    current_entity_label = None

    for word, label in word_preds:
        color = LABEL2COLOR.get(label, "")
        if use_color and color:
            colored_tokens.append(f"{color}{word}{RESET}")
        else:
            colored_tokens.append(f"[{label}]{word}" if label != "O" else word)

        # Entity span accumulation
        base = label[2:] if label.startswith(("B-", "I-")) else None
        if label.startswith("B-"):
            if current_entity_words:
                entities.append((" ".join(current_entity_words), current_entity_label))
            current_entity_words = [word]
            current_entity_label = base
        elif label.startswith("I-") and current_entity_label == base:
            current_entity_words.append(word)
        else:
            if current_entity_words:
                entities.append((" ".join(current_entity_words), current_entity_label))
            current_entity_words = []
            current_entity_label = None

    if current_entity_words:
        entities.append((" ".join(current_entity_words), current_entity_label))

    lines.append(" ".join(colored_tokens))
    if entities:
        lines.append("")
        lines.append("Entities found:")
        for span, etype in entities:
            color = LABEL2COLOR.get(f"B-{etype}", "")
            if use_color and color:
                lines.append(f"  {color}{span}{RESET}  [{etype}]")
            else:
                lines.append(f"  {span}  [{etype}]")
    else:
        lines.append("(no entities detected)")
    return "\n".join(lines)


def launch_gradio(model, head, processor, device):
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    def predict(text):
        if not text.strip():
            return "Enter some text above."
        try:
            word_preds = tag_text(text, model, head, processor, device)
        except Exception as e:
            return f"Error: {e}"

        # Build HTML with colored spans
        html_parts = []
        current_entity = []
        current_label = None

        def flush_entity():
            if current_entity and current_label:
                span_text = " ".join(current_entity)
                colors = {"PER": "#a8f0c6", "ORG": "#a8c8f0", "LOC": "#f0e0a8"}
                bg = colors.get(current_label, "#e0e0e0")
                html_parts.append(
                    f'<mark style="background:{bg};padding:2px 4px;border-radius:3px">'
                    f'{span_text} <sup style="font-size:0.7em">{current_label}</sup></mark>'
                )

        for word, label in word_preds:
            base = label[2:] if label.startswith(("B-", "I-")) else None
            if label.startswith("B-"):
                flush_entity()
                current_entity = [word]
                current_label = base
            elif label.startswith("I-") and current_label == base:
                current_entity.append(word)
            else:
                flush_entity()
                current_entity = []
                current_label = None
                html_parts.append(word)

        flush_entity()
        return " ".join(html_parts)

    examples = [
        ["Angela Merkel visited Berlin last Tuesday."],
        ["Kerem İstanbul'da çalışıyor ve Türk Hava Yolları'nda çalışıyor."],
        ["Der Bundesrat traf sich in Berlin, um über die Wirtschaft zu diskutieren."],
        ["محمد يعيش في القاهرة ويعمل في منظمة الأمم المتحدة."],
        ["Samsung Electronics a annoncé un partenariat avec l'Université de Paris."],
    ]

    iface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Enter text in any language...",
            label="Input Text",
        ),
        outputs=gr.HTML(label="Named Entities"),
        title="Zero-Shot Cross-Lingual NER",
        description=(
            "XLM-RoBERTa fine-tuned on English WikiANN + contrastive alignment + "
            "code-switching augmentation. Tags PER / ORG / LOC in any language - "
            "no target-language labels used during training."
        ),
        examples=examples,
        allow_flagging="never",
    )
    iface.launch()


def main():
    parser = argparse.ArgumentParser(description="Zero-shot cross-lingual NER demo")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/ner_checkpoint.pt")
    parser.add_argument("--text", type=str, default=None, help="Text to tag")
    parser.add_argument("--interactive", action="store_true",
                        help="Read lines from stdin")
    parser.add_argument("--web", action="store_true",
                        help="Launch Gradio web interface")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI color codes in CLI output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.checkpoint} on {device}...")
    model, head, processor = _load_model(args.config, args.checkpoint, device)
    print("Model loaded.\n")

    if args.web:
        launch_gradio(model, head, processor, device)
        return

    use_color = not args.no_color and sys.stdout.isatty()

    if args.text:
        preds = tag_text(args.text, model, head, processor, device)
        print(render_cli(preds, use_color=use_color))
        return

    if args.interactive:
        print("Interactive mode - enter text (Ctrl-D to quit):\n")
        try:
            while True:
                try:
                    line = input("> ").strip()
                except EOFError:
                    break
                if not line:
                    continue
                preds = tag_text(line, model, head, processor, device)
                print(render_cli(preds, use_color=use_color))
                print()
        except KeyboardInterrupt:
            pass
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
