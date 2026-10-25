from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models" / "distilbert_sarcasm" / "final_model"
OUTPUT_PATH = ROOT_DIR / "results" / "sample_predictions.csv"

SAMPLE_TEXTS = [
    ("I love it when my alarm goes off at 5am on a Monday.", 1),
    ("That was a very polite and completely normal response.", 0),
    ("Oh great, another hour of waiting in traffic.", 1),
    ("The presentation was clear and well organized.", 0),
    ("Fantastic, just what I needed: another bug in production.", 1),
    ("I had a wonderful day at work and everything went smoothly.", 0),
]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()

    texts = [text for text, _ in SAMPLE_TEXTS]
    labels = [label for _, label in SAMPLE_TEXTS]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        pred_labels = torch.argmax(probs, dim=-1).tolist()
        confident_scores = probs.max(dim=-1).values.tolist()

    rows = []
    for idx, (text, expected_label) in enumerate(SAMPLE_TEXTS):
        predicted_label = int(pred_labels[idx])
        confidence = round(float(confident_scores[idx]), 4)
        predicted_prob = round(float(probs[idx][predicted_label]), 4)
        rows.append(
            {
                "text": text,
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "predicted_probability": predicted_prob,
            }
        )

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("Sample predictions saved to:", OUTPUT_PATH)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
