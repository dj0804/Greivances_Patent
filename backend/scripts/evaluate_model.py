"""
Standalone evaluation script for the urgency classification model.

Usage (JSON, carves out 10%% test slice):
    python scripts/evaluate_model.py \
        --model outputs/models/best_model.h5 \
        --tokenizer outputs/models/tokenizer.pkl \
        --data data/raw/complaints/mock_training_data.json \
        --output outputs/reports/evaluation_report.json

Usage (dedicated CSV test file, evaluate everything):
    python scripts/evaluate_model.py \
        --model outputs/models/best_model.h5 \
        --tokenizer outputs/models/tokenizer.pkl \
        --data support_ticket_test.csv --full
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


def _read_records(data_path: Path):
    """
    Read records from a JSON or CSV file into a list of {'text', 'label'} dicts.

    JSON: list of dicts with 'text' and 'label' keys.
    CSV : auto-maps common column names (Query_Text/Priority_Label, etc.).
    """
    suffix = Path(data_path).suffix.lower()
    records = []

    if suffix == ".csv":
        with open(data_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            text_col = next(
                (c for c in ("Query_Text", "text", "Student_Query", "complaint_text") if c in cols),
                None,
            )
            label_col = next(
                (c for c in ("Priority_Label", "label", "urgency", "Priority") if c in cols),
                None,
            )
            if text_col is None or label_col is None:
                raise ValueError(
                    f"CSV missing recognizable text/label columns; found columns: {cols}"
                )
            for row in reader:
                records.append({"text": row[text_col], "label": row[label_col]})
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            records.append({"text": item["text"], "label": item["label"]})

    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate the urgency classification model")
    parser.add_argument("--model", type=str, required=True, help="Path to .h5 model file")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.pkl")
    parser.add_argument("--data", type=str, required=True, help="Path to JSON or CSV data file")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/reports/evaluation_report.json",
        help="Path to save evaluation results JSON",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate on the entire data file (use when --data is a dedicated test set). "
        "Default carves out a 10%% stratified slice for evaluation.",
    )
    args = parser.parse_args()

    import keras
    from keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

    from src.preprocessing.clean_text import TextCleaner

    # Load model
    print(f"Loading model from {args.model}")
    model = keras.models.load_model(args.model)

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)

    # Load data (JSON or CSV)
    print(f"Loading data from {args.data}")
    data = _read_records(Path(args.data))

    label_names = ["Low", "Medium", "High"]
    label_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 2}

    cleaner = TextCleaner()
    texts = []
    labels = []
    for item in data:
        texts.append(cleaner.clean(item["text"]))
        labels.append(label_map[item["label"]])

    texts = np.array(texts)
    labels = np.array(labels)

    if args.full:
        # Dedicated test file — evaluate on everything
        X_test_texts, y_test = texts, labels
    else:
        # 80/10/10 stratified split — use only the test 10%
        X_temp, X_test_texts, y_temp, y_test = train_test_split(
            texts, labels, test_size=0.10, random_state=42, stratify=labels
        )

    # Preprocess test texts
    sequences = tokenizer.texts_to_sequences(X_test_texts)
    X_test = pad_sequences(sequences, maxlen=100, padding="post", truncating="post")

    print(f"Test set size: {len(X_test)}")

    # Predict
    probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
    print("Confusion Matrix:")
    print(np.array(cm))

    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation report saved to {output_path}")


if __name__ == "__main__":
    main()
