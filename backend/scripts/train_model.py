"""
Script to train the urgency classification model.

Usage:
    python scripts/train_model.py --data data/processed/ \
                                   --model-type cnn_bilstm \
                                   --output outputs/models/
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import CNNModel, BiLSTMModel, CNNBiLSTMModel
from src.training import ModelTrainer
from src.config import MODEL_CONFIG, TRAINING_CONFIG, PREPROCESSING_CONFIG, MODELS_DIR, LOGS_DIR
from src.utils import setup_logger, load_json


import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

def load_data(data_path: Path):
    """Load and prepare training data."""
    logger = setup_logger("train_model")
    logger.info(f"Loading data from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    texts = []
    label_strings = []

    for item in data:
        texts.append(item["text"])
        raw = item["label"]
        # Normalise any legacy "Critical" to "High"
        label_strings.append("High" if raw == "Critical" else raw)

    logger.info(f"Loaded {len(texts)} samples")

    # Fit a LabelEncoder so inference can invert integer predictions
    label_encoder = LabelEncoder()
    label_encoder.fit(["Low", "Medium", "High"])
    int_labels = label_encoder.transform(label_strings)

    # Tokenization
    tokenizer = Tokenizer(num_words=PREPROCESSING_CONFIG["max_vocab_size"], oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    # Padding
    X = pad_sequences(sequences, maxlen=MODEL_CONFIG["max_sequence_length"], padding='post', truncating='post')

    # One-hot encode labels
    y = to_categorical(int_labels, num_classes=MODEL_CONFIG["num_classes"])

    return X, y, tokenizer, label_encoder, int_labels


def main():
    parser = argparse.ArgumentParser(description="Train urgency classification model")
    parser.add_argument("--data", type=str, required=True, help="Path to processed data")
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn_bilstm",
        choices=["cnn", "bilstm", "cnn_bilstm"],
        help="Model architecture"
    )
    parser.add_argument("--output", type=str, default=str(MODELS_DIR), help="Output directory")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"], help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG["batch_size"], help="Batch size")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("train_model", LOGS_DIR / "training.log")
    
    # Load data
    data_file = Path(args.data)
    if data_file.is_dir():
        # Fallback if a directory was passed instead of file
        data_file = data_file / "mock_training_data.json"
        
    X, y, tokenizer, label_encoder, int_labels = load_data(data_file)

    # Save the fitted tokenizer so it can be used during inference
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.pkl"
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Fitted vocabulary size: {len(tokenizer.word_index)} words")
    logger.info(f"Saved tokenizer to {tokenizer_path}")
    print(f"Saved tokenizer to {tokenizer_path}")

    # Save the fitted label encoder
    label_encoder_path = output_dir / "label_encoder.pkl"
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Saved label encoder to {label_encoder_path}")
    print(f"Saved label encoder to {label_encoder_path}")

    # Compute class weights from full integer label array
    import numpy as np
    classes = np.unique(int_labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=int_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Class weights: {class_weight_dict}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TRAINING_CONFIG["validation_split"],
        random_state=42
    )

    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model")
    
    if args.model_type == "cnn":
        model_class = CNNModel
    elif args.model_type == "bilstm":
        model_class = BiLSTMModel
    else:
        model_class = CNNBiLSTMModel
    
    model = model_class(
        vocab_size=PREPROCESSING_CONFIG["max_vocab_size"],
        embedding_dim=MODEL_CONFIG["embedding_dim"],
        max_length=MODEL_CONFIG["max_sequence_length"],
        num_classes=MODEL_CONFIG["num_classes"]
    )
    
    model.compile()
    logger.info("Model compiled successfully")
    
    # Train model
    trainer = ModelTrainer(
        model=model.get_model(),
        model_dir=Path(args.output),
        log_dir=LOGS_DIR / "tensorboard"
    )
    
    logger.info("Starting training...")
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=TRAINING_CONFIG["early_stopping_patience"],
        class_weight=class_weight_dict,
    )
    
    # Save final model
    # Note: API expects 'best_model.h5' - keep this name consistent
    output_path = Path(args.output) / "best_model.h5"
    trainer.save_model(output_path)
    
    # Also save with model type for reference
    backup_path = Path(args.output) / f"model_{args.model_type}_final.h5"
    trainer.save_model(backup_path)
    
    logger.info(f"Training complete! Model saved to {output_path}")
    logger.info(f"Backup copy saved to {backup_path}")


if __name__ == "__main__":
    main()
