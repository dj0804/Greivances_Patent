"""
Script to train the urgency classification model.

Usage:
    python scripts/train_model.py --data data/processed/ \
                                   --model-type cnn_bilstm \
                                   --output outputs/models/
"""

import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from src.models import CNNModel, BiLSTMModel, CNNBiLSTMModel
from src.training import ModelTrainer
from src.config import MODEL_CONFIG, TRAINING_CONFIG, MODELS_DIR, LOGS_DIR
from src.utils import setup_logger, load_json


def load_data(data_dir: Path):
    """Load and prepare training data."""
    # Placeholder - implement actual data loading
    # This should load your preprocessed sequences and labels
    logger = setup_logger("train_model")
    logger.info(f"Loading data from {data_dir}")
    
    # TODO: Implement actual data loading
    # For now, return dummy data
    X = np.random.randint(0, 10000, (1000, 100))
    y = np.random.randint(0, 3, (1000, 3))  # One-hot encoded
    
    return X, y


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
    X, y = load_data(Path(args.data))
    
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
        vocab_size=MODEL_CONFIG["max_vocab_size"],
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
        patience=TRAINING_CONFIG["early_stopping_patience"]
    )
    
    # Save final model
    output_path = Path(args.output) / f"model_{args.model_type}_final.h5"
    trainer.save_model(output_path)
    
    logger.info(f"Training complete! Model saved to {output_path}")


if __name__ == "__main__":
    main()
