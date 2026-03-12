"""
Centralized configuration for the grievance system.
"""
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Raw data subdirectories
COMPLAINTS_DIR = RAW_DATA_DIR / "complaints"
METADATA_DIR = RAW_DATA_DIR / "metadata"

# Processed data subdirectories
TOKENIZED_DIR = PROCESSED_DATA_DIR / "tokenized"
EMBEDDINGS_DIR = PROCESSED_DATA_DIR / "embeddings"
TEMPORAL_JSON_DIR = PROCESSED_DATA_DIR / "temporal_json"

# Output directories
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Model configuration
MODEL_CONFIG = {
    "embedding_dim": 300,
    "max_sequence_length": 100,
    "cnn_filters": 128,
    "cnn_kernel_size": 5,
    "lstm_units": 64,
    "dropout_rate": 0.5,
    "dense_units": [128, 64],
    "num_classes": 3,  # Low, Medium, High urgency
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "early_stopping_patience": 5,
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    "spacy_model": "en_core_web_sm",
    "max_vocab_size": 10000,
    "min_word_freq": 2,
    "urgency_keywords": ["not", "no", "now", "but", "yet", "never", "only"],
    "special_tags": [
        "[Multiple Exclamations]",
        "[Impatient Questioning]",
        "[Emphasis]",
        "[CAPS_ON]"
    ],
}

# Summarization configuration
ENABLE_SUMMARIZATION = True
SUMMARIZER_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
MAX_SUMMARY_LENGTH = 60
MIN_SUMMARY_LENGTH = 10

# FastText model path (update with actual path)
FASTTEXT_MODEL_PATH = os.getenv(
    "FASTTEXT_MODEL_PATH",
    str(EXTERNAL_DATA_DIR / "cc.en.300.bin")
)

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "formatter": "default",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}
