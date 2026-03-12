import os
import sys
import pickle
import numpy as np
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Src.config import PREPROCESSING_CONFIG, MODEL_CONFIG
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Src.Preprocessing.clean_text import TextCleaner

def test_model():
    print("Loading model and tokenizer...")
    
    # Paths
    model_path = Path("outputs/models/model_cnn_bilstm_final.h5")
    tokenizer_path = Path("outputs/models/tokenizer.pkl")
    
    if not model_path.exists() or not tokenizer_path.exists():
        print("Error: Could not find trained model or tokenizer in outputs/models/")
        return
        
    # Load Model
    model = load_model(str(model_path))
    
    # Load Tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
        
    print("✅ Model & Tokenizer loaded successfully!\n")
    
    # Labels corresponding to One-Hot Encoding used during training
    # 0 = Low, 1 = Medium, 2 = High (Critical was mapped to High)
    labels = ["Low", "Medium", "High"]
    
    cleaner = TextCleaner()
    
    print("=" * 60)
    print("🤖 AUTOMATED MODEL TESTING 🤖")
    print("Type your simulated complaint below to test the trained model.")
    print("Type 'quit' or 'q' to exit.")
    print("=" * 60)
    
    while True:
        text = input("\nEnter your complaint: ").strip()
        
        if text.lower() in ['q', 'quit', 'exit']:
            print("Exiting test mode. Goodbye!")
            break
            
        if not text:
            print("Please enter a valid complaint.")
            continue
            
        # Preprocess
        cleaned_text = cleaner.clean(text)
        
        # Tokenize & Sequence
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        
        # Pad sequence
        padded = pad_sequences(
            sequence, 
            maxlen=MODEL_CONFIG["max_sequence_length"], 
            padding='post', 
            truncating='post'
        )
        
        # Predict
        prediction_probs = model.predict(padded, verbose=0)[0]
        predicted_idx = np.argmax(prediction_probs)
        predicted_label = labels[predicted_idx]
        confidence = prediction_probs[predicted_idx] * 100
        
        # Output
        print("\n" + "-" * 40)
        print(f"Complaint: '{text}'")
        print(f"Prediction: [{predicted_label.upper()}] (Confidence: {confidence:.2f}%)")
        print("-" * 40)

if __name__ == "__main__":
    # Hide TensorFlow Info Logs for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    test_model()
