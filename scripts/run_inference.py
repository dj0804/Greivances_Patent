"""
Script to run inference on new complaints.

Usage:
    python scripts/run_inference.py --model outputs/models/best_model.h5 \
                                     --input data/raw/complaints/new_complaints.txt \
                                     --output outputs/reports/predictions.json
"""

import argparse
from pathlib import Path
import json
from datetime import datetime

from src.inference import GrievancePredictor
from src.decision_engine import UrgencyDecisionEngine
from src.utils import setup_logger, save_json
from src.config import FASTTEXT_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Run inference on complaints")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input", type=str, required=True, help="Input complaints file")
    parser.add_argument("--output", type=str, required=True, help="Output predictions file")
    parser.add_argument("--fasttext-model", type=str, default=FASTTEXT_MODEL_PATH, help="FastText model path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("inference")
    
    # Initialize predictor
    logger.info(f"Loading model from {args.model}")
    predictor = GrievancePredictor(
        model_path=Path(args.model),
        fasttext_model_path=Path(args.fasttext_model) if args.fasttext_model else None
    )
    
    # Initialize decision engine
    decision_engine = UrgencyDecisionEngine()
    
    # Load complaints
    logger.info(f"Loading complaints from {args.input}")
    with open(args.input, 'r') as f:
        complaints = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Processing {len(complaints)} complaints...")
    
    # Run predictions
    results = []
    for i, complaint_text in enumerate(complaints):
        # Make prediction
        prediction = predictor.predict(
            complaint_text=complaint_text,
            timestamp=datetime.now(),
            return_probabilities=True
        )
        
        # Apply decision engine
        decision = decision_engine.make_decision(
            model_prediction=prediction,
            temporal_features=prediction.get('temporal_features')
        )
        
        result = {
            'id': i,
            'text': complaint_text,
            'cleaned_text': prediction['cleaned_text'],
            'urgency_level': decision['urgency_level'],
            'confidence': decision['confidence'],
            'probabilities': prediction.get('probabilities'),
            'decision_details': decision,
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(result)
        
        # Log progress
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(complaints)} complaints")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving predictions to {output_path}")
    save_json(results, output_path)
    
    # Print summary
    urgency_counts = {}
    for result in results:
        level = result['urgency_level']
        urgency_counts[level] = urgency_counts.get(level, 0) + 1
    
    logger.info("\nPrediction Summary:")
    for level, count in sorted(urgency_counts.items()):
        logger.info(f"  {level}: {count} ({count/len(results)*100:.1f}%)")
    
    logger.info(f"\nDone! Processed {len(results)} complaints")


if __name__ == "__main__":
    main()
