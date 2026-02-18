"""
Script to build and save embeddings for the training dataset.

Usage:
    python scripts/build_embeddings.py --data data/processed/tokenized/train.json \
                                        --output data/processed/embeddings/ \
                                        --fasttext-model path/to/fasttext/model.bin
"""

import argparse
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

from src.embeddings import FastTextEncoder, VectorStore
from src.utils import setup_logger, save_json, load_json


def main():
    parser = argparse.ArgumentParser(description="Build embeddings from tokenized data")
    parser.add_argument("--data", type=str, required=True, help="Path to tokenized data")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--fasttext-model", type=str, required=True, help="FastText model path")
    parser.add_argument("--build-vector-store", action="store_true", help="Build vector store")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("build_embeddings")
    
    # Initialize FastText encoder
    logger.info(f"Loading FastText model from {args.fasttext_model}")
    encoder = FastTextEncoder(args.fasttext_model)
    
    # Load tokenized data
    logger.info(f"Loading tokenized data from {args.data}")
    data = load_json(Path(args.data))
    
    # Process embeddings
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings = []
    vector_store = None
    
    if args.build_vector_store:
        vector_store = VectorStore(dimension=encoder.vector_size)
    
    logger.info("Generating embeddings...")
    for idx, item in enumerate(tqdm(data)):
        tokens = item.get('tokens', [])
        complaint_id = item.get('id', f'complaint_{idx}')
        
        # Get sentence embedding
        embedding = encoder.encode_sentence(tokens)
        
        embeddings.append({
            'id': complaint_id,
            'embedding': embedding.tolist(),
        })
        
        # Add to vector store
        if vector_store:
            vector_store.add(
                vector=embedding,
                id=complaint_id,
                metadata=item.get('metadata', {})
            )
    
    # Save embeddings
    embeddings_file = output_dir / "embeddings.json"
    logger.info(f"Saving embeddings to {embeddings_file}")
    save_json(embeddings, embeddings_file)
    
    # Save as numpy array
    embeddings_array = np.array([e['embedding'] for e in embeddings])
    np.save(output_dir / "embeddings.npy", embeddings_array)
    
    # Save vector store
    if vector_store:
        store_path = output_dir / "vector_store.pkl"
        logger.info(f"Saving vector store to {store_path}")
        vector_store.save(store_path)
    
    logger.info(f"Done! Processed {len(embeddings)} complaints")


if __name__ == "__main__":
    main()
