import numpy as np
from gensim.models import fasttext

class FastTextEncoder:
    def __init__(self, model_path: str):
        print("Loading FastText model (this may take a minute)")
        self.model = fasttext.load_facebook_model(model_path)
        self.vector_size = self.model.vector_size

    def encode_tokens(self, tokens: list[str]) -> np.ndarray:
        vectors = [self.model.wv[token] for token in tokens]
        return np.array(vectors)

    def get_embedding_matrix(self, word_index: dict[str, int]) -> np.ndarray:
        """Creates the weight matrix for your Keras Embedding layer."""
        vocab_size = len(word_index) + 1
        embedding_matrix = np.zeros((vocab_size, self.vector_size))
        
        for word, i in word_index.items():
            if word in self.model.wv:
                embedding_matrix[i] = self.model.wv[word]
            else:
                embedding_matrix[i] = self.model.wv.get_vector(word)
                
        return embedding_matrix
    
