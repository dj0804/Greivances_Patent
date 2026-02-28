"""
Tokenization module with custom handling for urgency-related words.
"""

import spacy
from spacy.symbols import ORTH
from typing import List


class Tokenise:
    """
    Tokenizer with custom stop word handling and special tag preservation.
    """
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Preserve urgency-related words that are typically stop words
        urgency_words = {"not", "no", "now", "but", "yet", "never", "only"}
        for word in urgency_words:
            self.nlp.vocab[word].is_stop = False
        
        # Preserve special emotional tags as single tokens
        special_tags = [
            "[Multiple Exclamations]",
            "[Impatient Questioning]",
            "[Emphasis]",
            "[CAPS_ON]"
        ]
        for tag in special_tags:
            self.nlp.tokenizer.add_special_case(tag, [{ORTH: tag}])
    
    def tokenise(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text while filtering stop words and punctuation.
        
        Args:
            text: Cleaned text to tokenize
            
        Returns:
            List of tokens
        """
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        return tokens


# Singleton instance for efficiency
_shared_tokenizer = None


def get_tokens(text: str) -> List[str]:
    """
    Get tokens from text using shared tokenizer instance.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    global _shared_tokenizer
    if _shared_tokenizer is None:
        _shared_tokenizer = Tokenise()
    
    return _shared_tokenizer.tokenise(text)
