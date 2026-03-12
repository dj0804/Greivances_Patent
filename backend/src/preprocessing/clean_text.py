"""
Text cleaning module for grievance preprocessing.
Handles emoji conversion, punctuation patterns, and normalization.
"""

import regex as re
import emoji as ej
import contractions


class TextCleaner:
    """
    Cleans and normalizes grievance text while preserving emotional indicators.
    """
    
    def __init__(self):
        self.multiple_excl = re.compile(r"!{2,}")
        self.impatient_ques = re.compile(r"\?{2,}")
        self.emphasis = re.compile(r"[!?]{2,}")
        self.noise_chars = re.compile(r"[#$^*+]")
        self.extra_whitespace = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        """
        Apply full cleaning pipeline to input text.
        
        Args:
            text: Raw grievance text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        text = self._caps_check(text)
        text = self._emoji_to_text(text)
        text = self._punctuation_barrage(text)
        text = self._fix_contractions(text)
        text = self._regex_clean(text)
        text = self._whitespace_clean(text)
        
        return text.strip().lower()
    
    def _caps_check(self, text: str) -> str:
        """Detect all-caps text indicating urgency/shouting."""
        if text.isupper() and len(text) > 3:
            return f"[CAPS_ON] {text}"
        return text

    def _emoji_to_text(self, text: str) -> str:
        """Convert emojis to text descriptions."""
        return ej.demojize(text, delimiters=('[', ']'))

    def _punctuation_barrage(self, text: str) -> str:
        """Detect excessive punctuation patterns."""
        text = self.multiple_excl.sub(" [Multiple Exclamations] ", text)
        text = self.impatient_ques.sub(" [Impatient Questioning] ", text)
        text = self.emphasis.sub(" [Emphasis] ", text)
        return text
    
    def _fix_contractions(self, text: str) -> str:
        """Expand contractions (e.g., can't -> cannot)."""
        return contractions.fix(text)

    def _regex_clean(self, text: str) -> str:
        """Remove noise characters."""
        return self.noise_chars.sub("", text)

    def _whitespace_clean(self, text: str) -> str:
        """Normalize whitespace."""
        return self.extra_whitespace.sub(" ", text)


def clean_text_func(text: str) -> str:
    """
    Convenience function for text cleaning.
    
    Args:
        text: Raw grievance text
        
    Returns:
        Cleaned text
    """
    cleaner = TextCleaner()
    return cleaner.clean(text)
