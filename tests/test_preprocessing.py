"""
Unit tests for preprocessing module.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from src.preprocessing import TextCleaner, Tokenise


class TestTextCleaner(unittest.TestCase):
    """Test cases for TextCleaner."""
    
    def setUp(self):
        self.cleaner = TextCleaner()
    
    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        text = "This is a test."
        result = self.cleaner.clean(text)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_caps_detection(self):
        """Test all-caps detection."""
        text = "URGENT ISSUE"
        result = self.cleaner.clean(text)
        self.assertIn("[caps_on]", result.lower())
    
    def test_emoji_conversion(self):
        """Test emoji to text conversion."""
        text = "I'm happy 😊"
        result = self.cleaner.clean(text)
        # Emoji should be converted to text
        self.assertNotIn("😊", result)
    
    def test_punctuation_barrage(self):
        """Test excessive punctuation detection."""
        text = "Help me now!!!"
        result = self.cleaner.clean(text)
        self.assertIn("multiple exclamations", result.lower())
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.cleaner.clean("")
        self.assertEqual(result, "")
    
    def test_contraction_expansion(self):
        """Test contraction expansion."""
        text = "I can't do this"
        result = self.cleaner.clean(text)
        self.assertIn("cannot", result)


class TestTokenizer(unittest.TestCase):
    """Test cases for Tokenizer."""
    
    def setUp(self):
        self.tokenizer = Tokenise()
    
    def test_basic_tokenization(self):
        """Test basic tokenization."""
        text = "this is a test"
        tokens = self.tokenizer.tokenise(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
    
    def test_urgency_word_preservation(self):
        """Test that urgency words are preserved."""
        text = "not working now"
        tokens = self.tokenizer.tokenise(text)
        # 'not' and 'now' should not be filtered as stop words
        self.assertIn("not", tokens)
    
    def test_special_tag_preservation(self):
        """Test special tag handling."""
        text = "[Multiple Exclamations] help"
        tokens = self.tokenizer.tokenise(text)
        # Special tag should be preserved as single token
        self.assertTrue(any("[multiple exclamations]" in token.lower() for token in tokens))


if __name__ == '__main__':
    unittest.main()
