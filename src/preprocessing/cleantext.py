import spacy
import regex as re
import emoji as ej
import contractions 

# run this command before running this file
# python -m spacy download en_core_web_sm

class TextCleaner:
    def __init__(self):
        self.multiple_excl = re.compile(r"!{2,}")
        self.impatient_ques = re.compile(r"\?{2,}")
        self.emphasis = re.compile(r"[!?]{2,}")
        self.noise_chars = re.compile(r"[#$^*+]")
        self.extra_whitespace = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        if not text:
            return ""
        
        text = self._caps_check(text)
        text = self._emoji_to_text(text)
        text = self._punctuation_barrage(text)
        text = self._fix_contractions(text)
        text = self._regex_clean(text)
        text = self._whitespace_clean(text)
        
        return text.strip().lower()
    
    def _caps_check(self,text):
        if text.isupper() and len(text)>3:
            return f"[CAPS ON] {text}" 

    def _emoji_to_text(self,text):
        return ej.demojize(text,delimiters=('[',']'))

    def _punctuation_barrage(self,text):
        text = self.multiple_excl.sub(" [Multiple Exclamations] ", text)
        text = self.impatient_ques.sub(" [Impatient Questioning] ", text)
        text = self.emphasis.sub(" [Emphasis] ", text)
        return text
    
    def _fix_contractions(self, text):
        return contractions.fix(text)

    def _regex_clean(self,text):
        return self.noise_chars.sub("", text)

    def _whitespace_clean(self,text):
        return self.extra_whitespace.sub(" ", text)

def clean_text_func(text):
    cleaner = TextCleaner()
    return cleaner.clean(text)