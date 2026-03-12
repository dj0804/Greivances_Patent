import spacy
from spacy.symbols import ORTH

class Tokenise:
    def __init__(self):
        nlp=spacy.load("en_core_web_sm")
        urgency_words = {"not", "no", "now", "but", "yet", "never", "only"}
        for word in urgency_words:
            nlp.vocab[word].is_stop = False
        special_tags = ["[Multiple Exclamations]", "[Impatient Questioning]", "[Emphasis]", "[CAPS_ON]"]
        for tag in special_tags:
            self.nlp.tokenizer.add_special_case(tag, [{ORTH: tag}])
    
    def tokenise(self, text: str) -> list[str] :

        doc = self.nlp(text)
        self.tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
        ]
        return self.tokens
    
_shared_tokenizer = None

def Tokens(line):
    if _shared_tokenizer == None:
        _shared_tokenizer=Tokenise()
    
    return _shared_tokenizer.tokenise()