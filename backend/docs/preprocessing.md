# Grievances Patent - Preprocessing Pipeline Documentation

## Overview
This codebase implements a text preprocessing pipeline for grievance analysis, focusing on cleaning, tokenization, and semantic embedding of complaint text data. The system is designed to handle informal text with emotional indicators like excessive punctuation and capitalization.

---

## Project Structure

```
Greivances_Patent/
├── Requirements.txt          # Project dependencies
├── logs/                     # Documentation and logs
├── Src/
│   ├── Preprocessing/        # Text cleaning and tokenization
│   │   ├── CleanText.py
│   │   └── Tokeniser.py
│   ├── Embeddings/           # Semantic vector generation
│   │   └── Semantic_embedder.py
│   ├── Decision_Engine/      # Decision logic (to be implemented)
│   ├── Inference/            # Model inference (to be implemented)
│   ├── Models/               # Model definitions (to be implemented)
│   └── Training/             # Training pipelines (to be implemented)
```

---

## Dependencies

The project relies on the following key libraries:
- **spaCy**: NLP processing and tokenization
- **regex**: Advanced pattern matching
- **contractions**: Expanding contracted words
- **emoji**: Converting emojis to text
- **NLTK**: Natural language toolkit
- **gensim**: FastText word embeddings

### Setup Requirements
Before running, execute: `python -m spacy download en_core_web_sm`

---

## Preprocessing Components

### 1. Text Cleaning (`CleanText.py`)

#### Class: `TextCleaner`

**Purpose**: Standardizes and cleans raw grievance text while preserving emotional indicators that may signal urgency or frustration.

**Key Features**:

- **Capitalization Detection**: Identifies all-caps text (indicating shouting/urgency) and tags it with `[CAPS ON]`
- **Emoji Conversion**: Converts emojis to readable text descriptions using demojization
- **Punctuation Pattern Detection**:
  - Multiple exclamation marks (!!) → `[Multiple Exclamations]`
  - Multiple question marks (??) → `[Impatient Questioning]`
  - Mixed emphasis (!?, ?!, etc.) → `[Emphasis]`
- **Contraction Expansion**: Expands contractions (e.g., "can't" → "cannot")
- **Noise Removal**: Removes special characters (#, $, ^, *, +)
- **Whitespace Normalization**: Collapses multiple spaces into single spaces

**Processing Pipeline**:
```python
1. Caps check → 2. Emoji to text → 3. Punctuation barrage → 
4. Fix contractions → 5. Regex clean → 6. Whitespace clean → 
7. Strip and lowercase
```

**API**:
```python
cleaner = TextCleaner()
cleaned_text = cleaner.clean(raw_text)
# OR
cleaned_text = clean_text_func(raw_text)  # Convenience function
```

---

### 2. Tokenization (`Tokeniser.py`)

#### Class: `Tokenise`

**Purpose**: Converts cleaned text into lemmatized tokens while preserving sentiment-critical words.

**Key Features**:

- **Custom Stop Words**: Preserves urgency indicators that are typically removed:
  - "not", "no", "now", "but", "yet", "never", "only"
- **Special Tag Preservation**: Treats emotional tags as single tokens:
  - `[Multiple Exclamations]`
  - `[Impatient Questioning]`
  - `[Emphasis]`
  - `[CAPS_ON]`
- **Lemmatization**: Reduces words to base forms
- **Stop Word & Punctuation Filtering**: Removes common words and punctuation

**Processing Steps**:
```python
1. Load spaCy model → 2. Configure custom stop words → 
3. Add special cases → 4. Lemmatize and filter tokens
```

**API**:
```python
tokenizer = Tokenise()
tokens = tokenizer.tokenise(cleaned_text)
# OR
tokens = Tokens(line)  # Uses shared tokenizer instance
```

**Note**: There's a bug in the `Tokens()` function - it doesn't pass the `line` parameter to `tokenise()`.

---

### 3. Semantic Embeddings (`Semantic_embedder.py`)

#### Class: `FastTextEncoder`

**Purpose**: Converts tokens into dense vector representations using pre-trained FastText models.

**Key Features**:

- **FastText Model Loading**: Loads Facebook's pre-trained FastText models
- **Token Encoding**: Converts individual tokens to vector representations
- **Embedding Matrix Generation**: Creates weight matrices for Keras Embedding layers
- **Out-of-Vocabulary Handling**: Uses FastText's subword information for unknown words

**API**:
```python
encoder = FastTextEncoder(model_path="path/to/fasttext/model")

# Encode a list of tokens
vectors = encoder.encode_tokens(tokens)

# Create embedding matrix for neural networks
embedding_matrix = encoder.get_embedding_matrix(word_index_dict)
```

---

## Complete Preprocessing Pipeline

### Recommended Usage Flow:

```python
from Src.Preprocessing.CleanText import TextCleaner
from Src.Preprocessing.Tokeniser import Tokenise
from Src.Embeddings.Semantic_embedder import FastTextEncoder

# 1. Initialize components
cleaner = TextCleaner()
tokenizer = Tokenise()
encoder = FastTextEncoder("path/to/model")

# 2. Process text
raw_grievance = "I CAN'T BELIEVE THIS!!! Where's my order????"
cleaned_text = cleaner.clean(raw_grievance)
# → "[caps on] i cannot believe this [multiple exclamations] where is my order [impatient questioning]"

tokens = tokenizer.tokenise(cleaned_text)
# → ['[CAPS_ON]', 'believe', '[Multiple Exclamations]', 'order', '[Impatient Questioning]']

vectors = encoder.encode_tokens(tokens)
# → numpy array of shape (5, vector_size)
```

---

## Design Philosophy

### Emotional Context Preservation
Unlike standard NLP preprocessing that strips all emotional indicators, this pipeline **preserves and tags** them:
- Capitalization patterns
- Punctuation emphasis
- Emoji sentiment

This is critical for grievance analysis where emotional intensity often correlates with:
- Urgency level
- Customer satisfaction
- Required response priority

### Modular Architecture
Each component is independent and reusable:
- `TextCleaner`: Can be used standalone for data exploration
- `Tokenise`: Compatible with any cleaned text
- `FastTextEncoder`: Works with any token list

---

## Known Issues

1. **Tokeniser.py Bug**: The `Tokens()` function doesn't pass the `line` parameter to `tokenise()`
   ```python
   # Current (broken):
   return _shared_tokenizer.tokenise()
   
   # Should be:
   return _shared_tokenizer.tokenise(line)
   ```

2. **Global Variable**: `_shared_tokenizer` uses incorrect comparison (`==` instead of `is`)

3. **Missing NLP Instance**: In `Tokenise.__init__()`, `nlp` is defined locally but `self.nlp` is referenced in `tokenise()` method

---

## Future Components

Based on the project structure, the following modules are planned:
- **Decision_Engine/**: Logic for categorizing and routing grievances
- **Models/**: Neural network architectures for classification/regression
- **Training/**: Model training pipelines
- **Inference/**: Production inference endpoints

---

## Performance Considerations

- **SpaCy Model Load**: One-time cost per tokenizer instance (~100MB memory)
- **FastText Model Load**: Significant startup time and memory (~1-4GB depending on model)
- **Recommendation**: Use singleton patterns or shared instances in production

---

*Documentation generated: February 17, 2026*
