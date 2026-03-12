import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

class ComplaintSummarizer:
    """Summarizer for grievance complaints."""
    
    def __init__(self):
        self._summarizer = None
        self._model_loaded = False
        
    def _load_model(self):
        """Lazy load the summarization pipeline."""
        from src.config import SUMMARIZER_MODEL_NAME, ENABLE_SUMMARIZATION
        
        if not ENABLE_SUMMARIZATION:
            return
            
        if not self._model_loaded:
            logger.info(f"Loading summarization model: {SUMMARIZER_MODEL_NAME}...")
            start_time = time.time()
            try:
                from transformers import pipeline
                self._summarizer = pipeline("summarization", model=SUMMARIZER_MODEL_NAME)
                self._model_loaded = True
                load_time = time.time() - start_time
                logger.info(f"Summarization model loaded successfully in {load_time:.2f}s.")
            except Exception as e:
                logger.error(f"Failed to load summarization model '{SUMMARIZER_MODEL_NAME}': {e}")
                # Fallback to facebook/bart-large-cnn if permitted
                try:
                    logger.info("Attempting fallback to 'facebook/bart-large-cnn'...")
                    from transformers import pipeline
                    self._summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                    self._model_loaded = True
                    logger.info("Fallback model loaded successfully.")
                except Exception as fallback_e:
                    logger.error(f"Fallback model failed: {fallback_e}")
                    self._summarizer = None
                    self._model_loaded = True # mark as loaded to prevent constant retries
    
    def summarize(self, text: str) -> Optional[str]:
        """
        Generate a summary of the input text.
        Returns the summary string, or None if summarization fails or is disabled.
        """
        from src.config import ENABLE_SUMMARIZATION, MAX_SUMMARY_LENGTH, MIN_SUMMARY_LENGTH
        
        if not ENABLE_SUMMARIZATION:
            return None
            
        if not text or len(text.strip()) < 10:
            return None
            
        try:
            self._load_model()
            
            if self._summarizer is None:
                return None
                
            input_words = len(text.split())
            # For very short texts, use it as the summary
            if input_words < MIN_SUMMARY_LENGTH:
                return text.strip()
                
            max_len = min(MAX_SUMMARY_LENGTH, int(input_words * 0.8))
            max_len = max(max_len, MIN_SUMMARY_LENGTH + 5)
            
            # Ensure max_length > min_length
            if max_len <= MIN_SUMMARY_LENGTH:
                max_len = MIN_SUMMARY_LENGTH + 1
                
            result = self._summarizer(
                text, 
                max_length=max_len, 
                min_length=min(MIN_SUMMARY_LENGTH, max_len - 1), 
                do_sample=False
            )
            
            if result and isinstance(result, list) and len(result) > 0:
                summary = result[0].get("summary_text", "")
                return summary.strip()
                
            return None
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return None

# Singleton instance
summarizer_instance = ComplaintSummarizer()
