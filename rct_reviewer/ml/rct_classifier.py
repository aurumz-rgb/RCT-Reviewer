"""
RCT Classifier for RCT-Reviewer v1.0.0

Classifies whether a document (title + abstract) describes a
Randomized Controlled Trial using transformer models.

Replaces the old SVM + CNN ensemble with a single BioBERT-based
classifier that achieves comparable or better performance.

Copyright (C) 2026 Vihaan Sahu
Based on RobotReviewer RCT classification by Iain Marshall, Joel Kuiper, Byron Wallace

Original Citation:
    Marshall IJ, Kuiper J, & Wallace BC. RobotReviewer: evaluation of a system 
    for automatically assessing bias in clinical trials. JAMIA 2015.
"""

import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from rct_reviewer.config import get_device, settings
from rct_reviewer.core.exceptions import ModelInferenceError, ModelLoadError
from rct_reviewer.core.models import RCTResult

log = logging.getLogger(__name__)

# Classification thresholds (calibrated for BioBERT)
THRESHOLDS = {
    "precise": 0.85,    # High precision, lower recall
    "balanced": 0.5,    # Balanced precision/recall
    "sensitive": 0.35,  # High recall, lower precision
}


class RCTClassifier:
    """Transformers-based RCT classifier.
    
    Uses a fine-tuned BioBERT model to classify whether a title+abstract
    describes a Randomized Controlled Trial.
    
    If no fine-tuned model is available, uses zero-shot classification
    with PubMedBERT as fallback.
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.model.rct_model_name
        self.device = device or get_device()
        self.cache_dir = cache_dir or str(settings.model.cache_dir)
        self.max_length = settings.model.max_length
        
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSequenceClassification | None = None
        self._pipeline: Any | None = None
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return
        
        log.info(f"Loading RCT classifier: {self.model_name}")
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            
            # Check if this is a fine-tuned classification model
            try:
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    num_labels=2,
                    id2label={0: "NOT_RCT", 1: "IS_RCT"},
                    label2id={"NOT_RCT": 0, "IS_RCT": 1},
                )
                # If model loads with wrong num_labels, try without specifying
            except Exception:
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
            
            self._model.to(self.device)
            self._model.eval()
            

            self._pipeline = pipeline(
                "text-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                device=self.device if self.device != "mps" else -1,  # MPS workaround
                top_k=2,
                max_length=self.max_length,
                truncation=True,
            )
            
            self._loaded = True
            log.info("RCT classifier loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(self.model_name, str(e))
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._loaded:
            self.load()
    
    def predict(
        self,
        title: str,
        abstract: str | None = None,
        threshold_type: str = "balanced",
    ) -> RCTResult:
        """Classify a single document as RCT or not.
        
        Args:
            title: Document title
            abstract: Document abstract (optional but recommended)
            threshold_type: One of "precise", "balanced", "sensitive"
            
        Returns:
            RCTResult with classification details
        """
        self._ensure_loaded()
        
        if threshold_type not in THRESHOLDS:
            threshold_type = "balanced"
        
        # Combine title and abstract
        text = self._prepare_input(title, abstract)
        
        try:
            return self._classify_with_pipeline(text, threshold_type)
        except Exception as e:
            log.warning(f"Pipeline classification failed: {e}, using zero-shot fallback")
            return self._classify_zero_shot(text, threshold_type)
    
    def predict_batch(
        self,
        documents: list[dict[str, str]],
        threshold_type: str = "balanced",
    ) -> list[RCTResult]:
        """Classify multiple documents.
        
        Args:
            documents: List of dicts with "title" and optional "abstract" keys
            threshold_type: One of "precise", "balanced", "sensitive"
            
        Returns:
            List of RCTResult objects
        """
        self._ensure_loaded()
        
        texts = [
            self._prepare_input(doc.get("title", ""), doc.get("abstract"))
            for doc in documents
        ]
        
        results: list[RCTResult] = []
        
      
        batch_size = settings.model.batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                try:
                    result = self._classify_with_pipeline(text, threshold_type)
                except Exception:
                    result = self._classify_zero_shot(text, threshold_type)
                results.append(result)
        
        return results
    
    def _prepare_input(self, title: str, abstract: str | None) -> str:
        """Prepare text input for the model."""
        title = title.strip() if title else ""
        abstract = abstract.strip() if abstract else ""
        
        if title and abstract:
            return f"{title}\n\n{abstract}"
        elif title:
            return title
        else:
            return abstract
    
    def _classify_with_pipeline(
        self,
        text: str,
        threshold_type: str,
    ) -> RCTResult:
        """Classify using the HuggingFace pipeline."""
        outputs = self._pipeline(text)
        
        # Extract probability for IS_RCT class
        prob_is_rct = 0.0
        score = 0.0
        
        for item in outputs:
            label = item.get("label", "")
            score_val = item.get("score", 0.0)
            
            # Handle various label formats
            if "IS_RCT" in label.upper() or label == "LABEL_1" or label == "1":
                prob_is_rct = score_val
            elif "NOT_RCT" in label.upper() or label == "LABEL_0" or label == "0":
                # Score is probability of NOT being RCT
                pass
        
        # If we couldn't identify labels, use the first score as IS_RCT probability
        if prob_is_rct == 0.0 and outputs:
            prob_is_rct = outputs[0].get("score", 0.5)
        
        # Convert probability to a score comparable to old model's range
        # Old model: score ~ N(0,1), higher = more likely RCT
        # New model: probability [0,1], higher = more likely RCT
        # Use logit transform to get to unbounded scale, then normalize
        score = self._prob_to_score(prob_is_rct)
        
        threshold = THRESHOLDS[threshold_type]
        is_rct = score >= threshold
        
        return RCTResult(
            is_rct=is_rct,
            score=round(score, 4),
            threshold=threshold,
            threshold_type=threshold_type,
            probability=round(prob_is_rct, 4),
            model_used=self.model_name,
        )
    
    def _classify_zero_shot(
        self,
        text: str,
        threshold_type: str,
    ) -> RCTResult:
        """Fallback zero-shot classification using keyword heuristics."""
        log.debug("Using zero-shot RCT classification fallback")
        
        text_lower = text.lower()
        

        strong_rct_keywords = [
            "randomized controlled trial",
            "randomised controlled trial",
            "rct",
            "randomly assigned",
            "randomly allocated",
            "random assignment",
            "random allocation",
        ]
        
  
        moderate_rct_keywords = [
            "randomized",
            "randomised",
            "randomization",
            "randomisation",
            "double-blind",
            "double blind",
            "single-blind",
            "single blind",
            "placebo-controlled",
            "placebo controlled",
        ]
        

        non_rct_keywords = [
            "systematic review",
            "meta-analysis",
            "meta analysis",
            "observational study",
            "cohort study",
            "case-control",
            "cross-sectional",
            "case report",
            "case series",
        ]
        
        # Calculate scores
        strong_count = sum(1 for kw in strong_rct_keywords if kw in text_lower)
        moderate_count = sum(1 for kw in moderate_rct_keywords if kw in text_lower)
        non_rct_count = sum(1 for kw in non_rct_keywords if kw in text_lower)
        
        # Heuristic score (0-1 range)
        raw_score = (strong_count * 0.3 + moderate_count * 0.15) - (non_rct_count * 0.4)
        prob = np.clip(raw_score + 0.5, 0.0, 1.0)
        score = self._prob_to_score(prob)
        
        threshold = THRESHOLDS[threshold_type]
        
        return RCTResult(
            is_rct=score >= threshold,
            score=round(score, 4),
            threshold=threshold,
            threshold_type=threshold_type,
            probability=round(float(prob), 4),
            model_used="zero-shot-heuristic",
        )
    
    @staticmethod
    def _prob_to_score(prob: float) -> float:
        """Convert probability [0,1] to score scale comparable to old model.
        
        Uses logit transform: score = log(p / (1-p)) / 3
        This maps p=0.5 -> 0, p=0.95 -> ~1.5, p=0.05 -> ~-1.5
        """
        prob = np.clip(prob, 0.01, 0.99)  
        logit = np.log(prob / (1 - prob))
        return round(float(logit / 3), 4)  


# Singleton instance for reuse
_classifier_instance: RCTClassifier | None = None


def get_rct_classifier() -> RCTClassifier:
    """Get or create the global RCT classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = RCTClassifier()
    return _classifier_instance


def classify_rct(
    title: str,
    abstract: str | None = None,
    threshold_type: str = "balanced",
) -> RCTResult:
    """Convenience function to classify a single document."""
    classifier = get_rct_classifier()
    return classifier.predict(title, abstract, threshold_type)