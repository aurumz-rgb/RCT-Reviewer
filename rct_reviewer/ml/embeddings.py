"""
Embedding utilities for RCT-Reviewer v1.0.0

Provides a unified interface for computing embeddings using
HuggingFace transformer models. Replaces the old bert-as-service
dependency.

Copyright (C) 2024-2026 Vihaan Sahu
"""

import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rct_reviewer.config import get_device, settings
from rct_reviewer.core.exceptions import ModelLoadError

log = logging.getLogger(__name__)


class Embedder:
    """HuggingFace-based text embedder.
    
    Computes sentence/document embeddings using transformer models.
    Uses mean pooling over token embeddings (excluding [CLS] and [PAD]).
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_dir: str | None = None,
        max_length: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name or settings.model.ner_model_name
        self.device = device or get_device()
        self.cache_dir = cache_dir or str(settings.model.cache_dir)
        self.max_length = max_length or settings.model.max_length
        self.normalize = normalize
        
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return
        
        log.info(f"Loading embedder: {self.model_name}")
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            self._model.to(self.device)
            self._model.eval()
            
            self._loaded = True
            log.info("Embedder loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(self.model_name, str(e))
    
    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
    
    def encode(
        self,
        texts: str | list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        self._ensure_loaded()
        
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        batch_size = batch_size or settings.model.batch_size
        all_embeddings = []
        
        if show_progress:
            from tqdm import tqdm
            batches = tqdm(range(0, len(texts), batch_size), desc="Encoding")
        else:
            batches = range(0, len(texts), batch_size)
        
        for i in batches:
            batch = texts[i:i + batch_size]
            embeddings = self._encode_batch(batch)
            all_embeddings.append(embeddings)
        
        result = np.vstack(all_embeddings)
        
        if self.normalize:
            result = result / (np.linalg.norm(result, axis=1, keepdims=True) + 1e-8)
        
        return result
    
    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts."""
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
        
    
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        
 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        return mean_embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text, returning 1D array."""
        result = self.encode(text)
        return result[0]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        

        return float(np.dot(emb1, emb2))



_embedder_instances: dict[str, Embedder] = {}


def get_embedder(model_name: str | None = None) -> Embedder:
    """Get or create an Embedder instance."""
    model_name = model_name or settings.model.ner_model_name
    if model_name not in _embedder_instances:
        _embedder_instances[model_name] = Embedder(model_name=model_name)
    return _embedder_instances[model_name]


def encode_texts(
    texts: str | list[str],
    model_name: str | None = None,
) -> np.ndarray:
    """Convenience function to encode texts."""
    embedder = get_embedder(model_name)
    return embedder.encode(texts)


def compute_similarity(text1: str, text2: str) -> float:
    """Convenience function to compute similarity."""
    embedder = get_embedder()
    return embedder.similarity(text1, text2)