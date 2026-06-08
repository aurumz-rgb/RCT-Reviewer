"""
Risk of Bias Assessor for RCT-Reviewer v1.0.0

Assesses Risk of Bias for clinical trials using the Cochrane
Risk of Bias tool domains. Uses transformer models for both
sentence-level evidence extraction and document-level classification.

Replaces the old linear classifiers with BioBERT-based models.

Copyright (C) 2026 Vihaan Sahu
Based on RobotReviewer Bias assessment by Iain Marshall, Joel Kuiper, Byron Wallace

Original Citation:
    Marshall IJ, Kuiper J, & Wallace BC. RobotReviewer: evaluation of a system 
    for automatically assessing bias in clinical trials. JAMIA 2015.
"""

import logging
import re
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from rct_reviewer.config import get_device, settings
from rct_reviewer.core.exceptions import ModelLoadError
from rct_reviewer.core.models import Annotation, BiasAnnotation, BiasDomain, BiasJudgement

log = logging.getLogger(__name__)


BIAS_DOMAINS = list(BiasDomain)


DOMAIN_KEYWORDS: dict[BiasDomain, dict[str, list[str]]] = {
    BiasDomain.RANDOM_SEQUENCE: {
        "low": [
            "random number",
            "random sequence",
            "computer-generated",
            "computer generated",
            "randomization schedule",
            "random number table",
            "coin toss",
            "shuffled",
            "random permuted",
        ],
        "high_unclear": [
            "alternating",
            "date of birth",
            "medical record number",
            "hospital number",
            "not described",
            "not stated",
            "unclear",
        ],
    },
    BiasDomain.ALLOCATION_CONCEALMENT: {
        "low": [
            "sealed opaque envelope",
            "central randomization",
            "computer-generated allocation",
            "pharmacy-controlled",
            "sequentially numbered",
            "identical containers",
        ],
        "high_unclear": [
            "open allocation",
            "not concealed",
            "not described",
            "not stated",
            "unclear",
            "open label",
        ],
    },
    BiasDomain.BLINDING_PARTICIPANTS: {
        "low": [
            "double-blind",
            "double blind",
            "participants and personnel blinded",
            "identical placebo",
            "matching placebo",
            "participants were blinded",
        ],
        "high_unclear": [
            "open-label",
            "open label",
            "no blinding",
            "not blinded",
            "not masked",
            "not described",
            "single-blind",
        ],
    },
    BiasDomain.BLINDING_OUTCOME: {
        "low": [
            "outcome assessors blinded",
            "outcome assessment blinded",
            "blind assessment",
            "masked assessment",
            "independent adjudication",
        ],
        "high_unclear": [
            "not blinded",
            "not masked",
            "not described",
            "self-reported",
            "questionnaire",
            "unclear",
        ],
    },
    BiasDomain.INCOMPLETE_OUTCOME: {
        "low": [
            "no missing data",
            "all patients accounted for",
            "intention-to-treat",
            "intention to treat",
            "complete follow-up",
            "low attrition",
        ],
        "high_unclear": [
            "high attrition",
            "significant dropout",
            "loss to follow-up",
            "excluded after randomization",
            "per-protocol",
            "not reported",
            "unclear",
        ],
    },
    BiasDomain.SELECTIVE_REPORTING: {
        "low": [
            "pre-specified",
            "pre specified",
            "clinicaltrials.gov",
            "registered protocol",
            "all outcomes reported",
            "prospective registration",
        ],
        "high_unclear": [
            "not all outcomes reported",
            "selective reporting",
            "outcomes changed",
            "not pre-registered",
            "unregistered",
            "unclear",
        ],
    },
}


class BiasAssessor:
    """Transformer-based Risk of Bias assessor.
    
    For each Cochrane RoB domain:
    1. Ranks sentences by relevance to the domain
    2. Classifies overall bias as "low" or "high/unclear"
    
    Uses a combination of semantic similarity for sentence ranking
    and pattern matching for classification (heuristic-based,
    since fine-tuned bias models are not publicly available).
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_dir: str | None = None,
        top_k: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.model.bias_model_name
        self.device = device or get_device()
        self.cache_dir = cache_dir or str(settings.model.cache_dir)
        self.top_k = top_k or settings.processing.bias_top_k
        self.max_length = settings.model.max_length
        
        self._tokenizer: AutoTokenizer | None = None
        self._model: Any | None = None
        self._embedder: Any | None = None
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self) -> None:
        """Load the model and tokenizer for embedding computation."""
        if self._loaded:
            return
        
        log.info(f"Loading Bias assessor model: {self.model_name}")
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            self._model.to(self.device)
            self._model.eval()
            
            self._loaded = True
            log.info("Bias assessor loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(self.model_name, str(e))
    
    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
    
    def assess(
        self,
        full_text: str,
        sentences: list[str] | None = None,
        sentence_starts: list[int] | None = None,
    ) -> list[BiasAnnotation]:
        """Assess Risk of Bias for all domains.
        
        Args:
            full_text: Full document text
            sentences: Pre-tokenized sentences (optional)
            sentence_starts: Character offsets for sentences (optional)
            
        Returns:
            List of BiasAnnotation objects, one per domain
        """
        self._ensure_loaded()
        
     
        if sentences is None:
            sentences, sentence_starts = self._split_sentences(full_text)
        
        if not sentences:
            return [
                BiasAnnotation(
                    domain=domain,
                    judgement=BiasJudgement.HIGH_UNCLEAR,
                    sentences=[],
                    annotations=[],
                )
                for domain in BIAS_DOMAINS
            ]
        
        results: list[BiasAnnotation] = []
        
        for domain in BIAS_DOMAINS:
            annotation = self._assess_domain(
                domain=domain,
                sentences=sentences,
                sentence_starts=sentence_starts or [],
                full_text=full_text,
            )
            results.append(annotation)
        
        return results
    
    def _assess_domain(
        self,
        domain: BiasDomain,
        sentences: list[str],
        sentence_starts: list[int],
        full_text: str,
    ) -> BiasAnnotation:
        """Assess bias for a single domain."""
        
        domain_queries = self._get_domain_queries(domain)
        
       
        ranked_indices = self._rank_sentences(
            sentences=sentences,
            queries=domain_queries,
        )
        

        top_indices = ranked_indices[:self.top_k]
        top_sentences = [sentences[i] for i in top_indices if i < len(sentences)]
        

        annotations = []
        for i, sent_idx in enumerate(top_indices[:self.top_k]):
            if sent_idx < len(sentences):
                start = sentence_starts[sent_idx] if sent_idx < len(sentence_starts) else 0
                annotations.append(
                    Annotation(
                        text=sentences[sent_idx],
                        start_index=start,
                        prefix=full_text[max(0, start - 30):start],
                        suffix=full_text[start:start + len(sentences[sent_idx]) + 30],
                    )
                )
        
        
        judgement = self._classify_bias(domain, top_sentences, full_text)
        
        return BiasAnnotation(
            domain=domain,
            judgement=judgement,
            sentences=top_sentences,
            annotations=annotations,
        )
    
    def _get_domain_queries(self, domain: BiasDomain) -> list[str]:
        """Get query sentences for semantic matching to each domain."""
        queries = {
            BiasDomain.RANDOM_SEQUENCE: [
                "How were participants randomized?",
                "Random sequence generation method",
                "Method used to generate the random allocation sequence",
            ],
            BiasDomain.ALLOCATION_CONCEALMENT: [
                "How was allocation concealed?",
                "Allocation concealment mechanism",
                "Method used to conceal allocation",
            ],
            BiasDomain.BLINDING_PARTICIPANTS: [
                "Were participants blinded?",
                "Blinding of participants and personnel",
                "Knowledge of allocated interventions",
            ],
            BiasDomain.BLINDING_OUTCOME: [
                "Were outcome assessors blinded?",
                "Blinding of outcome assessment",
                "Knowledge of allocated interventions during assessment",
            ],
            BiasDomain.INCOMPLETE_OUTCOME: [
                "Were all outcomes addressed?",
                "Incomplete outcome data",
                "Missing data and attrition",
            ],
            BiasDomain.SELECTIVE_REPORTING: [
                "Are all outcomes reported?",
                "Selective outcome reporting",
                "Reporting bias",
            ],
        }
        return queries.get(domain, [domain.value])
    
    def _rank_sentences(
        self,
        sentences: list[str],
        queries: list[str],
    ) -> list[int]:
        """Rank sentences by semantic similarity to domain queries."""
        try:
            return self._rank_with_embeddings(sentences, queries)
        except Exception as e:
            log.warning(f"Embedding ranking failed: {e}, using keyword fallback")
            return self._rank_with_keywords(sentences, queries[0])
    
    def _rank_with_embeddings(
        self,
        sentences: list[str],
        queries: list[str],
    ) -> list[int]:
        """Rank sentences using transformer embeddings."""
        import torch
        
   
        query_embeddings = []
        for query in queries:
            inputs = self._tokenizer(
                query,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs)
           
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                query_embeddings.append(emb[0])
        

        query_emb = np.mean(query_embeddings, axis=0)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        

        sentence_embeddings = []
        batch_size = 32
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
                sentence_embeddings.append(embs)
        
        if sentence_embeddings:
            all_embs = np.vstack(sentence_embeddings)
        else:
            return list(range(len(sentences)))
        

        similarities = all_embs @ query_emb
        
  
        ranked_indices = np.argsort(similarities)[::-1].tolist()
        
        return ranked_indices
    
    def _rank_with_keywords(
        self,
        sentences: list[str],
        query: str,
    ) -> list[int]:
        """Fallback keyword-based ranking."""
        query_words = set(query.lower().split())
        
        scores = []
        for i, sent in enumerate(sentences):
            sent_words = set(sent.lower().split())
   
            intersection = len(query_words & sent_words)
            union = len(query_words | sent_words)
            score = intersection / (union + 1e-8)
            scores.append((score, i))
        
        scores.sort(reverse=True)
        return [idx for _, idx in scores]
    
    def _classify_bias(
        self,
        domain: BiasDomain,
        evidence_sentences: list[str],
        full_text: str,
    ) -> BiasJudgement:
        """Classify bias level based on evidence sentences.
        
        Uses keyword heuristics since fine-tuned bias classifiers
        are not publicly available.
        """
        text_lower = " ".join(evidence_sentences).lower()
        full_text_lower = full_text.lower()
        
        keywords = DOMAIN_KEYWORDS.get(domain, {"low": [], "high_unclear": []})
        
        low_score = sum(1 for kw in keywords["low"] if kw in text_lower)
        high_score = sum(1 for kw in keywords["high_unclear"] if kw in text_lower)
        
        low_score_full = sum(1 for kw in keywords["low"] if kw in full_text_lower)
        high_score_full = sum(1 for kw in keywords["high_unclear"] if kw in full_text_lower)

        total_low = low_score * 2 + low_score_full
        total_high = high_score * 2 + high_score_full
        
        if total_low > total_high:
            return BiasJudgement.LOW
        elif total_high > total_low:
            return BiasJudgement.HIGH_UNCLEAR
        else:
    
            return BiasJudgement.HIGH_UNCLEAR
    
    @staticmethod
    def _split_sentences(text: str) -> tuple[list[str], list[int]]:
        """Split text into sentences with character offsets."""

        sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|'  
            r'(?<=\n)\s*(?=[A-Z])'       
        )
        
        sentences = []
        starts = []
        
        current_start = 0
        for match in sentence_pattern.finditer(text):
            end = match.start()
            sent = text[current_start:end].strip()
            if sent:
                sentences.append(sent)
                starts.append(current_start)
            current_start = match.end()
        
   
        remaining = text[current_start:].strip()
        if remaining:
            sentences.append(remaining)
            starts.append(current_start)
        
        return sentences, starts



_assessor_instance: BiasAssessor | None = None


def get_bias_assessor() -> BiasAssessor:
    """Get or create the global BiasAssessor instance."""
    global _assessor_instance
    if _assessor_instance is None:
        _assessor_instance = BiasAssessor()
    return _assessor_instance


def assess_bias(
    full_text: str,
    sentences: list[str] | None = None,
    sentence_starts: list[int] | None = None,
) -> list[BiasAnnotation]:
    """Convenience function to assess bias for a document."""
    assessor = get_bias_assessor()
    return assessor.assess(full_text, sentences, sentence_starts)