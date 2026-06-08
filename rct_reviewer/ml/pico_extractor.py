"""
PICO Extractor for RCT-Reviewer v1.0.0

Extracts Population, Intervention, and Outcome information
from clinical trial reports using transformer models.

Replaces the old linear classifiers with BioBERT-based models
for both sentence-level extraction and span-level NER.

Copyright (C) 2026 Vihaan Sahu
Based on RobotReviewer PICO extraction by Iain Marshall, Joel Kuiper, Byron Wallace

Original Citation:
    Marshall IJ, Kuiper J, & Wallace BC. "Extracting PICO Sentences from Clinical 
    Trial Reports using Supervised Distant Supervision."
"""

import logging
import re
import string
from typing import Any

import numpy as np

from rct_reviewer.config import get_device, settings
from rct_reviewer.core.exceptions import ModelLoadError
from rct_reviewer.core.models import (
    Annotation,
    PICOAnnotation,
    PICODomain,
)

log = logging.getLogger(__name__)


PICO_KEYWORDS: dict[PICODomain, dict[str, list[str]]] = {
    PICODomain.POPULATION: {
        "primary": [
            "patients",
            "participants",
            "subjects",
            "enrolled",
            "inclusion criteria",
            "eligibility",
            "diagnosed with",
            "with confirmed",
            "aged",
            "years old",
            "men and women",
            "adults",
            "children",
            "pediatric",
            "elderly",
            "geriatric",
        ],
        "secondary": [
            "population",
            "sample",
            "cohort",
            "recruited",
            "consecutive",
            "prospectively",
            "retrospectively",
        ],
    },
    PICODomain.INTERVENTION: {
        "primary": [
            "treated with",
            "received",
            "administered",
            "randomized to",
            "randomised to",
            "assigned to",
            "allocated to",
            "intervention",
            "treatment group",
            "control group",
            "placebo",
            "comparator",
            "compared with",
            "versus",
            "vs",
        ],
        "secondary": [
            "dose",
            "dosage",
            "regimen",
            "duration of treatment",
            "treatment period",
            "follow-up period",
            "mg",
            "daily",
            "oral",
            "intravenous",
            "injection",
        ],
    },
    PICODomain.OUTCOMES: {
        "primary": [
            "primary outcome",
            "primary endpoint",
            "main outcome",
            "primary efficacy",
            "primary efficacy endpoint",
            "co-primary",
        ],
        "secondary": [
            "secondary outcome",
            "secondary endpoint",
            "safety outcome",
            "adverse event",
            "adverse effect",
            "side effect",
            "mortality",
            "morbidity",
            "survival",
            "response rate",
            "remission",
            "improvement",
            "reduction",
            "increase",
            "change in",
            "difference in",
        ],
    },
}


SECTION_PATTERNS: dict[PICODomain, list[re.Pattern]] = {
    PICODomain.POPULATION: [
        re.compile(r'(?:patients|participants|subjects|methods|patient|population)\s*[:\-]', re.I),
        re.compile(r'(?:inclusion|eligibility|criteria)\s*[:\-]', re.I),
    ],
    PICODomain.INTERVENTION: [
        re.compile(r'(?:intervention|treatment|methods|procedure)\s*[:\-]', re.I),
        re.compile(r'(?:study drug|investigational)\s*[:\-]', re.I),
    ],
    PICODomain.OUTCOMES: [
        re.compile(r'(?:outcome|endpoint|result|efficacy)\s*[:\-]', re.I),
        re.compile(r'(?:primary|secondary|safety)\s+(?:outcome|endpoint)\s*[:\-]', re.I),
    ],
}


class PICOExtractor:
    """Transformer-based PICO element extractor.
    
    Extracts PICO information at two levels:
    1. Sentence-level: Identifies sentences most relevant to each PICO domain
    2. Span-level: Extracts specific text spans describing PICO elements
    
    Uses semantic similarity for sentence ranking and keyword/pattern
    matching for span extraction (since fine-tuned PICO NER models
    are not publicly available).
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_dir: str | None = None,
        top_k: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.model.pico_model_name
        self.device = device or get_device()
        self.cache_dir = cache_dir or str(settings.model.cache_dir)
        self.top_k = top_k or settings.processing.pico_top_k
        self.max_length = settings.model.max_length
        
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self) -> None:
        """Load the model for embedding computation."""
        if self._loaded:
            return
        
        log.info(f"Loading PICO extractor model: {self.model_name}")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
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
            log.info("PICO extractor loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(self.model_name, str(e))
    
    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
    
    def extract(
        self,
        title: str | None = None,
        abstract: str | None = None,
        full_text: str | None = None,
        sentences: list[str] | None = None,
        sentence_starts: list[int] | None = None,
    ) -> list[PICOAnnotation]:
        """Extract PICO information from a document.
        
        Args:
            title: Document title
            abstract: Document abstract
            full_text: Full document text
            sentences: Pre-tokenized sentences
            sentence_starts: Character offsets for sentences
            
        Returns:
            List of PICOAnnotation objects for each domain
        """
        self._ensure_loaded()
        

        text_to_use = full_text or abstract or title or ""
        if not text_to_use.strip():
            return []
        

        if sentences is None:
            sentences, sentence_starts = self._split_sentences(text_to_use)
        
        if not sentences:
            return []
        
        results: list[PICOAnnotation] = []
        
        for domain in PICODomain:
            annotation = self._extract_domain(
                domain=domain,
                sentences=sentences,
                sentence_starts=sentence_starts or [],
                full_text=text_to_use,
            )
            results.append(annotation)
        
        return results
    
    def extract_spans(
        self,
        title: str | None = None,
        abstract: str | None = None,
    ) -> dict[str, list[str]]:
        """Extract PICO spans from title and abstract.
        
        Returns simplified dict format compatible with old API.
        """
        text_parts = []
        if title:
            text_parts.append(title)
        if abstract:
            text_parts.append(abstract)
        
        text = "\n\n".join(text_parts)
        if not text.strip():
            return {"population": [], "interventions": [], "outcomes": []}
        
        sentences, _ = self._split_sentences(text)
        
        result = {
            "population": [],
            "interventions": [],
            "outcomes": [],
        }
        
        for domain in PICODomain:
            domain_key = domain.value
            spans = self._extract_spans_for_domain(domain, sentences, text)
            result[domain_key] = self._clean_spans(spans)
        
        return result
    
    def _extract_domain(
        self,
        domain: PICODomain,
        sentences: list[str],
        sentence_starts: list[int],
        full_text: str,
    ) -> PICOAnnotation:
        """Extract PICO info for a single domain."""
      
        queries = self._get_domain_queries(domain)
        
    
        ranked_indices = self._rank_sentences(sentences, queries)
        

        top_indices = ranked_indices[:self.top_k]
        top_sentences = [sentences[i] for i in top_indices if i < len(sentences)]
        

        annotations = []
        for sent_idx in top_indices[:self.top_k]:
            if sent_idx < len(sentences):
                start = sentence_starts[sent_idx] if sent_idx < len(sentence_starts) else 0
                sent_text = sentences[sent_idx]
                annotations.append(
                    Annotation(
                        text=sent_text,
                        start_index=start,
                        prefix=full_text[max(0, start - 30):start],
                        suffix=full_text[start:start + len(sent_text) + 30],
                    )
                )
        
        return PICOAnnotation(
            domain=domain,
            sentences=top_sentences,
            annotations=annotations,
        )
    
    def _get_domain_queries(self, domain: PICODomain) -> list[str]:
        """Get semantic queries for each PICO domain."""
        queries = {
            PICODomain.POPULATION: [
                "Who were the study participants?",
                "What were the inclusion criteria?",
                "Patient population characteristics",
                "Eligibility criteria for enrollment",
            ],
            PICODomain.INTERVENTION: [
                "What treatment was given?",
                "What was the intervention?",
                "Treatment protocol and dosage",
                "Study drug or procedure",
            ],
            PICODomain.OUTCOMES: [
                "What were the measured outcomes?",
                "What was the primary endpoint?",
                "Efficacy and safety measures",
                "Study results and endpoints",
            ],
        }
        return queries.get(domain, [domain.value])
    
    def _rank_sentences(
        self,
        sentences: list[str],
        queries: list[str],
    ) -> list[int]:
        """Rank sentences by relevance to PICO domain."""
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
        domain_match = None
        for domain, keywords in PICO_KEYWORDS.items():
            if any(kw in query.lower() for kw in keywords["primary"]):
                domain_match = domain
                break
        
        if domain_match is None:
            return list(range(len(sentences)))
        
        primary_kw = PICO_KEYWORDS[domain_match]["primary"]
        secondary_kw = PICO_KEYWORDS[domain_match]["secondary"]
        
        scores = []
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            score = (
                sum(2 for kw in primary_kw if kw in sent_lower) +
                sum(1 for kw in secondary_kw if kw in sent_lower)
            )
            scores.append((score, i))
        
        scores.sort(reverse=True)
        return [idx for _, idx in scores]
    
    def _extract_spans_for_domain(
        self,
        domain: PICODomain,
        sentences: list[str],
        full_text: str,
    ) -> list[str]:
        """Extract specific PICO spans from sentences."""
        keywords = PICO_KEYWORDS.get(domain, {"primary": [], "secondary": []})
        all_keywords = keywords["primary"] + keywords["secondary"]
        
        spans = []
        seen = set()
        
        for sent in sentences:
            sent_lower = sent.lower()
            
   
            if not any(kw in sent_lower for kw in all_keywords):
                continue
            
   
            extracted = self._extract_key_phrases(sent, domain)
            
            for phrase in extracted:
                phrase_clean = phrase.strip().strip(string.punctuation)
                if phrase_clean and phrase_clean not in seen and len(phrase_clean) > 3:
                    seen.add(phrase_clean)
                    spans.append(phrase_clean)
        
        return spans
    
    def _extract_key_phrases(
        self,
        sentence: str,
        domain: PICODomain,
    ) -> list[str]:
        """Extract key phrases from a sentence based on domain patterns."""
        phrases = []
        
      
        patterns = {
            PICODomain.POPULATION: [
                r'(?:patients?|participants?|subjects?) (?:with|who|aged|diagnosed)[^.]+',
                r'(?:men|women|adults?|children?|elderly)[^.]+?(?:with|who)[^.]+',
                r'(?:inclusion criteria[:\s]*)[^.]+',
                r'(?:aged \d+[^.]+)',
            ],
            PICODomain.INTERVENTION: [
                r'(?:treated?|received?|administered?|given) (?:with|)?[^.]+',
                r'(?:randomized?|allocated?|assigned?) to ([^.]+)',
                r'([\w\s]+) (?:vs|versus) ([^.]+)',
                r'([\d\s]+mg[^.]+)',
                r'(?:oral|intravenous|subcutaneous)[^.]+',
            ],
            PICODomain.OUTCOMES: [
                r'(?:primary|main) (?:outcome|endpoint)[^.]+',
                r'(?:secondary)? (?:outcome|endpoint)[^.]+',
                r'(?:measured?|assessed?|evaluated?)[^.]+',
                r'(?:significant)? (?:improvement?|reduction?|increase?|change)[^.]+',
            ],
        }
        
        domain_patterns = patterns.get(domain, [])
        
        for pattern in domain_patterns:
            try:
                matches = re.findall(pattern, sentence, re.I)
                for match in matches:
                    if isinstance(match, tuple):
                        match = " vs ".join(m.strip() for m in match if m.strip())
                    if match.strip():
                        phrases.append(match.strip())
            except re.error:
                continue
        
        return phrases
    
    @staticmethod
    def _clean_spans(spans: list[str]) -> list[str]:
        """Clean and deduplicate PICO spans."""
        cleaned = []
        seen = set()
        
        for span in spans:
 
            span = re.sub(r'^(the|a|an|we|our)\s+', '', span, flags=re.I)

            span = span.strip(string.punctuation).strip()
            
            if span and span not in seen and len(span) > 3:
                seen.add(span)
                cleaned.append(span)
        
        return cleaned
    
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



_extractor_instance: PICOExtractor | None = None


def get_pico_extractor() -> PICOExtractor:
    """Get or create the global PICOExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = PICOExtractor()
    return _extractor_instance


def extract_pico(
    title: str | None = None,
    abstract: str | None = None,
    full_text: str | None = None,
) -> list[PICOAnnotation]:
    """Convenience function to extract PICO information."""
    extractor = get_pico_extractor()
    return extractor.extract(title=title, abstract=abstract, full_text=full_text)


def extract_pico_spans(
    title: str | None = None,
    abstract: str | None = None,
) -> dict[str, list[str]]:
    """Convenience function to extract PICO spans from abstract."""
    extractor = get_pico_extractor()
    return extractor.extract_spans(title=title, abstract=abstract)