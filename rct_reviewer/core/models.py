"""
Pydantic data models for RCT-Reviewer v1.0.0.

Replaces the old MultiDict with type-safe Pydantic models.

Copyright (C) 2026 Vihaan Sahu
Based on RobotReviewer data structures by Iain Marshall, Joel Kuiper, Byron Wallace
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BiasDomain(str, Enum):
    """Cochrane Risk of Bias domains."""
    RANDOM_SEQUENCE = "Random sequence generation"
    ALLOCATION_CONCEALMENT = "Allocation concealment"
    BLINDING_PARTICIPANTS = "Blinding of participants and personnel"
    BLINDING_OUTCOME = "Blinding of outcome assessment"
    INCOMPLETE_OUTCOME = "Incomplete outcome data"
    SELECTIVE_REPORTING = "Selective reporting"


class BiasJudgement(str, Enum):
    """Risk of Bias judgement levels."""
    LOW = "low"
    HIGH_UNCLEAR = "high/unclear"


class PICODomain(str, Enum):
    """PICO element types."""
    POPULATION = "population"
    INTERVENTION = "interventions"
    OUTCOMES = "outcomes"


class Annotation(BaseModel):
    """A text annotation with position information."""
    text: str
    start_index: int
    end_index: int | None = None
    prefix: str | None = None
    suffix: str | None = None
    confidence: float | None = None
    uuid: UUID = Field(default_factory=uuid4)


class PICOAnnotation(BaseModel):
    """PICO extraction result for one domain."""
    domain: PICODomain
    sentences: list[str] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    embeddings: list[list[float]] | None = None


class BiasAnnotation(BaseModel):
    """Risk of Bias assessment for one domain."""
    domain: BiasDomain
    judgement: BiasJudgement
    sentences: list[str] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    confidence: float | None = None


class RCTResult(BaseModel):
    """RCT classification result."""
    is_rct: bool
    score: float
    threshold: float
    threshold_type: str = "balanced"
    probability: float | None = None
    model_used: str = "BiomedNLP-PubMedBERT"
    
    @property
    def is_rct_precise(self) -> bool:
        """More strict RCT classification."""
        return self.score >= self.threshold * 1.2
    
    @property
    def is_rct_sensitive(self) -> bool:
        """More lenient RCT classification."""
        return self.score >= self.threshold * 0.8


class Author(BaseModel):
    """Publication author information."""
    initials: str = ""
    forename: str = ""
    lastname: str = ""
    orcid: str | None = None


class PublicationInfo(BaseModel):
    """Publication metadata extracted from document."""
    title: str | None = None
    abstract: str | None = None
    journal: str | None = None
    year: int | None = None
    month: int | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    authors: list[Author] = Field(default_factory=list)
    doi: str | None = None
    pmid: str | None = None
    trial_registry_id: str | None = None


class SampleSizeInfo(BaseModel):
    """Extracted sample size information."""
    total: int | None = None
    per_arm: dict[str, int] = Field(default_factory=dict)
    sentences: list[str] = Field(default_factory=list)


class PunchlineInfo(BaseModel):
    """Main conclusion/punchline extraction."""
    sentences: list[str] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)


class DocumentAnalysis(BaseModel):
    """Complete analysis result for one document.
    
    This replaces the old MultiDict structure with a type-safe
    Pydantic model that clearly separates source data from
    ML-generated annotations.
    """
    document_id: UUID = Field(default_factory=uuid4)
    filename: str
    file_hash: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Source data (from PDF parsing)
    full_text: str | None = None
    publication: PublicationInfo = Field(default_factory=PublicationInfo)
    parse_error: bool = False
    
    # ML results
    rct: RCTResult | None = None
    pico: list[PICOAnnotation] = Field(default_factory=list)
    pico_spans: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "population": [],
            "interventions": [],
            "outcomes": [],
        }
    )
    bias: list[BiasAnnotation] = Field(default_factory=list)
    sample_size: SampleSizeInfo | None = None
    punchline: PunchlineInfo | None = None
    
    # Processing metadata
    processing_time_seconds: float | None = None
    models_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    
    def to_summary_dict(self) -> dict[str, Any]:
        """Create a summary dictionary for table display."""
        return {
            "filename": self.filename,
            "title": self.publication.title or "N/A",
            "is_rct": self.rct.is_rct if self.rct else None,
            "rct_score": round(self.rct.score, 4) if self.rct else None,
            "rct_probability": round(self.rct.probability, 4) if self.rct and self.rct.probability else None,
            "pico_population": self.pico_spans.get("population", []) if self.pico_spans else [],
            "pico_interventions": self.pico_spans.get("interventions", []) if self.pico_spans else [],
            "pico_outcomes": self.pico_spans.get("outcomes", []) if self.pico_spans else [],
            "bias_summary": {
                b.domain.value: b.judgement.value for b in self.bias
            } if self.bias else {},
            "processing_time": round(self.processing_time_seconds, 2) if self.processing_time_seconds else None,
        }


class AnalysisBatch(BaseModel):
    """Batch of document analyses."""
    batch_id: UUID = Field(default_factory=uuid4)
    documents: list[DocumentAnalysis] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def add_document(self, doc: DocumentAnalysis) -> None:
        """Add a document analysis to the batch."""
        self.documents.append(doc)
    
    @property
    def total_rcts(self) -> int:
        """Count of documents classified as RCTs."""
        return sum(1 for d in self.documents if d.rct and d.rct.is_rct)
    
    @property
    def total_documents(self) -> int:
        """Total number of documents in batch."""
        return len(self.documents)
    
    @property
    def success_count(self) -> int:
        """Count of successfully parsed documents."""
        return sum(1 for d in self.documents if not d.parse_error)
    
    @property
    def error_count(self) -> int:
        """Count of documents with parse errors."""
        return sum(1 for d in self.documents if d.parse_error)
    
    def to_summary_table(self) -> list[dict[str, Any]]:
        """Get summary data for table display."""
        return [doc.to_summary_dict() for doc in self.documents]
    
    def get_bias_summary_matrix(self) -> dict[str, dict[str, str]]:
        """Get a matrix of bias judgements across all documents and domains."""
        domains = [d.value for d in BiasDomain]
        matrix: dict[str, dict[str, str]] = {}
        
        for doc in self.documents:
            if doc.parse_error:
                continue
            doc_key = doc.filename
            matrix[doc_key] = {}
            for bias in doc.bias:
                matrix[doc_key][bias.domain.value] = bias.judgement.value
            # Fill missing domains
            for domain in domains:
                if domain not in matrix[doc_key]:
                    matrix[doc_key][domain] = "N/A"
        
        return matrix