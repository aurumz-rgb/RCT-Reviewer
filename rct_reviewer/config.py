"""
RCT-Reviewer v1.0.0 Configuration

Uses Pydantic Settings for type-safe, environment-variable-aware configuration.

Copyright (C) 2026 Vihaan Sahu
Based on RobotReviewer by Iain Marshall, Joel Kuiper, Byron Wallace
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GrobidSettings(BaseSettings):
    """GROBID service configuration."""
    
    host: str = Field(
        default="http://localhost:8070",
        description="GROBID service URL"
    )
    timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts"
    )
    enabled: bool = Field(
        default=True,
        description="Whether to use GROBID for PDF parsing"
    )
    
    model_config = SettingsConfigDict(env_prefix="RCT_REVIEWER_GROBID_")


class ModelSettings(BaseSettings):
    """ML model configuration."""
    
    rct_model_name: str = Field(
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        description="Model for RCT classification"
    )
    pico_model_name: str = Field(
        default="dmis-lab/biobert-v1.1",
        description="Model for PICO extraction"
    )
    bias_model_name: str = Field(
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        description="Model for Risk of Bias assessment"
    )
    ner_model_name: str = Field(
        default="dmis-lab/biobert-v1.1",
        description="Model for NER tasks"
    )
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description="Device for inference"
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Batch size for inference"
    )
    max_length: int = Field(
        default=512,
        ge=128,
        le=4096,
        description="Maximum sequence length"
    )
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "rct-reviewer" / "models",
        description="Directory to cache downloaded models"
    )
    
    model_config = SettingsConfigDict(env_prefix="RCT_REVIEWER_MODEL_")


class ProcessingSettings(BaseSettings):
    """Text processing configuration."""
    
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy model to use"
    )
    max_sentences: int = Field(
        default=500,
        ge=10,
        le=5000,
        description="Maximum sentences to process per document"
    )
    pico_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top PICO sentences to return"
    )
    bias_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top bias sentences to return"
    )
    
    model_config = SettingsConfigDict(env_prefix="RCT_REVIEWER_PROCESSING_")


class UISettings(BaseSettings):
    """Streamlit UI configuration."""
    
    page_title: str = Field(
        default="RCT-Reviewer v1.0.0",
        description="Browser tab title"
    )
    page_icon: str = Field(
        default="🔬",
        description="Browser tab icon"
    )
    layout: Literal["centered", "wide"] = Field(
        default="wide",
        description="Page layout"
    )
    show_citation: bool = Field(
        default=True,
        description="Show citation notice in UI"
    )
    
    model_config = SettingsConfigDict(env_prefix="RCT_REVIEWER_UI_")


class Settings(BaseSettings):
    """Main settings container for RCT-Reviewer."""
    
    model_config = SettingsConfigDict(
        env_prefix="RCT_REVIEWER_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    
    grobid: GrobidSettings = Field(default_factory=GrobidSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    ui: UISettings = Field(default_factory=UISettings)
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper


# Global settings instance
settings = Settings()


def get_device() -> str:
    """Determine the best available device for inference."""
    if settings.model.device != "auto":
        return settings.model.device
    
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"