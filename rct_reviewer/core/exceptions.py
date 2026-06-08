"""
Custom exceptions for RCT-Reviewer v1.0.0.

Copyright (C) 2026 Vihaan Sahu
"""

from typing import Any


class RCTReviewerError(Exception):
    """Base exception for RCT-Reviewer."""
    
    def __init__(self, message: str, *args: Any) -> None:
        self.message = message
        super().__init__(message, *args)
    
    def __str__(self) -> str:
        return self.message


class PDFParseError(RCTReviewerError):
    """Error parsing PDF document."""
    
    def __init__(self, message: str, filename: str = "") -> None:
        self.filename = filename
        full_msg = f"PDF parse error for '{filename}': {message}" if filename else message
        super().__init__(full_msg)


class GrobidError(RCTReviewerError):
    """Error communicating with GROBID service."""
    
    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        full_msg = f"GROBID error (HTTP {status_code}): {message}" if status_code else f"GROBID error: {message}"
        super().__init__(full_msg)


class GrobidNotAvailableError(GrobidError):
    """GROBID service is not reachable."""
    
    def __init__(self, host: str = "unknown") -> None:
        self.host = host
        super().__init__(f"GROBID service at '{host}' is not available. Is it running?")


class ModelLoadError(RCTReviewerError):
    """Error loading ML model."""
    
    def __init__(self, model_name: str, message: str) -> None:
        self.model_name = model_name
        super().__init__(f"Failed to load model '{model_name}': {message}")


class ModelInferenceError(RCTReviewerError):
    """Error during model inference."""
    
    def __init__(self, model_name: str, message: str) -> None:
        self.model_name = model_name
        super().__init__(f"Inference error with model '{model_name}': {message}")


class TextProcessingError(RCTReviewerError):
    """Error during text processing."""
    pass


class ConfigurationError(RCTReviewerError):
    """Error in configuration."""
    pass