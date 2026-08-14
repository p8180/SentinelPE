"""Custom exceptions for SentinelPE."""

class SentinelPEError(Exception):
    """Base exception for SentinelPE."""

class ConfigurationError(SentinelPEError):
    """Raised when configuration is invalid."""

class DatasetError(SentinelPEError):
    """Raised for dataset loading or validation errors."""

class ModelTrainingError(SentinelPEError):
    """Raised when model training fails."""

class InferenceError(SentinelPEError):
    """Raised when inference cannot be completed."""

class APIError(SentinelPEError):
    """Raised for API-related failures."""
