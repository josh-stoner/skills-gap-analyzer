"""
Custom exceptions for the Skills Gap Analyzer application.

Provides specific exception types for better error handling and debugging.
"""


class AnalyzerException(Exception):
    """Base exception for all analyzer-related errors."""
    pass


class FileProcessingError(AnalyzerException):
    """Raised when file reading or processing fails."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to process file '{file_path}': {reason}")


class UnsupportedFileTypeError(FileProcessingError):
    """Raised when attempting to process an unsupported file type."""

    def __init__(self, file_path: str, file_type: str):
        super().__init__(
            file_path,
            f"File type '{file_type}' is not supported. Use PDF or TXT files."
        )


class LLMBackendError(AnalyzerException):
    """Raised when LLM backend encounters an error."""

    def __init__(self, backend_name: str, reason: str):
        self.backend_name = backend_name
        self.reason = reason
        super().__init__(f"{backend_name} backend error: {reason}")


class LLMUnavailableError(LLMBackendError):
    """Raised when LLM backend is not available."""

    def __init__(self, backend_name: str, reason: str, suggestion: str = ""):
        self.suggestion = suggestion
        message = f"{backend_name} is not available: {reason}"
        if suggestion:
            message += f"\n💡 Suggestion: {suggestion}"
        super().__init__(backend_name, message)


class JSONParsingError(AnalyzerException):
    """Raised when LLM output cannot be parsed as JSON."""

    def __init__(self, step_name: str, response_preview: str):
        self.step_name = step_name
        self.response_preview = response_preview
        super().__init__(
            f"Failed to parse JSON from {step_name}. "
            f"Response preview: {response_preview[:200]}"
        )


class DependencyMissingError(AnalyzerException):
    """Raised when a required dependency is not installed."""

    def __init__(self, package_name: str, install_command: str = ""):
        self.package_name = package_name
        self.install_command = install_command or f"pip install {package_name}"
        super().__init__(
            f"Required package '{package_name}' is not installed.\n"
            f"💡 Install with: {self.install_command}"
        )


class ConfigurationError(AnalyzerException):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, errors: list = None):
        self.errors = errors or []
        full_message = f"Configuration error: {message}"
        if self.errors:
            full_message += "\n" + "\n".join(f"  - {err}" for err in self.errors)
        super().__init__(full_message)


class APIKeyError(ConfigurationError):
    """Raised when API key is missing or invalid."""

    def __init__(self, service_name: str):
        super().__init__(
            f"API key for {service_name} is missing or invalid.\n"
            f"💡 Set the API key via environment variable or configuration file."
        )


class ValidationError(AnalyzerException):
    """Raised when input validation fails."""

    def __init__(self, field_name: str, value, reason: str):
        self.field_name = field_name
        self.value = value
        self.reason = reason
        super().__init__(
            f"Validation failed for '{field_name}' with value '{value}': {reason}"
        )
