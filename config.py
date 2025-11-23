"""
Configuration management for Skills Gap Analyzer.

Handles environment variables, API keys, and application settings.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import json


@dataclass
class AnalyzerConfig:
    """Central configuration for the Skills Gap Analyzer application."""

    # API Configuration
    gemini_api_key: Optional[str] = field(default=None)
    gemini_model: str = field(default="gemini-3-pro-preview")
    ollama_model: str = field(default="llama3.2")
    ollama_base_url: str = field(default="http://localhost:11434")

    # File paths
    output_base_dir: Path = field(default_factory=lambda: Path("analysis_results"))
    log_file: str = field(default="analyzer.log")

    # LLM Settings
    default_temperature: float = field(default=0.7)
    max_retries: int = field(default=3)
    timeout_seconds: int = field(default=300)

    # Analysis parameters
    default_upskilling_weeks: int = field(default=8)
    default_weekly_hours: int = field(default=10)

    # Logging
    verbose: bool = field(default=False)
    log_level: str = field(default="INFO")

    @classmethod
    def from_env(cls) -> "AnalyzerConfig":
        """
        Load configuration from environment variables.

        Returns:
            AnalyzerConfig: Configuration instance populated from environment.
        """
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-pro-preview"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            output_base_dir=Path(os.getenv("OUTPUT_DIR", "analysis_results")),
            log_file=os.getenv("LOG_FILE", "analyzer.log"),
            default_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "300")),
            default_upskilling_weeks=int(os.getenv("DEFAULT_WEEKS", "8")),
            default_weekly_hours=int(os.getenv("DEFAULT_HOURS", "10")),
            verbose=os.getenv("VERBOSE", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )

    @classmethod
    def from_file(cls, config_path: str) -> "AnalyzerConfig":
        """
        Load configuration from a JSON file.

        Args:
            config_path: Path to the configuration JSON file.

        Returns:
            AnalyzerConfig: Configuration instance populated from file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config file is invalid JSON.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Convert output_base_dir to Path if present
            if 'output_base_dir' in config_data:
                config_data['output_base_dir'] = Path(config_data['output_base_dir'])

            return cls(**config_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the configuration settings.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if self.default_temperature < 0 or self.default_temperature > 2:
            errors.append("Temperature must be between 0 and 2")

        if self.max_retries < 0:
            errors.append("Max retries must be non-negative")

        if self.timeout_seconds < 1:
            errors.append("Timeout must be at least 1 second")

        if self.default_upskilling_weeks < 1:
            errors.append("Upskilling weeks must be at least 1")

        if self.default_weekly_hours < 1:
            errors.append("Weekly hours must be at least 1")

        return (len(errors) == 0, errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "gemini_api_key": "***" if self.gemini_api_key else None,
            "gemini_model": self.gemini_model,
            "ollama_model": self.ollama_model,
            "ollama_base_url": self.ollama_base_url,
            "output_base_dir": str(self.output_base_dir),
            "log_file": self.log_file,
            "default_temperature": self.default_temperature,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "default_upskilling_weeks": self.default_upskilling_weeks,
            "default_weekly_hours": self.default_weekly_hours,
            "verbose": self.verbose,
            "log_level": self.log_level
        }
