#!/usr/bin/env python3
"""
Skills Gap Analyzer - AI-Powered Resume and Job Description Analysis.

This module provides comprehensive analysis of skills gaps between resumes
and job descriptions, with personalized upskilling recommendations.
"""
import json
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

import requests

# Local imports
from config import AnalyzerConfig
from exceptions import (
    FileProcessingError,
    UnsupportedFileTypeError,
    LLMBackendError,
    LLMUnavailableError,
    JSONParsingError,
    DependencyMissingError,
    ConfigurationError,
)
from utils import (
    normalize_skill_list,
    get_safe_filename,
    make_output_folder,
    validate_file_path,
    install_package,
    extract_json_from_text,
    truncate_text,
    colored,
)


class AnalyzerLogger:
    """
    Structured logging for Skills Gap Analyzer with console and file output.
    """

    def __init__(
        self,
        log_file: str = 'analyzer.log',
        level: int = logging.INFO,
        verbose: bool = False
    ):
        """
        Initialize the logger.

        Args:
            log_file: Path to the log file.
            level: Logging level (e.g., logging.INFO, logging.DEBUG).
            verbose: If True, print detailed logs to console.
        """
        self.logger = logging.getLogger('SkillsGapAnalyzer')
        self.logger.setLevel(level)
        self.verbose = verbose

        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler (only for warnings and errors by default)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.WARNING)

        # Add handlers if not already added
        if not self.logger.hasHandlers():
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def log(
        self,
        event: str,
        payload: Dict[str, Any],
        level: int = logging.INFO
    ) -> None:
        """
        Log an event with structured payload.

        Args:
            event: Event name/identifier.
            payload: Dictionary containing event details.
            level: Logging level.
        """
        try:
            msg = f"{event} | {json.dumps(payload, default=str)}"
            self.logger.log(level, msg)

            if self.verbose and level >= logging.INFO:
                print(colored(f"[{event}]", "green"))
                print(json.dumps(payload, indent=2, default=str))
        except Exception as e:
            self.logger.error(f"Logging error for event '{event}': {e}")

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
        if self.verbose:
            print(f"ℹ️  {message}")

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
        print(f"⚠️  {message}")

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
        print(f"❌ {message}")

    def success(self, message: str) -> None:
        """Log success message."""
        self.logger.info(f"SUCCESS: {message}")
        print(f"✅ {message}")


class LLMBackend(ABC):
    """Abstract base class for LLM backend implementations."""

    @abstractmethod
    def check_availability(self) -> Tuple[bool, str]:
        """
        Check if the backend is available and properly configured.

        Returns:
            Tuple of (is_available, status_message).
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt for generation.
            temperature: Sampling temperature (0.0 to 2.0).
            max_retries: Maximum number of retry attempts.

        Returns:
            Generated text.

        Raises:
            LLMBackendError: If generation fails after all retries.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name/identifier of this backend."""
        pass


class GeminiBackend(LLMBackend):
    """Google Gemini API backend implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        timeout: int = 300
    ):
        """
        Initialize Gemini backend.

        Args:
            api_key: Google API key.
            model: Gemini model name.
            timeout: Request timeout in seconds.

        Raises:
            DependencyMissingError: If google-generativeai is not installed.
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = None

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise DependencyMissingError(
                "google-generativeai",
                "pip install google-generativeai"
            )
        except Exception as e:
            raise LLMBackendError("Gemini", str(e))

    def check_availability(self) -> Tuple[bool, str]:
        """Check if Gemini API is accessible."""
        if not self.client:
            return False, "Client not initialized"

        try:
            # Test with a simple prompt
            response = self.client.generate_content("test")
            return True, f"Gemini {self.model} ready"
        except Exception as e:
            error_msg = str(e)[:100]
            return False, f"Gemini error: {error_msg}"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> str:
        """Generate text using Gemini API with retry logic."""
        if not self.client:
            raise LLMBackendError("Gemini", "Client not initialized")

        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config={"temperature": temperature}
                )
                return response.text
            except Exception as e:
                if attempt == max_retries - 1:
                    raise LLMBackendError("Gemini", f"Generation failed: {e}")
                # Wait before retry (exponential backoff)
                import time
                time.sleep(2 ** attempt)

        return ""

    def get_name(self) -> str:
        """Get backend name."""
        return f"Gemini ({self.model})"


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend implementation."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 300
    ):
        """
        Initialize Ollama backend.

        Args:
            model: Ollama model name.
            base_url: Ollama server base URL.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.api_endpoint = f"{base_url}/api/generate"

    def check_availability(self) -> Tuple[bool, str]:
        """Check if Ollama server is running and model is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )

            if response.status_code != 200:
                return False, f"Ollama returned status {response.status_code}"

            models = response.json().get('models', [])
            model_names = [m.get('name', '').split(':')[0] for m in models]
            model_base = self.model.split(':')[0]

            if model_base in model_names:
                return True, f"Ollama {self.model} ready"
            else:
                return False, (
                    f"Model '{self.model}' not found.\n"
                    f"💡 Pull with: ollama pull {self.model}"
                )

        except requests.exceptions.ConnectionError:
            return False, (
                "Ollama server not running.\n"
                "💡 Start with: ollama serve"
            )
        except Exception as e:
            return False, f"Ollama error: {str(e)[:100]}"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> str:
        """Generate text using Ollama API with retry logic."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json().get('response', '')
                else:
                    if attempt == max_retries - 1:
                        raise LLMBackendError(
                            "Ollama",
                            f"HTTP {response.status_code}: {response.text[:100]}"
                        )

            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    raise LLMBackendError(
                        "Ollama",
                        f"Request timeout after {self.timeout}s"
                    )

            except Exception as e:
                if attempt == max_retries - 1:
                    raise LLMBackendError("Ollama", str(e))

            # Wait before retry
            import time
            time.sleep(2 ** attempt)

        return ""

    def get_name(self) -> str:
        """Get backend name."""
        return f"Ollama ({self.model})"


class DocumentExtractor:
    """
    Extract text content from various document formats.

    Supports PDF and TXT files with proper error handling.
    """

    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text content.

        Raises:
            DependencyMissingError: If PyPDF2 is not installed.
            FileProcessingError: If extraction fails.
        """
        try:
            import PyPDF2
        except ImportError:
            if install_package("PyPDF2", auto_install=False):
                import PyPDF2
            else:
                raise DependencyMissingError("PyPDF2", "pip install PyPDF2")

        try:
            text_parts = []

            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    raise FileProcessingError(
                        file_path,
                        "PDF is encrypted. Please provide an unencrypted file."
                    )

                num_pages = len(pdf_reader.pages)

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text)
                    except Exception as e:
                        # Log page extraction error but continue
                        logging.warning(
                            f"Failed to extract text from page {page_num + 1}: {e}"
                        )

            extracted_text = "\n".join(text_parts)

            if not extracted_text.strip():
                raise FileProcessingError(
                    file_path,
                    "No text could be extracted from PDF. "
                    "The file may be image-based or corrupted."
                )

            return extracted_text

        except FileProcessingError:
            raise
        except Exception as e:
            raise FileProcessingError(file_path, f"PDF extraction error: {e}")

    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """
        Extract text from a plain text file.

        Args:
            file_path: Path to the text file.

        Returns:
            File content as string.

        Raises:
            FileProcessingError: If reading fails.
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()

                    if content.strip():
                        return content

                except UnicodeDecodeError:
                    continue

            raise FileProcessingError(
                file_path,
                "Unable to decode text file with supported encodings"
            )

        except FileProcessingError:
            raise
        except Exception as e:
            raise FileProcessingError(file_path, f"Text extraction error: {e}")

    @staticmethod
    def extract_from_file(file_path: str) -> str:
        """
        Extract text from a file based on its extension.

        Args:
            file_path: Path to the file.

        Returns:
            Extracted text content.

        Raises:
            UnsupportedFileTypeError: If file type is not supported.
            FileProcessingError: If extraction fails.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == '.pdf':
            return DocumentExtractor.extract_from_pdf(str(path))
        elif suffix in ['.txt', '.text']:
            return DocumentExtractor.extract_from_txt(str(path))
        else:
            raise UnsupportedFileTypeError(str(path), suffix)


class UnifiedOrchestrator:
    """
    Main orchestrator for resume and job description analysis.

    Coordinates between the LLM backend and different analysis steps.
    """

    def __init__(
        self,
        backend: LLMBackend,
        logger: Optional[AnalyzerLogger] = None,
        config: Optional[AnalyzerConfig] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            backend: LLM backend instance.
            logger: Logger instance (creates default if None).
            config: Configuration instance (creates default if None).
        """
        self.backend = backend
        self.logger = logger or AnalyzerLogger()
        self.config = config or AnalyzerConfig()
        self.context: Dict[str, Any] = {}

    def set_user_context(self, context_dict: Dict[str, Any]) -> None:
        """
        Set user-specific context for analysis.

        Args:
            context_dict: Dictionary containing user context.
        """
        self.context = context_dict
        self.logger.log("SET_CONTEXT", {"context": context_dict})

    def _extract_json(
        self,
        step_name: str,
        response: str,
        required_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract and parse JSON from LLM response.

        Args:
            step_name: Name of the analysis step (for logging).
            response: Raw LLM response text.
            required_keys: Optional list of required keys to validate.

        Returns:
            Parsed JSON as dictionary.

        Raises:
            JSONParsingError: If JSON extraction or parsing fails.
        """
        try:
            json_str = extract_json_from_text(response)

            if not json_str:
                self.logger.log(
                    f"{step_name}/EXTRACTION_FAIL",
                    {"error": "No JSON found", "response": response[:250]},
                    level=logging.ERROR
                )
                raise JSONParsingError(step_name, response[:250])

            parsed = json.loads(json_str)

            # Validate required keys if specified
            if required_keys:
                from utils import validate_json_structure
                is_valid, missing = validate_json_structure(parsed, required_keys)
                if not is_valid:
                    self.logger.warning(
                        f"{step_name}: Missing required keys: {missing}"
                    )

            self.logger.log(
                f"{step_name}/EXTRACTED",
                {"result_keys": list(parsed.keys())}
            )

            return parsed

        except json.JSONDecodeError as e:
            self.logger.log(
                f"{step_name}/JSON_ERROR",
                {"error": str(e), "response": response[:250]},
                level=logging.ERROR
            )
            raise JSONParsingError(step_name, response[:250])

        except JSONParsingError:
            raise

        except Exception as e:
            self.logger.log(
                f"{step_name}/ERROR",
                {"error": str(e), "response": response[:250]},
                level=logging.ERROR
            )
            raise JSONParsingError(step_name, response[:250])

    def parse_resume(
        self,
        resume_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse resume text and extract structured information.

        Args:
            resume_text: Raw resume text.
            context: Optional user context.

        Returns:
            Dictionary containing parsed resume data.
        """
        context = context or self.context
        self.logger.log("parse_resume/START", {"context": context})

        prompt = f"""Extract key information from this resume and return ONLY valid JSON.

RESUME:
{resume_text[:3000]}

User Context: {context}

Return exact JSON with this structure:
{{
  "name": "Full Name",
  "current_title": "Current Job Title",
  "current_company": "Current Company",
  "experience_years": 0,
  "skills": ["skill1", "skill2", ...],
  "education": ["degree1", "degree2", ...],
  "key_achievements": ["achievement1", ...],
  "industries": ["industry1", ...]
}}"""

        self.logger.log(
            "parse_resume/PROMPT",
            {"prompt_length": len(prompt), "preview": prompt[:300]}
        )

        try:
            response = self.backend.generate(
                prompt,
                temperature=self.config.default_temperature
            )

            self.logger.log(
                "parse_resume/RAW_RESPONSE",
                {"response_length": len(response), "preview": response[:300]}
            )

            return self._extract_json("parse_resume", response)

        except Exception as e:
            self.logger.error(f"Resume parsing failed: {e}")
            raise

    def parse_job(
        self,
        job_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse job description text and extract structured information.

        Args:
            job_text: Raw job description text.
            context: Optional user context.

        Returns:
            Dictionary containing parsed job data.
        """
        context = context or self.context
        self.logger.log("parse_job/START", {"context": context})

        prompt = f"""Extract key information from this job posting and return ONLY valid JSON.

JOB POSTING:
{job_text[:3000]}

User Context: {context}

Return JSON with this structure:
{{
  "job_title": "Position Title",
  "company_name": "Company Name",
  "location": "Location",
  "salary_range": "Salary Range",
  "job_type": "Full-time/Part-time/Contract",
  "seniority_level": "Entry/Mid/Senior",
  "experience_required": "X years",
  "required_skills": ["skill1", "skill2", ...],
  "preferred_skills": ["skill1", "skill2", ...],
  "key_responsibilities": ["responsibility1", ...],
  "education_required": "Degree requirement"
}}"""

        self.logger.log(
            "parse_job/PROMPT",
            {"prompt_length": len(prompt), "preview": prompt[:300]}
        )

        try:
            response = self.backend.generate(
                prompt,
                temperature=self.config.default_temperature
            )

            self.logger.log(
                "parse_job/RAW_RESPONSE",
                {"response_length": len(response), "preview": response[:300]}
            )

            return self._extract_json("parse_job", response)

        except Exception as e:
            self.logger.error(f"Job parsing failed: {e}")
            raise

    def match_skills(
        self,
        resume_data: Dict[str, Any],
        job_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform skills gap analysis between resume and job description.

        Args:
            resume_data: Parsed resume data.
            job_data: Parsed job data.
            context: Optional user context.

        Returns:
            Dictionary containing match assessment.
        """
        context = context or self.context
        self.logger.log(
            "match_skills/START",
            {
                "resume_name": resume_data.get('name', 'N/A'),
                "job_title": job_data.get('job_title', 'N/A'),
                "context": context
            }
        )

        candidate_skills = ', '.join(
            normalize_skill_list(resume_data.get('skills', []))[:15]
        )
        required_skills = ', '.join(
            normalize_skill_list(job_data.get('required_skills', []))[:15]
        )
        preferred_skills = ', '.join(
            normalize_skill_list(job_data.get('preferred_skills', []))[:10]
        )

        prompt = f"""Compare candidate skills with job requirements and return ONLY valid JSON.

CANDIDATE: {resume_data.get('name', 'Candidate')}
CURRENT ROLE: {resume_data.get('current_title', 'N/A')}
EXPERIENCE: {resume_data.get('experience_years', 0)} years
SKILLS: {candidate_skills}

TARGET JOB: {job_data.get('job_title', 'Position')}
REQUIRED SKILLS: {required_skills}
PREFERRED SKILLS: {preferred_skills}

User Context: {context}

Provide a comprehensive skills assessment. Return JSON:
{{
  "overall_match_score": 0-100,
  "skills_match": ["matched skill1", "matched skill2", ...],
  "critical_gaps": ["missing critical skill1", ...],
  "beneficial_gaps": ["missing beneficial skill1", ...],
  "strengths": ["key strength1", ...],
  "concerns": ["concern1", ...],
  "recommendation": "Detailed recommendation text"
}}"""

        self.logger.log(
            "match_skills/PROMPT",
            {"prompt_length": len(prompt), "preview": prompt[:300]}
        )

        try:
            response = self.backend.generate(
                prompt,
                temperature=0.5
            )

            self.logger.log(
                "match_skills/RAW_RESPONSE",
                {"response_length": len(response), "preview": response[:300]}
            )

            result = self._extract_json("match_skills", response)

            # Log match score
            score = result.get('overall_match_score', 0)
            self.logger.info(f"Skills match score: {score}/100")

            return result

        except Exception as e:
            self.logger.error(f"Skills matching failed: {e}")
            raise


class ResourceRecommender:
    """
    Generate personalized learning resources and upskilling plans.
    """

    def __init__(
        self,
        backend: LLMBackend,
        logger: Optional[AnalyzerLogger] = None,
        config: Optional[AnalyzerConfig] = None
    ):
        """
        Initialize the resource recommender.

        Args:
            backend: LLM backend instance.
            logger: Logger instance.
            config: Configuration instance.
        """
        self.backend = backend
        self.logger = logger or AnalyzerLogger()
        self.config = config or AnalyzerConfig()

    def generate_resources(
        self,
        skill_gaps: List[str],
        job_role: str,
        candidate_background: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate learning resources for identified skill gaps.

        Args:
            skill_gaps: List of skills to acquire.
            job_role: Target job role.
            candidate_background: Candidate's background summary.
            context: Optional user context.

        Returns:
            Dictionary containing learning resources and timeline.
        """
        context = context or {}
        self.logger.log(
            "generate_resources/START",
            {
                "skill_gaps_count": len(skill_gaps),
                "job_role": job_role,
                "context": context
            }
        )

        skill_gap_strings = normalize_skill_list(skill_gaps[:8])

        prompt = f"""You are a career development expert. Generate SPECIFIC learning resources.

TARGET ROLE: {job_role}
CANDIDATE BACKGROUND: {candidate_background}
SKILL GAPS TO ADDRESS: {', '.join(skill_gap_strings)}
USER CONTEXT: {context}

Generate a detailed, REALISTIC learning plan with ACTUAL resources.

Return ONLY valid JSON:
{{
  "courses": [
    {{"title": "Course Name", "provider": "Platform", "duration": "X weeks", "url": "URL or N/A"}},
    ...
  ],
  "books": [
    {{"title": "Book Title", "author": "Author Name", "isbn": "ISBN or N/A"}},
    ...
  ],
  "free_resources": [
    {{"title": "Resource Name", "type": "Documentation/Tutorial/Video", "url": "URL"}},
    ...
  ],
  "projects": [
    {{"title": "Project Name", "description": "Brief description", "skills_practiced": ["skill1", ...]}},
    ...
  ],
  "timeline": {{
    "phase_1_weeks_1_4": ["Task 1", "Task 2", ...],
    "phase_2_weeks_5_8": ["Task 1", "Task 2", ...],
    "phase_3_weeks_9_12": ["Task 1", "Task 2", ...]
  }}
}}"""

        self.logger.log(
            "generate_resources/PROMPT",
            {"prompt_length": len(prompt), "preview": prompt[:300]}
        )

        try:
            response = self.backend.generate(
                prompt,
                temperature=0.7
            )

            self.logger.log(
                "generate_resources/RAW_RESPONSE",
                {"response_length": len(response), "preview": response[:300]}
            )

            json_str = extract_json_from_text(response)
            if not json_str:
                self.logger.error("Failed to extract JSON from resource generation")
                return {}

            resources = json.loads(json_str)

            self.logger.log(
                "generate_resources/EXTRACTED",
                {
                    "courses": len(resources.get('courses', [])),
                    "books": len(resources.get('books', [])),
                    "free_resources": len(resources.get('free_resources', [])),
                    "projects": len(resources.get('projects', []))
                }
            )

            return resources

        except Exception as e:
            self.logger.error(f"Resource generation failed: {e}")
            return {}


def select_backend(
    config: Optional[AnalyzerConfig] = None,
    verbose: bool = False
) -> Optional[LLMBackend]:
    """
    Interactive backend selection with availability checking.

    Args:
        config: Configuration instance (uses defaults if None).
        verbose: Enable verbose output.

    Returns:
        Initialized LLM backend, or None if selection fails.
    """
    config = config or AnalyzerConfig()

    print("\n" + "=" * 70)
    print("🤖 SELECT AI BACKEND")
    print("=" * 70)
    print("\nAvailable options:")
    print("  1. Gemini (Google Cloud API)")
    print("  2. Ollama (Local LLM)")

    choice = input("\nSelect backend (1-2): ").strip()

    if choice == "2":
        # Ollama backend
        model = input(
            f"\nEnter model name (default: {config.ollama_model}): "
        ).strip() or config.ollama_model

        try:
            backend = OllamaBackend(
                model=model,
                base_url=config.ollama_base_url,
                timeout=config.timeout_seconds
            )

            available, message = backend.check_availability()
            if available:
                print(f"✅ {message}")
                return backend
            else:
                print(f"❌ {message}")
                return None

        except Exception as e:
            print(f"❌ Failed to initialize Ollama: {e}")
            return None

    elif choice == "1":
        # Gemini backend
        api_key = config.gemini_api_key or input(
            "\nPaste your Gemini API key: "
        ).strip()

        if not api_key:
            print("❌ API key is required for Gemini")
            return None

        model_choice = input(
            f"\nSelect model (default: {config.gemini_model}): "
        ).strip() or config.gemini_model

        try:
            backend = GeminiBackend(
                api_key=api_key,
                model=model_choice,
                timeout=config.timeout_seconds
            )

            available, message = backend.check_availability()
            if available:
                print(f"✅ {message}")
                return backend
            else:
                print(f"❌ {message}")
                return None

        except DependencyMissingError as e:
            print(f"❌ {e}")
            return None
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            return None

    else:
        print("❌ Invalid choice")
        return None


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="AI Skills Gap Analyzer - Compare resumes with job descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python skills_gap_analyzer.py

  # With file paths
  python skills_gap_analyzer.py --resume my_resume.pdf --job job_desc.txt

  # With config file
  python skills_gap_analyzer.py --config config.json --resume resume.pdf --job job.txt

  # Verbose mode
  python skills_gap_analyzer.py --resume resume.pdf --job job.txt --verbose
        """
    )

    parser.add_argument(
        '--resume',
        type=str,
        help='Path to resume file (PDF or TXT)'
    )

    parser.add_argument(
        '--job',
        type=str,
        help='Path to job description file (PDF or TXT)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (default: analysis_results)'
    )

    parser.add_argument(
        '--weeks',
        type=int,
        help='Upskilling timeframe in weeks (default: 8)'
    )

    parser.add_argument(
        '--backend',
        type=str,
        choices=['gemini', 'ollama'],
        help='LLM backend to use (bypasses interactive selection)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Skills Gap Analyzer v5.0'
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the Skills Gap Analyzer.

    Handles CLI arguments, file validation, backend selection,
    and orchestrates the complete analysis workflow.
    """
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    try:
        if args.config:
            config = AnalyzerConfig.from_file(args.config)
        else:
            config = AnalyzerConfig.from_env()

        # Override config with CLI arguments
        if args.output_dir:
            config.output_base_dir = Path(args.output_dir)
        if args.weeks:
            config.default_upskilling_weeks = args.weeks
        if args.verbose:
            config.verbose = True

        # Validate configuration
        is_valid, errors = config.validate()
        if not is_valid:
            raise ConfigurationError("Invalid configuration", errors)

    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # Print header
    print("\n" + "=" * 70)
    print("🚀 AI Skills Gap Analyzer v5.0")
    print("=" * 70)

    # Get file paths
    if args.resume and args.job:
        resume_path = args.resume
        job_path = args.job
    else:
        resume_path = input("\nResume file (PDF or TXT): ").strip().strip('\"').strip("'")
        job_path = input("Job description (PDF or TXT): ").strip().strip('\"').strip("'")

    # Validate files
    allowed_exts = ['.pdf', '.txt']

    is_valid, error = validate_file_path(resume_path, allowed_exts, max_size_mb=10)
    if not is_valid:
        print(f"❌ Resume file error: {error}")
        sys.exit(1)

    is_valid, error = validate_file_path(job_path, allowed_exts, max_size_mb=10)
    if not is_valid:
        print(f"❌ Job description error: {error}")
        sys.exit(1)

    # Create output folder
    try:
        out_folder = make_output_folder(
            resume_path,
            job_path,
            config.output_base_dir
        )
        output_file = out_folder / "skills_gap_analysis_results.json"
        log_file = out_folder / "analyzer.log"
    except OSError as e:
        print(f"❌ Failed to create output folder: {e}")
        sys.exit(1)

    # Initialize logger
    logger = AnalyzerLogger(
        log_file=str(log_file),
        level=logging.DEBUG if config.verbose else logging.INFO,
        verbose=config.verbose
    )

    logger.info(f"Output folder: {out_folder}")
    logger.info(f"Configuration: {config.to_dict()}")

    # Select backend
    try:
        if args.backend == 'gemini':
            if not config.gemini_api_key:
                config.gemini_api_key = input("\nGemini API key: ").strip()
            backend = GeminiBackend(
                api_key=config.gemini_api_key,
                model=config.gemini_model,
                timeout=config.timeout_seconds
            )
        elif args.backend == 'ollama':
            backend = OllamaBackend(
                model=config.ollama_model,
                base_url=config.ollama_base_url,
                timeout=config.timeout_seconds
            )
        else:
            backend = select_backend(config, config.verbose)

        if not backend:
            print("\n❌ No backend available. Exiting.")
            sys.exit(1)

        # Check availability
        available, message = backend.check_availability()
        if not available:
            print(f"❌ Backend not available: {message}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Backend initialization failed: {e}")
        sys.exit(1)

    # Extract documents
    print("\n📄 EXTRACTING DOCUMENTS")
    print("-" * 70)

    try:
        logger.info(f"Extracting resume from: {resume_path}")
        resume_text = DocumentExtractor.extract_from_file(resume_path)

        if len(resume_text.strip()) < 50:
            raise FileProcessingError(
                resume_path,
                "Extracted text is too short. Check file content."
            )

        logger.success(f"Resume extracted: {len(resume_text)} characters")
        print(f"  ✅ Resume: {len(resume_text)} characters")
        print(f"     Preview: {truncate_text(resume_text, 80)}")

    except Exception as e:
        logger.error(f"Resume extraction failed: {e}")
        print(f"❌ {e}")
        sys.exit(1)

    try:
        logger.info(f"Extracting job description from: {job_path}")
        job_text = DocumentExtractor.extract_from_file(job_path)

        if len(job_text.strip()) < 50:
            raise FileProcessingError(
                job_path,
                "Extracted text is too short. Check file content."
            )

        logger.success(f"Job description extracted: {len(job_text)} characters")
        print(f"  ✅ Job description: {len(job_text)} characters")
        print(f"     Preview: {truncate_text(job_text, 80)}")

    except Exception as e:
        logger.error(f"Job extraction failed: {e}")
        print(f"❌ {e}")
        sys.exit(1)

    # Analysis phase
    print("\n🔍 ANALYSIS")
    print("-" * 70)

    orchestrator = UnifiedOrchestrator(backend, logger, config)

    # Get upskilling timeframe
    if args.weeks:
        available_weeks = args.weeks
    else:
        weeks_input = input(
            f"\nEnter upskilling timeframe in weeks (default: {config.default_upskilling_weeks}): "
        ).strip()
        available_weeks = int(weeks_input) if weeks_input.isdigit() else config.default_upskilling_weeks

    context = {
        "desired_timeframe_weeks": available_weeks,
        "weekly_hours_available": config.default_weekly_hours
    }
    orchestrator.set_user_context(context)

    try:
        # Parse resume
        print("  📋 Parsing resume...")
        resume_data = orchestrator.parse_resume(resume_text)
        logger.success(f"Resume parsed: {resume_data.get('name', 'N/A')}")
        print(f"     ✅ Name: {resume_data.get('name', 'N/A')}")
        print(f"     ✅ Title: {resume_data.get('current_title', 'N/A')}")

        # Parse job description
        print("\n  📋 Parsing job description...")
        job_data = orchestrator.parse_job(job_text)
        logger.success(f"Job parsed: {job_data.get('job_title', 'N/A')}")
        print(f"     ✅ Position: {job_data.get('job_title', 'N/A')}")
        print(f"     ✅ Company: {job_data.get('company_name', 'N/A')}")

        # Match skills
        print("\n  🎯 Matching skills...")
        match_data = orchestrator.match_skills(resume_data, job_data)
        score = match_data.get('overall_match_score', 0)
        logger.success(f"Skills matched: {score}/100")
        print(f"     ✅ Match Score: {score}/100")

        strengths = normalize_skill_list(match_data.get('strengths', []))
        gaps = normalize_skill_list(match_data.get('critical_gaps', []))

        if strengths:
            print(f"     ✅ Top Strengths: {', '.join(strengths[:3])}")
        if gaps:
            print(f"     ⚠️  Critical Gaps: {', '.join(gaps[:3])}")

        # Generate resources
        print("\n  📚 Generating learning resources...")
        recommender = ResourceRecommender(backend, logger, config)
        resources = recommender.generate_resources(
            skill_gaps=match_data.get('critical_gaps', []),
            job_role=job_data.get('job_title', 'Target Position'),
            candidate_background=(
                f"{resume_data.get('current_title', 'Professional')} "
                f"with {resume_data.get('experience_years', 0)} years experience"
            ),
            context=context
        )

        num_courses = len(resources.get('courses', []))
        num_books = len(resources.get('books', []))
        num_projects = len(resources.get('projects', []))

        logger.success(
            f"Resources generated: {num_courses} courses, "
            f"{num_books} books, {num_projects} projects"
        )
        print(f"     ✅ Courses: {num_courses}")
        print(f"     ✅ Books: {num_books}")
        print(f"     ✅ Projects: {num_projects}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis error: {e}")
        sys.exit(1)

    # Save results
    try:
        results = {
            "session_metadata": {
                "backend": backend.get_name(),
                "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "version": "5.0",
                "timeline_weeks": available_weeks,
                "weekly_hours": config.default_weekly_hours,
                "resume_file": str(resume_path),
                "job_description_file": str(job_path)
            },
            "resume_analysis": resume_data,
            "job_analysis": job_data,
            "match_assessment": match_data,
            "learning_resources": resources
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.success(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        print(f"\n❌ Save error: {e}")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  Backend Used: {backend.get_name()}")
    print(f"  Match Score: {score}/100")
    print(f"  Timeline: {available_weeks} weeks")
    print(f"\n  📁 Results: {output_file}")
    print(f"  📁 Log: {log_file}")
    print("\n  💡 Generate formatted reports with: python report_formatter.py")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        logging.exception("Unexpected error in main")
        sys.exit(1)
