"""
Utility functions shared across the Skills Gap Analyzer application.

Includes normalization, validation, and helper functions.
"""
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Union
from datetime import datetime

from exceptions import ValidationError


def normalize_skill_list(skill_list: List[Any]) -> List[str]:
    """
    Normalize a list of skills from various formats to plain strings.

    Handles:
    - Dictionary objects with keys like 'skill', 'name', 'title', 'label'
    - Plain strings
    - Other objects (converts to string)

    Args:
        skill_list: List of skills in various formats.

    Returns:
        List of normalized skill strings.

    Example:
        >>> normalize_skill_list([{"skill": "Python"}, "Java", {"name": "SQL"}])
        ['Python', 'Java', 'SQL']
    """
    normalized = []

    for item in skill_list:
        if isinstance(item, dict):
            # Try common keys first
            for key in ('skill', 'name', 'title', 'label'):
                if key in item:
                    normalized.append(str(item[key]))
                    break
            else:
                # If no common key found, join all values
                normalized.append(" | ".join(str(v) for v in item.values()))
        else:
            normalized.append(str(item))

    return normalized


def get_safe_filename(s: str, max_length: int = 100) -> str:
    """
    Convert a string to a safe filename by removing invalid characters.

    Args:
        s: Input string to sanitize.
        max_length: Maximum length of the resulting filename.

    Returns:
        Sanitized filename string.

    Example:
        >>> get_safe_filename("My File: Test (2024)")
        'My_File__Test__2024_'
    """
    # Replace invalid characters with underscores
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in s.strip())
    # Truncate to max_length
    return safe[:max_length]


def make_output_folder(
    resume_file: Union[str, Path],
    job_file: Union[str, Path],
    base_dir: Union[str, Path] = "analysis_results"
) -> Path:
    """
    Create a timestamped output folder for analysis results.

    Args:
        resume_file: Path to the resume file.
        job_file: Path to the job description file.
        base_dir: Base directory for output folders.

    Returns:
        Path object pointing to the created output folder.

    Raises:
        OSError: If folder creation fails.
    """
    resume_base = get_safe_filename(Path(resume_file).stem)
    job_base = get_safe_filename(Path(job_file).stem)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    foldername = f"{date_str}_{resume_base}_vs_{job_base}"
    out_folder = Path(base_dir) / foldername

    try:
        out_folder.mkdir(parents=True, exist_ok=True)
        return out_folder
    except OSError as e:
        raise OSError(f"Failed to create output folder '{out_folder}': {e}")


def validate_file_path(
    file_path: Union[str, Path],
    allowed_extensions: List[str] = None,
    max_size_mb: float = None
) -> tuple[bool, str]:
    """
    Validate a file path for existence, type, and size.

    Args:
        file_path: Path to the file to validate.
        allowed_extensions: List of allowed file extensions (e.g., ['.pdf', '.txt']).
        max_size_mb: Maximum file size in megabytes.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    path = Path(file_path)

    # Check existence
    if not path.exists():
        return False, f"File not found: {file_path}"

    if not path.is_file():
        return False, f"Path is not a file: {file_path}"

    # Check extension
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            return False, (
                f"Invalid file type '{path.suffix}'. "
                f"Allowed types: {', '.join(allowed_extensions)}"
            )

    # Check size
    if max_size_mb:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, (
                f"File size ({size_mb:.1f}MB) exceeds maximum "
                f"allowed size ({max_size_mb}MB)"
            )

    return True, ""


def install_package(package_name: str, auto_install: bool = False) -> bool:
    """
    Attempt to install a missing Python package.

    Args:
        package_name: Name of the package to install.
        auto_install: If True, install automatically without prompting.

    Returns:
        True if installation succeeded, False otherwise.
    """
    if not auto_install:
        response = input(
            f"\n📦 Package '{package_name}' is required but not installed.\n"
            f"   Install now? (y/n): "
        ).strip().lower()

        if response != 'y':
            print(f"   Skipping installation. Install manually with: pip install {package_name}")
            return False

    print(f"   Installing {package_name}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"   ✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"   ❌ Failed to install {package_name}")
        print(f"   Please install manually: pip install {package_name}")
        return False


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON object from text that may contain additional content.

    Args:
        text: Text containing JSON (possibly with surrounding text).

    Returns:
        Extracted JSON string, or empty string if not found.
    """
    # Try to find JSON object using regex (raw string)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group()

    # Try to find JSON array
    array_match = re.search(r'\[.*\]', text, re.DOTALL)
    if array_match:
        return array_match.group()

    return ""


def validate_json_structure(data: dict, required_keys: List[str]) -> tuple[bool, List[str]]:
    """
    Validate that a dictionary contains all required keys.

    Args:
        data: Dictionary to validate.
        required_keys: List of required key names.

    Returns:
        Tuple of (is_valid, missing_keys).
    """
    missing = [key for key in required_keys if key not in data]
    return (len(missing) == 0, missing)


def format_duration(weeks: int) -> str:
    """
    Format duration in weeks to human-readable string.

    Args:
        weeks: Number of weeks.

    Returns:
        Formatted string (e.g., "8 weeks (~2 months)").
    """
    if weeks < 1:
        return "< 1 week"
    elif weeks == 1:
        return "1 week"
    elif weeks < 4:
        return f"{weeks} weeks"
    elif weeks < 52:
        months = round(weeks / 4.33)
        return f"{weeks} weeks (~{months} month{'s' if months > 1 else ''})"
    else:
        years = round(weeks / 52, 1)
        return f"{weeks} weeks (~{years} year{'s' if years > 1 else ''})"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding a suffix if truncated.

    Args:
        text: Text to truncate.
        max_length: Maximum length before truncation.
        suffix: Suffix to add if text is truncated.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def colored(text: str, color: str, attrs: List[str] = None) -> str:
    """
    Placeholder for colored terminal output (can be replaced with actual implementation).

    Args:
        text: Text to color.
        color: Color name.
        attrs: Additional attributes (e.g., ['bold']).

    Returns:
        Original text (or colored if implementation added).
    """
    # Placeholder - can integrate with 'termcolor' or 'colorama' if needed
    return text
