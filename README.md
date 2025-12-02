# Skills Gap Analyzer
—————————————————————————————————

## 🚀 Overview

Professional, production-ready AI-powered skills gap analysis tool that compares resumes with job descriptions and provides personalized upskilling recommendations.

## 📁 Project Structure

```
skills-gap-analyzer/
├── config.py                    # Configuration management
├── exceptions.py                # Custom exception classes
├── utils.py                     # Shared utility functions
├── skills_gap_analyzer.py       # Main analyzer application
├── report_formatter.py          # Report generation
├── REFACTORING_SUMMARY.txt      # Detailed refactoring documentation
└── README.md                    # This file
```

## ✨ Key Features

### Production-Ready Architecture
- ✅ **Modular Design**: 5 focused files with clear separation of concerns
- ✅ **Type Safety**: Full type annotations throughout
- ✅ **Error Handling**: Custom exception hierarchy with helpful messages
- ✅ **Logging**: Structured logging with file and console output
- ✅ **Configuration**: Environment variables + JSON config support
- ✅ **CLI Support**: Full argparse integration for automation

### LLM Backend Support
- **Google Gemini** (including Gemini 3!)
- **Ollama** (local LLMs)
- Easy to extend with additional backends (Claude, GPT, etc.)

### Advanced Features
- Automatic dependency installation
- PDF and TXT file support
- Retry logic with exponential backoff
- Multiple report formats (Executive Summary, Detailed Analysis, Interview Prep, Action Plan)
- Configurable upskilling timelines
- Comprehensive input validation

## 📦 Installation

```bash
# Clone or download the project files
cd skills-gap-analyzer

# Install required dependencies
pip install google-generativeai PyPDF2 requests

# Optional: For Ollama support
# Install Ollama from https://ollama.com
```

## ⚙️ Configuration

### Option 1: Environment Variables

```bash
export GEMINI_API_KEY="your-api-key-here"
export GEMINI_MODEL="gemini-3-pro-preview"
export OLLAMA_MODEL="llama3.2"
export OUTPUT_DIR="./my_results"
export VERBOSE="true"
```

### Option 2: Configuration File

Create `config.json`:

```json
{
  "gemini_api_key": "your-api-key",
  "gemini_model": "gemini-3-pro-preview",
  "ollama_model": "llama3.2",
  "ollama_base_url": "http://localhost:11434",
  "output_base_dir": "analysis_results",
  "default_temperature": 0.7,
  "max_retries": 3,
  "verbose": false
}
```

## 🎯 Usage

### Interactive Mode

```bash
python skills_gap_analyzer.py
```

### Command Line Mode

```bash
# Basic usage
python skills_gap_analyzer.py --resume resume.pdf --job job_description.txt

# With configuration file
python skills_gap_analyzer.py --config config.json --resume resume.pdf --job job.txt

# Specify backend
python skills_gap_analyzer.py --resume resume.pdf --job job.txt --backend gemini

# Custom timeline
python skills_gap_analyzer.py --resume resume.pdf --job job.txt --weeks 12

# Verbose mode
python skills_gap_analyzer.py --resume resume.pdf --job job.txt --verbose
```

### Generate Reports

```bash
# Generate all reports
python report_formatter.py --input results.json

# Generate specific reports
python report_formatter.py --input results.json --reports summary action

# Custom output directory
python report_formatter.py --input results.json --output ./reports
```

## 📊 Output

### Analysis Results
- `skills_gap_analysis_results.json` - Complete analysis data
- `analyzer.log` - Detailed execution log

### Generated Reports
- `Executive_Summary.md` - High-level overview
- `Detailed_Analysis.md` - Comprehensive breakdown
- `Interview_Prep_Guide.md` - Interview preparation tips
- `Action_Plan.md` - Personalized learning roadmap

## 🏗️ Architecture

### Core Components

**config.py**
- Centralized configuration management
- Environment variable support
- JSON file loading
- Configuration validation

**exceptions.py**
- Custom exception hierarchy
- Informative error messages
- Proper error classification

**utils.py**
- Shared utility functions
- File validation
- Text processing
- Package installation helpers

**skills_gap_analyzer.py**
- LLM backend abstraction
- Document extraction (PDF/TXT)
- Resume and job parsing
- Skills matching analysis
- Resource recommendation

**report_formatter.py**
- Multiple report formats
- Markdown generation
- Template support
- CLI integration

### Design Patterns
- **Strategy Pattern**: LLM backend selection
- **Factory Pattern**: Backend initialization
- **Template Method**: Report generation

## 🛡️ Error Handling

Comprehensive error handling with custom exceptions:

```python
try:
    backend = GeminiBackend(api_key="...")
except DependencyMissingError as e:
    # Suggests: pip install google-generativeai

try:
    text = DocumentExtractor.extract_from_file("resume.pdf")
except UnsupportedFileTypeError as e:
    # Clear message about supported formats

try:
    results = orchestrator.parse_resume(text)
except LLMBackendError as e:
    # Retry logic already attempted
```

## 🧪 Testing

The modular architecture makes testing straightforward:

```python
# Unit test example
def test_normalize_skill_list():
    from utils import normalize_skill_list

    input_skills = [
        {"skill": "Python"},
        "Java",
        {"name": "SQL"}
    ]

    result = normalize_skill_list(input_skills)
    assert result == ["Python", "Java", "SQL"]
```

## 🔧 Extending

### Adding a New LLM Backend

```python
class ClaudeBackend(LLMBackend):
    def check_availability(self) -> Tuple[bool, str]:
        # Implementation

    def generate(self, prompt: str, **kwargs) -> str:
        # Implementation

    def get_name(self) -> str:
        return "Claude (Anthropic)"
```

### Adding a New Report Type

```python
class ReportFormatter:
    def create_custom_report(self) -> str:
        # Your custom report logic
        return markdown_content
```

## 📝 Code Quality

- **Type Annotations**: 100% coverage
- **Docstrings**: Google-style documentation
- **Error Handling**: Production-grade
- **Logging**: Structured and comprehensive
- **SOLID Principles**: Fully applied
- **DRY**: No code duplication

## 🤝 Contributing

This refactored codebase follows Python best practices:

1. Use type hints everywhere
2. Add docstrings to all public methods
3. Handle errors explicitly
4. Write modular, testable code
5. Follow PEP 8 style guide

## 📄 License

Open source - use as needed for your projects!

## 🙏 Acknowledgments

Refactored from the original Skills Gap Analyzer with:
- ✅ Complete modularization
- ✅ Professional error handling
- ✅ Configuration management
- ✅ CLI automation support
- ✅ Production-ready architecture

---

**Version**: 5.0 (Refactored)  
**Status**: Production Ready ✅  
**Last Updated**: November 2025

For detailed refactoring notes, see `REFACTORING_SUMMARY.txt`
