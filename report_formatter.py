#!/usr/bin/env python3
"""
Report Formatter for Skills Gap Analyzer.

Generates well-formatted Markdown reports from analysis results.
"""
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import shared utilities
from utils import normalize_skill_list, validate_file_path
from exceptions import FileProcessingError, ConfigurationError


class ReportFormatter:
    """
    Format analysis results into professional Markdown reports.

    Supports multiple report types: Executive Summary, Detailed Analysis,
    Interview Prep Guide, and Action Plan.
    """

    def __init__(
        self,
        results: Dict[str, Any],
        template_style: str = "default"
    ):
        """
        Initialize the report formatter.

        Args:
            results: Analysis results dictionary from JSON file.
            template_style: Template style ('default', 'minimal', 'detailed').
        """
        self.results = results
        self.template_style = template_style

        # Extract main sections
        self.metadata = results.get("session_metadata", {})
        self.resume = results.get("resume_analysis", {})
        self.job = results.get("job_analysis", {})
        self.match = results.get("match_assessment", {})
        self.resources = results.get("learning_resources", {})
        self.upskilling = results.get("upskilling_plan", {})

    def create_executive_summary(self) -> str:
        """
        Create executive summary report.

        Returns:
            Markdown-formatted executive summary.
        """
        score = self.match.get('overall_match_score', 0)
        strengths = normalize_skill_list(self.match.get('strengths', []))
        critical_gaps = normalize_skill_list(self.match.get('critical_gaps', []))

        # Format score indicator
        if score >= 80:
            score_indicator = "🟢 Excellent Match"
        elif score >= 60:
            score_indicator = "🟡 Good Match"
        elif score >= 40:
            score_indicator = "🟠 Moderate Match"
        else:
            score_indicator = "🔴 Needs Development"

        summary = [
            "# 📊 Executive Summary",
            "",
            "## Candidate Profile",
            f"- **Name:** {self.resume.get('name', 'N/A')}",
            f"- **Current Title:** {self.resume.get('current_title', 'N/A')}",
            f"- **Experience:** {self.resume.get('experience_years', 0)} years",
            "",
            "## Target Position",
            f"- **Role:** {self.job.get('job_title', 'N/A')}",
            f"- **Company:** {self.job.get('company_name', 'N/A')}",
            f"- **Location:** {self.job.get('location', 'N/A')}",
            "",
            "## Match Assessment",
            f"- **Overall Score:** {score}/100 - {score_indicator}",
            "",
            "### Key Strengths ✅",
            self._format_list(strengths[:5]),
            "",
            "### Critical Gaps ⚠️",
            self._format_list(critical_gaps[:5]),
            "",
            "## Recommendation",
            f"{self.match.get('recommendation', 'N/A')}",
            "",
            "---",
            f"*Analysis Date: {self.metadata.get('analysis_date', 'N/A')}*",
            f"*Backend: {self.metadata.get('backend', 'N/A')}*"
        ]

        return "\n".join(summary)

    def create_detailed_analysis(self) -> str:
        """
        Create comprehensive detailed analysis report.

        Returns:
            Markdown-formatted detailed analysis.
        """
        score = self.match.get('overall_match_score', 0)

        report = [
            "# 🔍 Detailed Skills Gap Analysis",
            "",
            "## Candidate Information",
            f"**Name:** {self.resume.get('name', 'N/A')}",
            f"**Current Role:** {self.resume.get('current_title', 'N/A')}",
            f"**Current Company:** {self.resume.get('current_company', 'N/A')}",
            f"**Experience:** {self.resume.get('experience_years', 0)} years",
            "",
            "### Current Skills",
            self._format_list(normalize_skill_list(self.resume.get('skills', []))),
            "",
            "### Education",
            self._format_list(self.resume.get('education', [])),
            "",
            "### Key Achievements",
            self._format_list(self.resume.get('key_achievements', [])),
            "",
            "---",
            "",
            "## Target Position",
            f"**Job Title:** {self.job.get('job_title', 'N/A')}",
            f"**Company:** {self.job.get('company_name', 'N/A')}",
            f"**Location:** {self.job.get('location', 'N/A')}",
            f"**Job Type:** {self.job.get('job_type', 'N/A')}",
            f"**Seniority:** {self.job.get('seniority_level', 'N/A')}",
            f"**Experience Required:** {self.job.get('experience_required', 'N/A')}",
            "",
            "### Required Skills",
            self._format_list(normalize_skill_list(self.job.get('required_skills', []))),
            "",
            "### Preferred Skills",
            self._format_list(normalize_skill_list(self.job.get('preferred_skills', []))),
            "",
            "### Key Responsibilities",
            self._format_list(self.job.get('key_responsibilities', [])),
            "",
            "---",
            "",
            "## Skills Match Analysis",
            f"**Overall Match Score:** {score}/100",
            "",
            "### ✅ Matched Skills",
            self._format_list(normalize_skill_list(self.match.get('skills_match', []))),
            "",
            "### 🎯 Key Strengths",
            self._format_list(normalize_skill_list(self.match.get('strengths', []))),
            "",
            "### ⚠️ Critical Gaps",
            self._format_list(normalize_skill_list(self.match.get('critical_gaps', []))),
            "",
            "### 💡 Beneficial Gaps (Nice-to-Have)",
            self._format_list(normalize_skill_list(self.match.get('beneficial_gaps', []))),
            "",
            "### ⚡ Concerns",
            self._format_list(self.match.get('concerns', [])),
            ""
        ]

        # Add upskilling plan if available
        if self.upskilling:
            report.extend([
                "---",
                "",
                "## 📈 Upskilling Plan",
                f"**Plan Title:** {self.upskilling.get('plan_title', 'N/A')}",
                f"**Total Duration:** {self.upskilling.get('total_duration_weeks', 'N/A')} weeks",
                f"**Total Hours Required:** {self.upskilling.get('total_hours_required', 'N/A')} hours",
                f"**Success Rate Estimate:** {self.upskilling.get('success_rate_estimate', 'N/A')}%",
                ""
            ])

            for phase in self.upskilling.get('phases', []):
                phase_num = phase.get('phase_number', 'N/A')
                phase_name = phase.get('phase_name', 'N/A')

                report.extend([
                    f"### Phase {phase_num}: {phase_name}",
                    f"**Duration:** {phase.get('duration_weeks', 'N/A')} weeks",
                    f"**Weekly Commitment:** {phase.get('weekly_commitment', 'N/A')} hours",
                    "",
                    "**Focus Areas:**",
                    self._format_list(normalize_skill_list(phase.get('focus_skills', []))),
                    "",
                    "**Resources:**",
                    self._format_list(normalize_skill_list(phase.get('specific_resources', []))),
                    "",
                    "**Micro-Projects:**",
                    self._format_list(normalize_skill_list(phase.get('micro_projects', []))),
                    "",
                    "**Success Metrics:**",
                    self._format_list(normalize_skill_list(phase.get('success_metrics', []))),
                    ""
                ])

        report.extend([
            "---",
            f"*Generated: {self.metadata.get('analysis_date', 'N/A')}*"
        ])

        return "\n".join(report)

    def create_interview_prep_guide(self) -> str:
        """
        Create interview preparation guide.

        Returns:
            Markdown-formatted interview prep guide.
        """
        strengths = normalize_skill_list(self.match.get('strengths', []))
        critical_gaps = normalize_skill_list(self.match.get('critical_gaps', []))

        guide = [
            "# 💼 Interview Preparation Guide",
            "",
            f"**Position:** {self.job.get('job_title', 'N/A')}",
            f"**Company:** {self.job.get('company_name', 'N/A')}",
            "",
            "## Your Key Strengths to Highlight",
            "",
            "Emphasize these skills and experiences during the interview:",
            ""
        ]

        for i, strength in enumerate(strengths, 1):
            guide.append(f"{i}. **{strength}**")
            guide.append(f"   - Prepare specific examples and achievements")
            guide.append(f"   - Quantify impact where possible")
            guide.append("")

        if critical_gaps:
            guide.extend([
                "## Addressing Skill Gaps",
                "",
                "If asked about these areas, here's how to respond:",
                ""
            ])

            for gap in critical_gaps[:5]:
                guide.append(f"### {gap}")
                guide.append("- Acknowledge the learning opportunity")
                guide.append("- Highlight related transferable skills")
                guide.append("- Express enthusiasm for learning")
                guide.append("- Mention any recent steps you've taken")
                guide.append("")

        guide.extend([
            "## Recommended Talking Points",
            "",
            self.match.get('recommendation', 'N/A'),
            "",
            "## General Interview Tips",
            "",
            "1. **Research the Company**",
            "   - Understand their products/services",
            "   - Know recent news and achievements",
            "   - Review their tech stack (if technical role)",
            "",
            "2. **Prepare Questions**",
            "   - About the role and team",
            "   - About growth opportunities",
            "   - About company culture",
            "",
            "3. **Practice STAR Method**",
            "   - Situation: Set the context",
            "   - Task: Describe the challenge",
            "   - Action: Explain what you did",
            "   - Result: Share the outcome",
            "",
            "---",
            f"*Good luck with your interview at {self.job.get('company_name', 'the company')}!*"
        ])

        return "\n".join(guide)

    def create_action_plan(self) -> str:
        """
        Create actionable learning plan.

        Returns:
            Markdown-formatted action plan.
        """
        weeks = self.metadata.get('timeline_weeks', 'N/A')
        weekly_hours = self.metadata.get('weekly_hours', 'N/A')

        plan = [
            "# 🎯 Action Plan",
            "",
            f"**Timeline:** {weeks} weeks",
            f"**Weekly Commitment:** {weekly_hours} hours",
            "",
            "## Learning Resources",
            ""
        ]

        # Courses
        courses = self.resources.get('courses', [])
        if courses:
            plan.extend(["### 📚 Recommended Courses", ""])
            for course in courses:
                if isinstance(course, dict):
                    title = course.get('title', 'N/A')
                    provider = course.get('provider', '')
                    duration = course.get('duration', '')
                    plan.append(f"- **{title}**")
                    if provider:
                        plan.append(f"  - Provider: {provider}")
                    if duration:
                        plan.append(f"  - Duration: {duration}")
                else:
                    plan.append(f"- {course}")
            plan.append("")

        # Books
        books = self.resources.get('books', [])
        if books:
            plan.extend(["### 📖 Recommended Books", ""])
            for book in books:
                if isinstance(book, dict):
                    title = book.get('title', 'N/A')
                    author = book.get('author', '')
                    plan.append(f"- **{title}**")
                    if author:
                        plan.append(f"  - Author: {author}")
                else:
                    plan.append(f"- {book}")
            plan.append("")

        # Free resources
        free_resources = self.resources.get('free_resources', [])
        if free_resources:
            plan.extend(["### 🌐 Free Resources", ""])
            for resource in free_resources:
                if isinstance(resource, dict):
                    title = resource.get('title', 'N/A')
                    res_type = resource.get('type', '')
                    url = resource.get('url', '')
                    plan.append(f"- **{title}**")
                    if res_type:
                        plan.append(f"  - Type: {res_type}")
                    if url:
                        plan.append(f"  - URL: {url}")
                else:
                    plan.append(f"- {resource}")
            plan.append("")

        # Projects
        projects = self.resources.get('projects', [])
        if projects:
            plan.extend(["## 💻 Practice Projects", ""])
            for project in projects:
                if isinstance(project, dict):
                    title = project.get('title', 'N/A')
                    description = project.get('description', '')
                    skills = project.get('skills_practiced', [])
                    plan.append(f"### {title}")
                    if description:
                        plan.append(f"{description}")
                    if skills:
                        plan.append(f"**Skills Practiced:** {', '.join(skills)}")
                    plan.append("")
                else:
                    plan.append(f"- {project}")
                    plan.append("")

        # Timeline
        timeline = self.resources.get('timeline', {})
        if timeline:
            plan.extend(["## 📅 Week-by-Week Timeline", ""])

            if isinstance(timeline, dict):
                for phase, tasks in sorted(timeline.items()):
                    phase_name = phase.replace('_', ' ').title()
                    plan.append(f"### {phase_name}")
                    if isinstance(tasks, list):
                        for task in tasks:
                            plan.append(f"- {task}")
                    else:
                        plan.append(f"{tasks}")
                    plan.append("")
            else:
                plan.append(str(timeline))
                plan.append("")

        plan.extend([
            "---",
            "",
            "## 🎓 Success Tips",
            "",
            "1. **Stay Consistent:** Dedicate regular time each week",
            "2. **Practice Hands-On:** Build projects, not just consume content",
            "3. **Join Communities:** Connect with others learning similar skills",
            "4. **Track Progress:** Keep a learning journal",
            "5. **Stay Motivated:** Remember your goal!",
            "",
            f"*Start Date: {self.metadata.get('analysis_date', 'Today')}*"
        ])

        return "\n".join(plan)

    @staticmethod
    def _format_list(items: List[Any], max_items: Optional[int] = None) -> str:
        """
        Format a list as Markdown bullet points.

        Args:
            items: List of items to format.
            max_items: Maximum number of items to include.

        Returns:
            Formatted Markdown list.
        """
        if not items:
            return "- *None identified*"

        if max_items:
            items = items[:max_items]

        return "\n".join(f"- {item}" for item in items)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate Markdown reports from Skills Gap Analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python report_formatter.py

  # With results file
  python report_formatter.py --input results.json

  # Specify output directory
  python report_formatter.py --input results.json --output ./reports

  # Generate specific reports
  python report_formatter.py --input results.json --reports summary action
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Path to skills_gap_analysis_results.json file'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for reports (default: same as input file)'
    )

    parser.add_argument(
        '--reports',
        nargs='+',
        choices=['summary', 'detailed', 'interview', 'action', 'all'],
        default=['all'],
        help='Report types to generate (default: all)'
    )

    parser.add_argument(
        '--template',
        type=str,
        choices=['default', 'minimal', 'detailed'],
        default='default',
        help='Report template style (default: default)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Report Formatter v2.0'
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for report generation.
    """
    # Parse arguments
    args = parse_arguments()

    # Get input file
    if args.input:
        source_path = args.input
    else:
        source_path = input("Path to skills_gap_analysis_results.json: ").strip()

    # Validate input file
    is_valid, error = validate_file_path(source_path, ['.json'])
    if not is_valid:
        print(f"❌ {error}")
        sys.exit(1)

    results_path = Path(source_path)

    # Determine output folder
    if args.output:
        out_folder = Path(args.output)
        out_folder.mkdir(parents=True, exist_ok=True)
    else:
        out_folder = results_path.parent

    # Load results
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in results file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load results: {e}")
        sys.exit(1)

    # Initialize formatter
    formatter = ReportFormatter(results, template_style=args.template)

    # Determine which reports to generate
    report_types = args.reports
    if 'all' in report_types:
        report_types = ['summary', 'detailed', 'interview', 'action']

    # Generate reports
    reports = {}

    if 'summary' in report_types:
        reports["Executive_Summary.md"] = formatter.create_executive_summary()

    if 'detailed' in report_types:
        reports["Detailed_Analysis.md"] = formatter.create_detailed_analysis()

    if 'interview' in report_types:
        reports["Interview_Prep_Guide.md"] = formatter.create_interview_prep_guide()

    if 'action' in report_types:
        reports["Action_Plan.md"] = formatter.create_action_plan()

    # Write reports
    print("\n" + "=" * 70)
    print("📝 GENERATING REPORTS")
    print("=" * 70 + "\n")

    for filename, content in reports.items():
        try:
            file_path = out_folder / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ {file_path}")
        except Exception as e:
            print(f"❌ Failed to write {filename}: {e}")

    print("\n" + "=" * 70)
    print("✅ ALL REPORTS GENERATED")
    print("=" * 70)
    print(f"\n📁 Output folder: {out_folder}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Report generation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
