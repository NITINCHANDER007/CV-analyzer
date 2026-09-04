# AI CV Analyzer
# Parser module — handles PDF and DOCX extraction, and GitHub data fetching

from .parse_resume import parse_resume, extract_text          # noqa: F401
from .github_fetch import (                                    # noqa: F401
    fetch_github_profile,
    merge_github_into_profile,
    GitHubAPIError,
)

__all__ = [
    "parse_resume",
    "extract_text",
    "fetch_github_profile",
    "merge_github_into_profile",
    "GitHubAPIError",
]
