# AI CV Analyzer
# Recommender module — AI-powered improvement suggestions

from .recommend import (                  # noqa: F401
    generate_recommendations,
    recommendations_summary,
)

__all__ = [
    "generate_recommendations",
    "recommendations_summary",
]
