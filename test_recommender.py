"""
test_recommender.py
-------------------
End-to-end test for the recommender pipeline:

    Mock candidate → score_profile → generate_recommendations → pretty-print

Usage
-----
    python test_recommender.py                      # default: ML Engineer
    python test_recommender.py --role "Data Analyst"
    python test_recommender.py --role "BI Developer" --no-weak
    python test_recommender.py --all                # all 5 roles
    python test_recommender.py --json               # save JSON to data/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scoring.score_profile import score_profile, list_available_roles
from recommender.recommend import (
    generate_recommendations,
    recommendations_summary,
)

# ---------------------------------------------------------------------------
# Same mock candidate as test_scorer.py (intentionally missing some skills
# to generate interesting recommendations for each role)
# ---------------------------------------------------------------------------
MOCK_CANDIDATE = {
    "name":  "Jane Doe",
    "email": "jane.doe@email.com",
    "skills": [
        "Python", "PyTorch", "Scikit-learn",
        "Machine Learning", "Deep Learning",
        "Docker", "FastAPI", "SQL", "Git", "AWS",
        # Intentionally missing: TensorFlow, MLOps, Tableau, Power BI, DAX,
        # Data Visualization, Statistics, A/B Testing, System Design,
        # Product Management, Agile, User Research, etc.
    ],
    "experience": [
        {"role": "ML Engineer",        "company": "DataSoft Inc",  "years": "2019-2021"},
        {"role": "Senior ML Engineer", "company": "TechCorp AI",   "years": "2021-2024"},
    ],
    "projects": [
        {
            "title":       "Resume Analyzer",
            "description": "AI tool that parses resumes and scores them using "
                           "sentence-transformers and Gemini API. Deployed via Streamlit.",
        },
        {
            "title":       "Object Detection Dashboard",
            "description": "YOLOv8-based real-time detection, served via FastAPI and Docker.",
        },
    ],
    "education": [
        {"degree": "M.Sc. AI", "institution": "Stanford", "year": "2019"},
    ],
    "github": {
        "commits_last_year": 210,
        "total_stars": 78,
        "public_repos_count": 12,
        "top_languages": ["Python", "Jupyter Notebook"],
    },
    "github_projects": [
        {"title": "llm-experiments", "description": "LLM prompting and RAG experiments", "stars": 44},
    ],
}

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

PRIORITY_ICONS = {1: "🔴", 2: "🟡", 3: "🟢"}
GAP_ICONS      = {"missing": "❌", "weak": "⚠️ "}
SERVICE_ICONS  = {
    "course":      "📚",
    "mentorship":  "🤝",
    "tool":        "🛠️ ",
    "project":     "🚀",
}


def _print_recommendations(recs: list[dict], role: str) -> None:
    if not recs:
        print(f"\n  ✅  No significant skill gaps found for role: {role}\n")
        return

    summary = recommendations_summary(recs)
    print(f"\n{'═' * 70}")
    print(f"  📋  Recommendations for: {role}")
    print(f"{'─' * 70}")
    print(f"  Total gaps : {summary['total_gaps']}   "
          f"Missing: {summary['missing_count']}   "
          f"Weak: {summary['weak_count']}   "
          f"Critical: {summary['critical_count']}")
    print(f"{'═' * 70}")

    for i, rec in enumerate(recs, 1):
        p_icon   = PRIORITY_ICONS[rec["priority"]]
        g_icon   = GAP_ICONS[rec["gap_type"]]
        s_icon   = SERVICE_ICONS.get(rec["service_type"], "📌")
        sim_pct  = f"{rec['similarity_score']:.0%}"

        print(f"\n  {i:>2}. {p_icon} [{rec['priority_label']}]  "
              f"{g_icon} {rec['gap_skill']}  "
              f"(importance: {rec['importance']}/3, similarity: {sim_pct})")
        print(f"       {s_icon}  {rec['recommended_service']}  [{rec['service_type']}]")
        # Wrap reason at 65 chars
        reason = rec["reason"]
        words  = reason.split()
        line   = "       💬  "
        for word in words:
            if len(line) + len(word) > 75:
                print(line)
                line = "           " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

    print(f"\n{'═' * 70}\n")


def _run_for_role(role: str, include_weak: bool, save_json: bool) -> list[dict]:
    print(f"\n⏳ Scoring '{MOCK_CANDIDATE['name']}' vs '{role}'…")
    scoring = score_profile(MOCK_CANDIDATE, role)
    print(f"   Score: {scoring['total_score']:.1f}/100  [{scoring['grade']}]")

    recs = generate_recommendations(
        MOCK_CANDIDATE,
        role,
        scoring,
        include_weak=include_weak,
    )
    _print_recommendations(recs, role)

    if save_json:
        out = ROOT / "data" / f"recommendations_{role.replace(' ', '_').lower()}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {"scoring": scoring, "recommendations": recs},
                f, indent=2, ensure_ascii=False,
            )
        print(f"  💾 Saved: {out}\n")

    return recs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test the recommender pipeline")
    parser.add_argument("--role", default="ML Engineer",
                        help="Target role (default: ML Engineer)")
    parser.add_argument("--all", action="store_true",
                        help="Run against all available roles")
    parser.add_argument("--no-weak", action="store_true",
                        help="Only surface missing skills (skip weak)")
    parser.add_argument("--json", action="store_true",
                        help="Save results to data/ as JSON")
    args = parser.parse_args()

    include_weak = not args.no_weak

    print("\n🤖 SMARRTIF AI — Recommendation Engine Test")
    print(f"   Candidate : {MOCK_CANDIDATE['name']}")
    print(f"   Skills    : {', '.join(MOCK_CANDIDATE['skills'])}")
    print(f"   Mode      : {'all gaps' if include_weak else 'missing only'}")

    available = list_available_roles()

    if args.all:
        for role in available:
            _run_for_role(role, include_weak, args.json)
    else:
        if args.role not in available:
            print(f"\n❌ Unknown role '{args.role}'. "
                  f"Available: {', '.join(available)}")
            sys.exit(1)
        _run_for_role(args.role, include_weak, args.json)


if __name__ == "__main__":
    main()
