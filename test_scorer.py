"""
test_scorer.py
--------------
Smoke-test for score_profile.py.

Runs a pre-built mock candidate through the scorer against every role,
then scores a "GitHub-enriched" candidate to show the merged pipeline.

Usage
-----
    python test_scorer.py                     # scores mock candidate vs all roles
    python test_scorer.py --role "ML Engineer"  # single role
    python test_scorer.py --json              # dump raw JSON to data/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scoring.score_profile import (
    score_profile,
    score_against_all_roles,
    list_available_roles,
)

# ---------------------------------------------------------------------------
# Mock candidate (mirrors what parse_resume + github_fetch would produce)
# ---------------------------------------------------------------------------

MOCK_CANDIDATE = {
    "name":  "Jane Doe",
    "email": "jane.doe@email.com",
    "phone": "+1-555-234-5678",
    "skills": [
        "Python", "PyTorch", "TensorFlow", "Scikit-learn",
        "Machine Learning", "Deep Learning", "MLOps",
        "Docker", "FastAPI", "SQL", "Git", "AWS",
        "Transformers", "LangChain",
    ],
    "experience": [
        {"role": "Senior ML Engineer",  "company": "TechCorp AI",  "years": "2021-2024"},
        {"role": "ML Engineer",         "company": "DataSoft Inc",  "years": "2019-2021"},
    ],
    "projects": [
        {
            "title":       "Resume Analyzer",
            "description": "End-to-end AI tool that parses PDF/DOCX resumes, embeds skills "
                           "with sentence-transformers, and scores candidates against JD "
                           "requirements using weighted cosine similarity. Deployed via Streamlit.",
        },
        {
            "title":       "Real-Time Object Detection Dashboard",
            "description": "YOLOv8-based object detection system with a live webcam feed, "
                           "served through FastAPI, containerised with Docker.",
        },
        {
            "title":       "LLM Fine-tuning Pipeline",
            "description": "LoRA fine-tuning pipeline for domain-specific LLMs using Hugging Face "
                           "PEFT, experiment tracking via MLflow, results published on HF Hub.",
        },
    ],
    "education": [
        {"degree": "M.Sc. Artificial Intelligence", "institution": "Stanford University", "year": "2019"},
        {"degree": "B.Tech Computer Science",       "institution": "IIT Bombay",          "year": "2017"},
    ],
    # Simulated GitHub data (normally added by merge_github_into_profile)
    "github": {
        "username":          "janedoe",
        "commits_last_year": 312,
        "total_stars":       148,
        "public_repos_count": 17,
        "top_languages":     ["Python", "Jupyter Notebook", "Shell"],
    },
    "github_projects": [
        {"title": "llm-finetune",    "description": "LoRA fine-tuning scripts",            "stars": 89},
        {"title": "yolo-dashboard",  "description": "Real-time detection dashboard",       "stars": 45},
        {"title": "cv-analyser",     "description": "AI-powered resume scoring tool",       "stars": 14},
    ],
}

# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

GRADE_EMOJI = {
    "A+": "🏆", "A": "🥇", "B+": "🥈", "B": "🥉",
    "C+": "🎯", "C": "📋", "D+": "📉", "D": "❌",
}
BAR_WIDTH = 30


def _bar(score: float) -> str:
    filled = round(score / 100 * BAR_WIDTH)
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def _print_score_card(result: dict, verbose: bool = False) -> None:
    role  = result["role"]
    total = result["total_score"]
    grade = result["grade"]
    emoji = GRADE_EMOJI.get(grade, "")

    print(f"\n{'═' * 68}")
    print(f"  {emoji}  {role:<30}  Score: {total:>5.1f} / 100   Grade: {grade}")
    print(f"  {_bar(total)}  {total:.1f}%")
    print(f"{'─' * 68}")

    bd = result["breakdown"]

    dims = [
        ("skill_match",          "Skill Match         "),
        ("experience_relevance", "Experience Relevance"),
        ("project_depth",        "Project Depth       "),
        ("github_activity",      "GitHub Activity     "),
    ]
    for key, label in dims:
        d  = bd[key]
        s  = d["score"]
        w  = d["weight"]
        wc = d["weighted_contribution"]
        print(f"  {label}  {_bar(s)}  {s:>5.1f}  ×{w:.0%} = {wc:.1f}")

    if verbose:
        print()
        # Skill match detail
        sm = bd["skill_match"]
        print(f"  Skill match rate  : {sm.get('match_rate', '—')}%")
        matched  = [m["required"] for m in sm.get("matched_skills", [])]
        missing  = sm.get("missing_skills", [])
        if matched:
            print(f"  ✅ Matched skills  : {', '.join(matched)}")
        if missing:
            print(f"  ❌ Missing skills  : {', '.join(missing)}")

        # Experience
        ex = bd["experience_relevance"]
        print(f"  Total experience  : {ex.get('total_years', 0)} yrs "
              f"({ex.get('experience_entries', 0)} entries)")

        # Project
        pj = bd["project_depth"]
        print(f"  Projects          : {pj.get('project_count', 0)} "
              f"(avg desc: {pj.get('avg_description_length', 0):.0f} chars)")

        # GitHub
        gh = bd["github_activity"]
        if gh.get("available"):
            print(f"  GitHub            : {gh.get('commits_last_year', 0)} commits/yr  "
                  f"⭐ {gh.get('total_stars', 0)}  📦 {gh.get('public_repos_count', 0)} repos")
        else:
            print(f"  GitHub            : not available")

    print(f"{'═' * 68}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test the CV scoring engine")
    parser.add_argument("--role", default=None,
                        help="Score against a specific role only")
    parser.add_argument("--json", action="store_true",
                        help="Save raw JSON results to data/")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-skill match details")
    args = parser.parse_args()

    print("\n🔬 CV Scoring Engine — Test Run")
    print(f"   Candidate : {MOCK_CANDIDATE['name']}")
    print(f"   Skills    : {len(MOCK_CANDIDATE['skills'])}")
    print(f"   Experience: {len(MOCK_CANDIDATE['experience'])} entries")
    print(f"   Projects  : {len(MOCK_CANDIDATE['projects']) + len(MOCK_CANDIDATE['github_projects'])} total")
    print(f"   GitHub    : {'✅' if MOCK_CANDIDATE.get('github') else '❌'}")
    print(f"\n   Available roles: {', '.join(list_available_roles())}\n")

    if args.role:
        # Single role
        print(f"📊 Scoring against: {args.role}")
        result = score_profile(MOCK_CANDIDATE, args.role)
        _print_score_card(result, verbose=args.verbose or True)
        results = [result]
    else:
        # All roles — sorted by score
        print("📊 Scoring against all roles…")
        results = score_against_all_roles(MOCK_CANDIDATE)
        for i, r in enumerate(results, 1):
            verbose_flag = (args.verbose or i == 1)   # verbose for the top match
            _print_score_card(r, verbose=verbose_flag)

        # Best-fit summary
        best = results[0]
        print(f"\n🎯 Best-fit role: {best['role']}  ({best['total_score']:.1f}/100  {best['grade']})\n")

    # Optional JSON dump
    if args.json:
        out_dir = ROOT / "data"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / "scoring_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results if not args.role else results[0], f, indent=2)
        print(f"💾 Raw JSON saved to: {out_file}\n")


if __name__ == "__main__":
    main()
