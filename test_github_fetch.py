"""
test_github_fetch.py
--------------------
Smoke-test for the GitHub profile fetcher.

Usage
-----
    # With a token (recommended):
    $env:GITHUB_TOKEN = "ghp_your_token_here"
    python test_github_fetch.py octocat

    # Without a token (60 req/hr limit, no commit search):
    python test_github_fetch.py torvalds

    # Merge with a parsed resume:
    python test_github_fetch.py octocat --merge
"""

from __future__ import annotations

import sys
import os
import json
import argparse
from pathlib import Path

# Allow running from the repo root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from parser.github_fetch import fetch_github_profile, merge_github_into_profile, GitHubAPIError

# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _divider(title: str = "", width: int = 64) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * pad)
    else:
        print("─" * width)


def _print_profile(gh: dict) -> None:
    """Pretty-print the github sub-dict."""
    print("\n" + "═" * 64)
    print(f"  🐙  GitHub Profile: @{gh['username']}")
    print("═" * 64)

    _divider("Identity")
    print(f"  Name      : {gh.get('name') or '—'}")
    print(f"  Bio       : {gh.get('bio') or '—'}")
    print(f"  Location  : {gh.get('location') or '—'}")
    print(f"  Company   : {gh.get('company') or '—'}")
    print(f"  URL       : {gh.get('github_url', '')}")
    print(f"  Followers : {gh.get('followers', 0):,}")

    _divider("Stats")
    print(f"  Public repos   : {gh.get('public_repos_count', 0):,}")
    print(f"  Total ⭐ stars  : {gh.get('total_stars', 0):,}")
    print(f"  Commits (1 yr) : {gh.get('commits_last_year', 0):,}")

    _divider("Top Languages")
    breakdown = gh.get("language_breakdown", {})
    if breakdown:
        for lang, count in list(breakdown.items())[:10]:
            bar = "█" * count
            print(f"  {lang:<20} {bar}  ({count} repo{'s' if count != 1 else ''})")
    else:
        print("  (none detected)")

    _divider("Top Repos (by ⭐)")
    repos = gh.get("repos", [])[:8]
    if repos:
        for r in repos:
            lang_tag = f"[{r['language']}]" if r["language"] else "[—]"
            stars    = f"⭐ {r['stars']}"
            desc     = (r["description"] or "")[:60]
            print(f"  {r['name']:<30} {lang_tag:<12} {stars:<8}")
            if desc:
                print(f"    └─ {desc}")
            if r.get("topics"):
                topics_str = " ".join(f"#{t}" for t in r["topics"][:5])
                print(f"    └─ {topics_str}")
    else:
        print("  (no original repos found)")

    print("═" * 64 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the GitHub profile fetcher"
    )
    parser.add_argument(
        "username",
        nargs="?",
        default="octocat",
        help="GitHub username to fetch (default: octocat)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Also demonstrate merging into a dummy candidate profile",
    )
    parser.add_argument(
        "--include-forks",
        action="store_true",
        dest="include_forks",
        help="Include forked repos in stats",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub PAT (overrides GITHUB_TOKEN env var)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "\n⚠️  No GITHUB_TOKEN set. Using unauthenticated requests.\n"
            "   Rate limit: 60 req/hr. Commit search will be skipped.\n"
            "   Set GITHUB_TOKEN for full functionality.\n"
        )

    username = args.username
    print(f"\n🔍 Fetching GitHub profile for: @{username}")
    print(f"   Token   : {'✅ provided' if token else '❌ not set'}")
    print(f"   Forks   : {'included' if args.include_forks else 'excluded'}")

    try:
        result = fetch_github_profile(
            username,
            token=token,
            include_forks=args.include_forks,
        )
    except GitHubAPIError as exc:
        print(f"\n❌ GitHub API Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Unexpected error: {exc}")
        raise

    _print_profile(result["github"])

    # ------------------------------------------------------------------
    # Save JSON output
    # ------------------------------------------------------------------
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{username}_github.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"💾 Full JSON saved to: {out_file}\n")

    # ------------------------------------------------------------------
    # Optional merge demo
    # ------------------------------------------------------------------
    if args.merge:
        dummy_candidate = {
            "name":       "Jane Doe",
            "email":      "jane@example.com",
            "skills":     ["Python", "Machine Learning", "Docker"],
            "experience": [{"role": "ML Engineer", "company": "Acme", "years": "2021-2024"}],
            "projects":   [{"title": "Resume Analyser", "description": "AI-powered CV tool"}],
            "education":  [{"degree": "M.Sc. AI", "institution": "Stanford", "year": "2019"}],
        }

        print("🔀 Merging GitHub data into a dummy candidate profile…")
        merged = merge_github_into_profile(dummy_candidate, result)

        merged_file = out_dir / f"{username}_merged_profile.json"
        with open(merged_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        print(f"\n  Skills after merge ({len(merged['skills'])} total):")
        for s in merged["skills"]:
            print(f"    • {s}")

        print(f"\n  GitHub projects attached: {len(merged['github_projects'])}")
        print(f"\n💾 Merged profile saved to: {merged_file}\n")


if __name__ == "__main__":
    main()
