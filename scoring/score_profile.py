"""
score_profile.py
----------------
Scores a parsed candidate profile against a target job role.

Scoring Dimensions
~~~~~~~~~~~~~~~~~~
┌──────────────────────┬────────┬──────────────────────────────────────────────┐
│ Dimension            │ Weight │ Method                                       │
├──────────────────────┼────────┼──────────────────────────────────────────────┤
│ Skill Match          │  50 %  │ Weighted cosine similarity via               │
│                      │        │ sentence-transformers (all-MiniLM-L6-v2)     │
│ Experience Relevance │  25 %  │ Total years of experience mapped to a        │
│                      │        │ seniority curve (0–100)                      │
│ Project Depth        │  15 %  │ Project count + avg description richness     │
│ GitHub Activity      │  10 %  │ Commits / year, stars, repos (if present)    │
└──────────────────────┴────────┴──────────────────────────────────────────────┘

Final score = Σ (dimension_score × dimension_weight), normalised to 0–100.

Usage
~~~~~
    from scoring.score_profile import score_profile

    result = score_profile(candidate_profile, "ML Engineer")
    print(result["total_score"])   # e.g. 78.4
    print(result["grade"])         # e.g. "B+"
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("score_profile")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_NAME   = "all-MiniLM-L6-v2"
_ROLES_PATH   = Path(__file__).resolve().parents[1] / "data" / "roles.json"

# Dimension weights must sum to 1.0
_DIM_WEIGHTS = {
    "skill_match":          0.50,
    "experience_relevance": 0.25,
    "project_depth":        0.15,
    "github_activity":      0.10,
}

# Seniority curve: (min_years, max_years, score)
# Score interpolated linearly within each band
_EXPERIENCE_BANDS = [
    (0,  1,   0,  30),   # (min_yrs, max_yrs, score_at_min, score_at_max)
    (1,  3,  30,  55),
    (3,  5,  55,  75),
    (5,  8,  75,  90),
    (8, 30,  90, 100),
]

# Cosine similarity threshold below which a skill is considered "missing"
_MATCH_THRESHOLD = 0.40

# Grade boundaries (score → letter)
_GRADE_THRESHOLDS = [
    (93, "A+"), (87, "A"), (80, "B+"), (73, "B"),
    (65, "C+"), (58, "C"), (50, "D+"), (0,  "D"),
]


# ---------------------------------------------------------------------------
# Module-level model cache (loaded once per process)
# ---------------------------------------------------------------------------
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance (lazy-loaded)."""
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model '%s'…", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Roles loader
# ---------------------------------------------------------------------------

def load_roles(roles_path: Path = _ROLES_PATH) -> dict:
    """Load and return the roles dict from roles.json, stripping _meta."""
    if not roles_path.exists():
        raise FileNotFoundError(
            f"roles.json not found at '{roles_path}'. "
            "Run the project from the repo root."
        )
    with open(roles_path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def list_available_roles(roles_path: Path = _ROLES_PATH) -> list[str]:
    """Return a sorted list of all available role names."""
    return sorted(load_roles(roles_path).keys())


# ---------------------------------------------------------------------------
# Dimension 1 — Skill Match (sentence-transformers cosine similarity)
# ---------------------------------------------------------------------------

def _embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode a list of texts into L2-normalised embedding vectors."""
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(vecs)


def _score_skill_match(
    candidate_skills: list[str],
    role_requirements: list[dict],
    model: SentenceTransformer,
) -> dict:
    """
    Compute a weighted skill match score (0–100).

    For each required skill:
    - Embed it and find its maximum cosine similarity against *all*
      candidate skill embeddings.
    - Multiply the similarity by the role's importance weight (1–3).
    - Accumulate weighted similarity and total possible weight.

    Final score = (sum_weighted_sim / max_possible_weight) × 100.

    Returns a detail dict including per-skill similarity and matched/missing lists.
    """
    if not candidate_skills:
        return {
            "score": 0.0,
            "matched_skills": [],
            "missing_skills": [r["skill"] for r in role_requirements],
            "per_skill_detail": [],
        }

    # Embed all candidate skills once
    cand_embeddings = _embed(candidate_skills, model)

    required_skill_texts = [r["skill"] for r in role_requirements]
    req_embeddings       = _embed(required_skill_texts, model)

    # cosine_similarity returns shape (n_required, n_candidate)
    sim_matrix = cosine_similarity(req_embeddings, cand_embeddings)

    total_weighted_sim  = 0.0
    max_possible_weight = 0.0
    matched: list[dict] = []
    missing: list[str]  = []
    per_skill: list[dict] = []

    for idx, req in enumerate(role_requirements):
        sims            = sim_matrix[idx]            # similarity against all candidate skills
        best_sim        = float(np.max(sims))
        best_cand_idx   = int(np.argmax(sims))
        best_cand_skill = candidate_skills[best_cand_idx]
        weight          = req["weight"]

        weighted_sim     = best_sim * weight
        total_weighted_sim  += weighted_sim
        max_possible_weight += weight            # perfect match = weight × 1.0

        detail = {
            "required_skill":    req["skill"],
            "weight":            weight,
            "best_match":        best_cand_skill,
            "similarity":        round(best_sim, 4),
            "weighted_contribution": round(weighted_sim, 4),
            "matched":           best_sim >= _MATCH_THRESHOLD,
        }
        per_skill.append(detail)

        if best_sim >= _MATCH_THRESHOLD:
            matched.append({"required": req["skill"], "matched_to": best_cand_skill,
                            "similarity": round(best_sim, 4)})
        else:
            missing.append(req["skill"])

    raw_score = (total_weighted_sim / max_possible_weight) if max_possible_weight else 0.0
    score_0_100 = round(min(raw_score * 100, 100.0), 2)

    return {
        "score":           score_0_100,
        "matched_skills":  matched,
        "missing_skills":  missing,
        "match_rate":      round(len(matched) / len(role_requirements) * 100, 1),
        "per_skill_detail": per_skill,
    }


# ---------------------------------------------------------------------------
# Dimension 2 — Experience Relevance
# ---------------------------------------------------------------------------

def _parse_years_from_entry(entry: dict) -> float:
    """
    Best-effort parse of a years value from an experience entry.

    Handles formats like:
    - "2020-2023"  → 3.0
    - "2019-Present" / "2019-present" / "2021-Now" → current_year - 2019
    - "3 years"    → 3.0
    - "2 yrs"      → 2.0
    - "Jan 2020 - Mar 2022" → ~2.2
    - plain int/float
    """
    years_raw = entry.get("years", "") or ""
    now = datetime.now().year

    if not years_raw:
        return 0.0

    years_str = str(years_raw).strip()

    # "N years" / "N yrs"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", years_str, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # "YYYY-YYYY" or "YYYY – YYYY"
    m = re.search(r"(\d{4})\s*[-–]\s*(\d{4})", years_str)
    if m:
        return max(0.0, float(m.group(2)) - float(m.group(1)))

    # "YYYY-Present" / "YYYY-Now" / "YYYY-Current"
    m = re.search(r"(\d{4})\s*[-–]\s*(?:present|now|current|today)", years_str, re.IGNORECASE)
    if m:
        return max(0.0, float(now - int(m.group(1))))

    # Bare 4-digit year → assume still there, 0 years (joining year)
    m = re.fullmatch(r"\d{4}", years_str.strip())
    if m:
        return 0.0

    # Plain number
    try:
        return max(0.0, float(years_str))
    except ValueError:
        return 0.0


def _score_experience(candidate: dict) -> dict:
    """
    Map total years of professional experience to a 0–100 score using
    the seniority curve defined in _EXPERIENCE_BANDS.
    """
    experience_entries = candidate.get("experience", []) or []

    # Also check github_projects for additional signals — not used here,
    # but we surface the count in details.
    total_years = sum(_parse_years_from_entry(e) for e in experience_entries)

    # Interpolate within the matching band
    score_0_100 = 0.0
    for min_y, max_y, score_min, score_max in _EXPERIENCE_BANDS:
        if min_y <= total_years <= max_y:
            if max_y == min_y:
                score_0_100 = float(score_max)
            else:
                t = (total_years - min_y) / (max_y - min_y)
                score_0_100 = score_min + t * (score_max - score_min)
            break
    else:
        # total_years > last band max
        score_0_100 = 100.0

    return {
        "score":              round(score_0_100, 2),
        "total_years":        round(total_years, 1),
        "experience_entries": len(experience_entries),
        "parsed_entries": [
            {
                "role":    e.get("role", ""),
                "company": e.get("company", ""),
                "years_raw": e.get("years", ""),
                "years_parsed": round(_parse_years_from_entry(e), 1),
            }
            for e in experience_entries
        ],
    }


# ---------------------------------------------------------------------------
# Dimension 3 — Project Depth
# ---------------------------------------------------------------------------

def _score_project_depth(candidate: dict) -> dict:
    """
    Proxy for project portfolio richness:

    - Base score from project count (resume projects + GitHub projects)
    - Bonus from average description length (richness signal)
    - Combined and capped at 100.

    Score formula:
        count_score   = min(project_count / 6, 1.0) × 70
        richness_bonus = min(avg_desc_len / 200, 1.0) × 30
        total = count_score + richness_bonus
    """
    resume_projects = candidate.get("projects", []) or []
    github_projects = candidate.get("github_projects", []) or []
    all_projects    = resume_projects + github_projects

    project_count = len(all_projects)

    descriptions = [
        p.get("description", "") or ""
        for p in all_projects
    ]
    avg_desc_len = (
        sum(len(d) for d in descriptions) / len(descriptions)
        if descriptions else 0.0
    )

    count_score    = min(project_count / 6.0, 1.0) * 70.0
    richness_bonus = min(avg_desc_len / 200.0, 1.0) * 30.0
    score_0_100    = round(min(count_score + richness_bonus, 100.0), 2)

    return {
        "score":            score_0_100,
        "project_count":    project_count,
        "resume_projects":  len(resume_projects),
        "github_projects":  len(github_projects),
        "avg_description_length": round(avg_desc_len, 1),
        "count_score":      round(count_score, 2),
        "richness_bonus":   round(richness_bonus, 2),
    }


# ---------------------------------------------------------------------------
# Dimension 4 — GitHub Activity
# ---------------------------------------------------------------------------

def _score_github_activity(candidate: dict) -> dict:
    """
    Composite GitHub activity score (0–100).

    Sub-signals and their contribution ceilings:
    - commits_last_year : up to 50 pts  (saturates at 500 commits/yr)
    - total_stars       : up to 30 pts  (saturates at 200 stars)
    - public_repos_count: up to 20 pts  (saturates at 20 repos)

    Returns score = 0.0 and available=False when no GitHub data is present.
    """
    gh = candidate.get("github")

    if not gh or not isinstance(gh, dict):
        return {
            "score":     0.0,
            "available": False,
            "note":      "No GitHub data in profile. Run github_fetch and merge first.",
        }

    commits     = gh.get("commits_last_year", 0) or 0
    stars       = gh.get("total_stars", 0) or 0
    repos       = gh.get("public_repos_count", 0) or 0

    commit_score = min(commits / 500.0, 1.0) * 50.0
    star_score   = min(stars   / 200.0, 1.0) * 30.0
    repo_score   = min(repos   / 20.0,  1.0) * 20.0

    score_0_100  = round(commit_score + star_score + repo_score, 2)

    return {
        "score":             score_0_100,
        "available":         True,
        "commits_last_year": commits,
        "total_stars":       stars,
        "public_repos_count": repos,
        "commit_score":      round(commit_score, 2),
        "star_score":        round(star_score, 2),
        "repo_score":        round(repo_score, 2),
    }


# ---------------------------------------------------------------------------
# Grade assignment
# ---------------------------------------------------------------------------

def _assign_grade(score: float) -> str:
    for threshold, grade in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "D"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_profile(
    candidate: dict,
    role_name: str,
    roles_path: Path = _ROLES_PATH,
    dim_weights: dict | None = None,
) -> dict:
    """
    Score a candidate profile against a target job role.

    Parameters
    ----------
    candidate : dict
        Output of ``parse_resume.parse_resume()``, optionally merged with
        ``github_fetch.merge_github_into_profile()``.
    role_name : str
        Must match a key in ``data/roles.json`` (case-sensitive).
        Call ``list_available_roles()`` to enumerate valid values.
    roles_path : Path
        Override path to roles.json (useful for testing).
    dim_weights : dict, optional
        Override dimension weights. Must be a dict with the same four keys
        as ``_DIM_WEIGHTS`` and values that sum to 1.0.

    Returns
    -------
    dict
        Full scoring breakdown::

            {
                "candidate_name": str,
                "role":           str,
                "total_score":    float,   # 0–100
                "grade":          str,     # A+, A, B+, …, D
                "dimension_weights": dict,
                "breakdown": {
                    "skill_match":          { score, weight, weighted_contribution, … },
                    "experience_relevance": { score, weight, weighted_contribution, … },
                    "project_depth":        { score, weight, weighted_contribution, … },
                    "github_activity":      { score, weight, weighted_contribution, … },
                }
            }

    Raises
    ------
    ValueError
        If ``role_name`` is not found in roles.json.
    FileNotFoundError
        If roles.json cannot be located.

    Example
    -------
    >>> result = score_profile(candidate, "ML Engineer")
    >>> print(result["total_score"], result["grade"])
    78.4  B+
    """
    weights = dim_weights or _DIM_WEIGHTS

    # Validate weights
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-6):
        raise ValueError(
            f"dim_weights values must sum to 1.0 (got {sum(weights.values()):.4f})"
        )

    # Load role requirements
    roles = load_roles(roles_path)
    if role_name not in roles:
        available = ", ".join(sorted(roles.keys()))
        raise ValueError(
            f"Role '{role_name}' not found in roles.json. "
            f"Available roles: {available}"
        )
    role_requirements = roles[role_name]

    candidate_skills = candidate.get("skills", []) or []
    logger.info(
        "Scoring '%s' (%d skills) against role '%s' (%d required skills)…",
        candidate.get("name", "Unknown"),
        len(candidate_skills),
        role_name,
        len(role_requirements),
    )

    # --- Dimension 1: Skill Match ---
    model = _get_model()
    skill_detail = _score_skill_match(candidate_skills, role_requirements, model)

    # --- Dimension 2: Experience ---
    exp_detail = _score_experience(candidate)

    # --- Dimension 3: Project Depth ---
    proj_detail = _score_project_depth(candidate)

    # --- Dimension 4: GitHub Activity ---
    gh_detail = _score_github_activity(candidate)

    # --- Combine ---
    dims = {
        "skill_match":          skill_detail,
        "experience_relevance": exp_detail,
        "project_depth":        proj_detail,
        "github_activity":      gh_detail,
    }

    total_score = 0.0
    breakdown   = {}
    for dim_key, detail in dims.items():
        w   = weights[dim_key]
        s   = detail["score"]
        wc  = round(s * w, 4)
        total_score += wc
        breakdown[dim_key] = {
            **detail,
            "weight":                round(w, 4),
            "weighted_contribution": wc,
        }

    total_score = round(min(max(total_score, 0.0), 100.0), 2)
    grade       = _assign_grade(total_score)

    logger.info(
        "Score for '%s' → '%s': %.1f / 100  [%s]",
        candidate.get("name", "Unknown"),
        role_name,
        total_score,
        grade,
    )

    return {
        "candidate_name":     candidate.get("name", "Unknown"),
        "role":               role_name,
        "total_score":        total_score,
        "grade":              grade,
        "dimension_weights":  {k: round(v, 4) for k, v in weights.items()},
        "breakdown":          breakdown,
    }


def score_against_all_roles(
    candidate: dict,
    roles_path: Path = _ROLES_PATH,
) -> list[dict]:
    """
    Score a candidate against every role in roles.json and return a list
    sorted by total_score descending — useful for role recommendation.

    Returns
    -------
    list[dict]
        Each entry: { role, total_score, grade, breakdown }
    """
    roles = load_roles(roles_path)
    model = _get_model()   # load once before the loop

    results = []
    for role_name in roles:
        try:
            result = score_profile(candidate, role_name, roles_path=roles_path)
            results.append(result)
        except Exception as exc:
            logger.warning("Skipping role '%s': %s", role_name, exc)

    return sorted(results, key=lambda r: r["total_score"], reverse=True)
