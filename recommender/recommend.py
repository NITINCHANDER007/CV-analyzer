"""
recommend.py
------------
Gap analysis and learning recommendation engine for the AI CV Analyzer.

Given a candidate profile, target role, and a scoring breakdown (from
score_profile.py), this module:

1. Identifies MISSING skills — role requirements with cosine similarity
   below the "match" threshold.
2. Identifies WEAK skills — role requirements that are technically matched
   but with a similarity score that indicates superficial coverage
   (similarity in the "weak" band) or a high importance weight (weight 3)
   despite a mediocre similarity score.
3. Maps each gap to a SMARRTIF AI service / course category via a two-layer
   lookup: exact keyword match first, then sentence-transformer semantic
   fallback so novel skill names are always handled.
4. Returns a ranked, deduplicated list of recommendations:

    [
        {
            "gap_skill":              str,
            "gap_type":               "missing" | "weak",
            "importance":             int,        # 1-3 from roles.json
            "similarity_score":       float,      # cosine sim (0-1)
            "recommended_service":    str,        # SMARRTIF category name
            "service_type":           str,        # "course" | "mentorship" | "tool" | "project"
            "reason":                 str,        # human-readable rationale
            "priority":               int,        # 1 (critical) → 3 (optional)
        }
    ]

Usage
~~~~~
    from recommender.recommend import generate_recommendations

    recs = generate_recommendations(candidate, "ML Engineer", scoring_result)
    for r in recs:
        print(r["gap_skill"], "→", r["recommended_service"])
"""

from __future__ import annotations

import logging
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
logger = logging.getLogger("recommend")

# ---------------------------------------------------------------------------
# Thresholds (mirror / reference score_profile.py constants)
# ---------------------------------------------------------------------------
_MISSING_THRESHOLD = 0.40   # below this → skill is "missing"
_WEAK_THRESHOLD    = 0.65   # 0.40–0.65  → skill is "weak"
_STRONG_THRESHOLD  = 0.65   # ≥ this     → adequately covered (no gap)

# Minimum importance weight for a weak skill to generate a recommendation
# (weight-1 skills that are weak are usually optional, skip unless missing)
_WEAK_MIN_WEIGHT   = 2

# ---------------------------------------------------------------------------
# SMARRTIF AI Service Catalogue
# Each entry: service_name, service_type, keywords (for fast lookup),
#             semantic_anchor (sentence embedded for fallback matching)
# ---------------------------------------------------------------------------
_SERVICE_CATALOGUE: list[dict] = [
    # ── Data & Analytics ─────────────────────────────────────────────────
    {
        "service":        "SQL & Database Mastery",
        "service_type":   "course",
        "keywords":       ["sql", "postgresql", "mysql", "database", "query", "relational"],
        "anchor":         "SQL relational database querying and data manipulation",
    },
    {
        "service":        "Data Analysis Bootcamp",
        "service_type":   "course",
        "keywords":       ["data analysis", "data analyst", "data cleaning", "data wrangling",
                           "pandas", "numpy", "excel", "google sheets", "spreadsheet",
                           "business acumen"],
        "anchor":         "data analysis data wrangling pandas statistical analysis",
    },
    {
        "service":        "Statistics & Probability Fundamentals",
        "service_type":   "course",
        "keywords":       ["statistics", "probability", "hypothesis testing", "a/b testing",
                           "statistical inference", "regression", "correlation"],
        "anchor":         "statistics probability hypothesis testing A/B testing experiments",
    },
    {
        "service":        "Data Visualization Masterclass",
        "service_type":   "course",
        "keywords":       ["data visualization", "visualisation", "tableau", "power bi",
                           "dashboard", "charts", "plots", "matplotlib", "seaborn",
                           "data viz"],
        "anchor":         "data visualization dashboards Tableau Power BI charts reporting",
    },
    {
        "service":        "Business Intelligence & Reporting",
        "service_type":   "course",
        "keywords":       ["bi developer", "business intelligence", "dax", "data modeling",
                           "data model", "etl", "data warehouse", "data warehousing",
                           "azure synapse", "azure fabric", "microsoft fabric",
                           "stakeholder communication", "reporting"],
        "anchor":         "business intelligence BI data modeling ETL data warehouse DAX Power BI",
    },

    # ── Machine Learning & AI ────────────────────────────────────────────
    {
        "service":        "ML Foundations Course",
        "service_type":   "course",
        "keywords":       ["machine learning", "ml", "scikit-learn", "sklearn",
                           "supervised learning", "unsupervised learning", "classification",
                           "regression", "clustering", "feature engineering",
                           "ai / ml fundamentals", "ai ml"],
        "anchor":         "machine learning supervised unsupervised algorithms scikit-learn",
    },
    {
        "service":        "Deep Learning & Neural Networks",
        "service_type":   "course",
        "keywords":       ["deep learning", "neural network", "pytorch", "tensorflow",
                           "keras", "cnn", "rnn", "lstm", "transformer", "attention",
                           "backpropagation"],
        "anchor":         "deep learning neural networks PyTorch TensorFlow convolutional recurrent",
    },
    {
        "service":        "MLOps & Model Deployment Workshop",
        "service_type":   "course",
        "keywords":       ["mlops", "model deployment", "model serving", "mlflow",
                           "kubeflow", "bentoml", "triton", "model registry",
                           "ci/cd for ml", "monitoring", "drift"],
        "anchor":         "MLOps model deployment serving monitoring CI/CD machine learning pipelines",
    },
    {
        "service":        "LLM & Generative AI Programme",
        "service_type":   "course",
        "keywords":       ["llm", "large language model", "generative ai", "gpt",
                           "prompt engineering", "langchain", "rag", "fine-tuning",
                           "hugging face", "transformers", "openai"],
        "anchor":         "large language models prompt engineering generative AI RAG fine-tuning",
    },

    # ── Software Engineering & DevOps ────────────────────────────────────
    {
        "service":        "Python Engineering Track",
        "service_type":   "course",
        "keywords":       ["python", "python programming", "object-oriented", "oop",
                           "packaging", "testing", "pytest"],
        "anchor":         "Python programming software engineering best practices",
    },
    {
        "service":        "Backend & API Development",
        "service_type":   "course",
        "keywords":       ["rest api", "api design", "restful", "fastapi", "django",
                           "flask", "backend", "http", "microservices",
                           "fastapi / django"],
        "anchor":         "REST API design backend web development FastAPI Django Flask",
    },
    {
        "service":        "System Design & Architecture Mentorship",
        "service_type":   "mentorship",
        "keywords":       ["system design", "architecture", "scalability", "distributed",
                           "high availability", "load balancing"],
        "anchor":         "system design distributed systems scalability architecture",
    },
    {
        "service":        "Cloud & DevOps Accelerator",
        "service_type":   "course",
        "keywords":       ["cloud", "aws", "gcp", "azure", "docker", "kubernetes",
                           "k8s", "container", "terraform", "iac",
                           "cloud platforms (aws/gcp)", "cloud platforms"],
        "anchor":         "cloud computing AWS GCP Azure Docker Kubernetes DevOps",
    },
    {
        "service":        "Message Queues & Event Streaming",
        "service_type":   "course",
        "keywords":       ["kafka", "rabbitmq", "message queue", "pubsub", "pub/sub",
                           "event streaming", "celery", "message queues (kafka/rabbitmq)",
                           "redis", "redis / caching", "caching"],
        "anchor":         "message queues Kafka RabbitMQ event streaming Redis caching",
    },
    {
        "service":        "Git & Version Control Workshop",
        "service_type":   "course",
        "keywords":       ["git", "github", "gitlab", "version control", "branching",
                           "pull request", "code review"],
        "anchor":         "Git version control branching GitHub collaborative development",
    },

    # ── Product & Soft Skills ────────────────────────────────────────────
    {
        "service":        "AI Product Management Programme",
        "service_type":   "course",
        "keywords":       ["product management", "product manager", "product associate",
                           "roadmap", "roadmap planning", "product strategy",
                           "ai product associate"],
        "anchor":         "product management roadmap planning product strategy AI product",
    },
    {
        "service":        "Agile & Scrum Certification Prep",
        "service_type":   "course",
        "keywords":       ["agile", "scrum", "kanban", "sprint", "jira",
                           "agile / scrum"],
        "anchor":         "Agile Scrum Kanban sprint planning project management",
    },
    {
        "service":        "User Research & UX Methods",
        "service_type":   "course",
        "keywords":       ["user research", "ux research", "usability testing",
                           "interviews", "persona", "user journey", "ux"],
        "anchor":         "user research UX usability testing interviews personas",
    },
    {
        "service":        "Stakeholder Communication & Influence",
        "service_type":   "mentorship",
        "keywords":       ["stakeholder communication", "stakeholder management",
                           "communication", "presentation", "executive communication"],
        "anchor":         "stakeholder communication presentations influence business communication",
    },
    {
        "service":        "Competitive & Market Analysis",
        "service_type":   "course",
        "keywords":       ["competitive analysis", "market research", "competitor",
                           "market analysis", "swot"],
        "anchor":         "competitive analysis market research industry analysis SWOT",
    },

    # ── Hands-on / Project-based ─────────────────────────────────────────
    {
        "service":        "Capstone AI Project Lab",
        "service_type":   "project",
        "keywords":       [],   # semantic fallback only
        "anchor":         "hands-on AI project end-to-end machine learning application",
    },
    {
        "service":        "1-on-1 Career Mentorship Session",
        "service_type":   "mentorship",
        "keywords":       [],   # catch-all semantic fallback
        "anchor":         "career coaching mentorship professional development skill gap",
    },
]

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
_model: Optional[SentenceTransformer] = None
_service_embeddings: Optional[np.ndarray] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model for recommender…")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_service_embeddings(model: SentenceTransformer) -> np.ndarray:
    """Embed all service anchors once and cache them."""
    global _service_embeddings
    if _service_embeddings is None:
        anchors = [s["anchor"] for s in _SERVICE_CATALOGUE]
        _service_embeddings = model.encode(
            anchors, show_progress_bar=False, normalize_embeddings=True
        )
    return _service_embeddings


# ---------------------------------------------------------------------------
# Service lookup (keyword → semantic fallback)
# ---------------------------------------------------------------------------

def _map_skill_to_service(skill: str, model: SentenceTransformer) -> dict:
    """
    Return the best-matching service catalogue entry for a given skill name.

    Strategy
    --------
    1. Keyword scan (fast, deterministic): check if any keyword in each
       catalogue entry is a substring of the lowercased skill name or vice-versa.
       Return the first match.
    2. Semantic fallback: embed the skill name and find the highest-cosine-
       similarity service anchor. Always returns a result.
    """
    skill_lower = skill.lower()

    # Layer 1 — keyword scan
    for entry in _SERVICE_CATALOGUE:
        for kw in entry["keywords"]:
            if kw in skill_lower or skill_lower in kw:
                return entry

    # Layer 2 — semantic similarity
    skill_vec = model.encode([skill], show_progress_bar=False, normalize_embeddings=True)
    svc_vecs  = _get_service_embeddings(model)
    sims      = cosine_similarity(skill_vec, svc_vecs)[0]
    best_idx  = int(np.argmax(sims))
    logger.debug(
        "Semantic fallback for '%s' → '%s' (sim=%.3f)",
        skill, _SERVICE_CATALOGUE[best_idx]["service"], sims[best_idx]
    )
    return _SERVICE_CATALOGUE[best_idx]


# ---------------------------------------------------------------------------
# Gap identification
# ---------------------------------------------------------------------------

def _identify_gaps(scoring_breakdown: dict) -> list[dict]:
    """
    Extract missing and weak skills from the scoring breakdown.

    Returns a list of gap dicts:
    {
        required_skill, importance, gap_type, similarity_score,
        best_candidate_match
    }
    sorted by (importance DESC, similarity ASC) so the most critical,
    least-covered gaps come first.
    """
    skill_match = scoring_breakdown.get("breakdown", {}).get("skill_match", {})
    per_skill   = skill_match.get("per_skill_detail", [])

    if not per_skill:
        # Fallback: use the flat missing_skills list with unknown similarity
        missing = skill_match.get("missing_skills", [])
        return [
            {
                "required_skill":       s,
                "importance":           2,      # unknown → assume important
                "gap_type":             "missing",
                "similarity_score":     0.0,
                "best_candidate_match": None,
            }
            for s in missing
        ]

    gaps: list[dict] = []
    for detail in per_skill:
        sim    = detail.get("similarity", 0.0)
        weight = detail.get("weight", 1)
        skill  = detail.get("required_skill", "")

        if sim < _MISSING_THRESHOLD:
            gaps.append({
                "required_skill":       skill,
                "importance":           weight,
                "gap_type":             "missing",
                "similarity_score":     round(sim, 4),
                "best_candidate_match": detail.get("best_match"),
            })
        elif sim < _STRONG_THRESHOLD and weight >= _WEAK_MIN_WEIGHT:
            gaps.append({
                "required_skill":       skill,
                "importance":           weight,
                "gap_type":             "weak",
                "similarity_score":     round(sim, 4),
                "best_candidate_match": detail.get("best_match"),
            })

    # Sort: importance (desc), then similarity (asc — weakest first within same importance)
    gaps.sort(key=lambda g: (-g["importance"], g["similarity_score"]))
    return gaps


# ---------------------------------------------------------------------------
# Priority assignment
# ---------------------------------------------------------------------------

def _assign_priority(gap_type: str, importance: int) -> int:
    """
    Map (gap_type, importance) to a 1–3 recommendation priority.

    Priority 1 = critical (act now)
    Priority 2 = important (plan soon)
    Priority 3 = optional (nice improvement)
    """
    if gap_type == "missing" and importance == 3:
        return 1
    if gap_type == "missing" and importance == 2:
        return 1
    if gap_type == "missing" and importance == 1:
        return 2
    if gap_type == "weak" and importance == 3:
        return 2
    if gap_type == "weak" and importance == 2:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Reason builder
# ---------------------------------------------------------------------------

def _build_reason(
    gap: dict,
    service: dict,
    role_name: str,
    candidate_name: str,
) -> str:
    """Compose a natural-language reason string for the recommendation."""
    skill   = gap["required_skill"]
    gtype   = gap["gap_type"]
    sim     = gap["similarity_score"]
    imp     = gap["importance"]
    match   = gap.get("best_candidate_match")
    svc     = service["service"]
    stype   = service["service_type"]

    importance_label = {3: "critical", 2: "important", 1: "nice-to-have"}[imp]

    if gtype == "missing":
        match_clause = (
            f" The closest existing skill is '{match}' (similarity {sim:.0%}),"
            f" which is insufficient for this role."
            if match else ""
        )
        return (
            f"'{skill}' is a {importance_label} requirement for {role_name} "
            f"but is not present in {candidate_name}'s profile.{match_clause} "
            f"The '{svc}' {stype} directly targets this gap."
        )
    else:  # weak
        return (
            f"'{skill}' is {importance_label} for {role_name}. "
            f"{candidate_name} shows partial coverage via '{match}' "
            f"(similarity {sim:.0%}), but a stronger foundation is needed. "
            f"The '{svc}' {stype} will solidify this area."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_recommendations(
    candidate: dict,
    role_name: str,
    scoring_result: dict,
    max_recommendations: int = 10,
    include_weak: bool = True,
) -> list[dict]:
    """
    Generate a ranked list of skill-gap recommendations for a candidate.

    Parameters
    ----------
    candidate : dict
        Parsed candidate profile (output of ``parse_resume``/``merge_github``).
    role_name : str
        Target job role name (must match a key in roles.json).
    scoring_result : dict
        Full output of ``score_profile(candidate, role_name)`` — the
        ``breakdown.skill_match.per_skill_detail`` list is required.
    max_recommendations : int
        Cap on returned recommendations (default 10).
    include_weak : bool
        If False, only "missing" skills generate recommendations.

    Returns
    -------
    list[dict]
        Sorted by priority (ascending = most urgent first), each entry::

            {
                "gap_skill":           str,
                "gap_type":            "missing" | "weak",
                "importance":          int,          # 1-3
                "importance_label":    str,          # "critical" | "important" | "nice-to-have"
                "similarity_score":    float,        # 0.0 – 1.0
                "best_candidate_match": str | None,
                "recommended_service": str,
                "service_type":        str,          # "course"|"mentorship"|"tool"|"project"
                "reason":              str,
                "priority":            int,          # 1 (urgent) → 3 (optional)
                "priority_label":      str,          # "Critical" | "Important" | "Optional"
            }

    Example
    -------
    >>> recs = generate_recommendations(candidate, "ML Engineer", scoring)
    >>> for r in recs:
    ...     print(r["priority"], r["gap_skill"], "→", r["recommended_service"])
    1  MLOps / Model Deployment  →  MLOps & Model Deployment Workshop
    2  TensorFlow                →  Deep Learning & Neural Networks
    """
    candidate_name = candidate.get("name", "the candidate")
    model          = _get_model()

    # 1. Identify gaps from the scoring breakdown
    gaps = _identify_gaps(scoring_result)
    if not include_weak:
        gaps = [g for g in gaps if g["gap_type"] == "missing"]

    logger.info(
        "Found %d gap(s) for '%s' targeting '%s' (%d missing, %d weak).",
        len(gaps),
        candidate_name,
        role_name,
        sum(1 for g in gaps if g["gap_type"] == "missing"),
        sum(1 for g in gaps if g["gap_type"] == "weak"),
    )

    if not gaps:
        logger.info("No skill gaps detected — candidate is a strong match.")
        return []

    # 2. Map each gap to a service, build output records
    recommendations: list[dict] = []
    seen_services: set[str] = set()

    for gap in gaps:
        service  = _map_skill_to_service(gap["required_skill"], model)
        priority = _assign_priority(gap["gap_type"], gap["importance"])
        reason   = _build_reason(gap, service, role_name, candidate_name)

        svc_name = service["service"]

        # Deduplicate: if the same service already appears, skip if the
        # earlier entry was missing (higher priority). Allow if this gap is
        # a different skill pointing to the same course but it's critical.
        dedup_key = (svc_name, gap["gap_type"])
        if dedup_key in seen_services:
            continue
        seen_services.add(dedup_key)

        importance_labels = {3: "critical", 2: "important", 1: "nice-to-have"}
        priority_labels   = {1: "Critical", 2: "Important", 3: "Optional"}

        recommendations.append({
            "gap_skill":             gap["required_skill"],
            "gap_type":              gap["gap_type"],
            "importance":            gap["importance"],
            "importance_label":      importance_labels[gap["importance"]],
            "similarity_score":      gap["similarity_score"],
            "best_candidate_match":  gap.get("best_candidate_match"),
            "recommended_service":   svc_name,
            "service_type":          service["service_type"],
            "reason":                reason,
            "priority":              priority,
            "priority_label":        priority_labels[priority],
        })

        if len(recommendations) >= max_recommendations:
            break

    # 3. Final sort: priority asc → importance desc → sim asc
    recommendations.sort(
        key=lambda r: (r["priority"], -r["importance"], r["similarity_score"])
    )

    logger.info("Returning %d recommendation(s).", len(recommendations))
    return recommendations


def recommendations_summary(recommendations: list[dict]) -> dict:
    """
    Produce a compact summary dict from a recommendations list.

    Useful for the UI layer or for passing to Gemini as context.

    Returns
    -------
    dict::

        {
            "total_gaps":      int,
            "missing_count":   int,
            "weak_count":      int,
            "critical_count":  int,
            "by_priority": {
                "Critical":  [ {gap_skill, recommended_service, reason}, … ],
                "Important": [ … ],
                "Optional":  [ … ],
            }
        }
    """
    summary: dict = {
        "total_gaps":     len(recommendations),
        "missing_count":  sum(1 for r in recommendations if r["gap_type"] == "missing"),
        "weak_count":     sum(1 for r in recommendations if r["gap_type"] == "weak"),
        "critical_count": sum(1 for r in recommendations if r["priority"] == 1),
        "by_priority":    {"Critical": [], "Important": [], "Optional": []},
    }
    for rec in recommendations:
        label = rec["priority_label"]
        summary["by_priority"][label].append({
            "gap_skill":           rec["gap_skill"],
            "recommended_service": rec["recommended_service"],
            "service_type":        rec["service_type"],
            "reason":              rec["reason"],
        })
    return summary
