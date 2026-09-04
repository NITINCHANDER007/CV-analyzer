"""
app.py
------
Streamlit UI for the SMARRTIF AI — CV Analyser.

Full pipeline:
    Upload resume  →  Parse (Gemini)  →  [GitHub fetch]  →  Score  →  Recommend

Run
---
    streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env file into os.environ
load_dotenv()

import streamlit as st
import plotly.graph_objects as go

# ── path fix so imports work when launched from project root ──────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parser.parse_resume import parse_resume
from parser.github_fetch import fetch_github_profile, merge_github_into_profile, GitHubAPIError
from scoring.score_profile import score_profile, list_available_roles
from recommender.recommend import generate_recommendations, recommendations_summary

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMARRTIF AI — CV Analyser",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Base ────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f0f23 50%, #0a0a18 100%);
    color: #e2e8f0;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13132b 0%, #0d0d1f 100%);
    border-right: 1px solid rgba(124, 58, 237, 0.2);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    color: #a0aec0;
}

/* ── Header ──────────────────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #1a0533 0%, #0f1a3d 100%);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center,
        rgba(124, 58, 237, 0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}

/* ── Cards ───────────────────────────────────────────────────────── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(124, 58, 237, 0.35); }

.card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 0.75rem;
}

/* ── Score display ───────────────────────────────────────────────── */
.score-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.5rem 0;
}
.score-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.score-denom {
    font-size: 1.5rem;
    color: #4a5568;
    font-weight: 400;
}
.grade-badge {
    display: inline-block;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    padding: 0.3rem 1.2rem;
    border-radius: 99px;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}
.score-label {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 0.4rem;
}

/* ── Gap pills ───────────────────────────────────────────────────── */
.gap-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem 0.2rem;
}
.pill-missing { background: rgba(239,68,68,0.12); color: #f87171;
                border: 1px solid rgba(239,68,68,0.3); }
.pill-weak    { background: rgba(251,191,36,0.10); color: #fbbf24;
                border: 1px solid rgba(251,191,36,0.3); }

/* ── Recommendation cards ────────────────────────────────────────── */
.rec-card {
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid;
    background: rgba(255,255,255,0.025);
}
.rec-card.p1 { border-color: #ef4444; }
.rec-card.p2 { border-color: #f59e0b; }
.rec-card.p3 { border-color: #10b981; }
.rec-skill {
    font-weight: 600;
    font-size: 0.95rem;
    color: #e2e8f0;
    margin-bottom: 0.15rem;
}
.rec-service {
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.rec-service span { padding: 0.1rem 0.5rem; border-radius: 4px;
                    font-size: 0.72rem; }
.stype-course     { background: rgba(99,102,241,0.15); color: #818cf8; }
.stype-mentorship { background: rgba(16,185,129,0.15); color: #34d399; }
.stype-project    { background: rgba(245,158,11,0.15); color: #fbbf24; }
.rec-reason { font-size: 0.8rem; color: #718096; line-height: 1.5; }

/* ── Dividers & section headers ──────────────────────────────────── */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1.5rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Streamlit overrides ─────────────────────────────────────────── */
.stFileUploader, .stSelectbox, .stTextInput {
    border-radius: 10px !important;
}
div[data-testid="stFileUploaderDropzone"] {
    background: rgba(124,58,237,0.05);
    border: 1.5px dashed rgba(124,58,237,0.4) !important;
    border-radius: 12px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stMetricValue"] {
    color: #a78bfa !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
}

/* ── Alert box ───────────────────────────────────────────────────── */
.info-box {
    background: rgba(37,99,235,0.08);
    border: 1px solid rgba(37,99,235,0.25);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #93c5fd;
    margin-bottom: 1rem;
}
.warn-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #fcd34d;
    margin-bottom: 1rem;
}
.err-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #fca5a5;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grade_color(grade: str) -> str:
    return {
        "A+": "#34d399", "A": "#34d399",
        "B+": "#60a5fa", "B": "#60a5fa",
        "C+": "#fbbf24", "C": "#fbbf24",
        "D+": "#f87171", "D": "#f87171",
    }.get(grade, "#94a3b8")


def _score_label(score: float) -> str:
    if score >= 87: return "Excellent match 🚀"
    if score >= 73: return "Strong candidate 💪"
    if score >= 58: return "Moderate match 📈"
    if score >= 40: return "Needs development 🌱"
    return "Significant gaps ⚠️"


def _service_type_class(stype: str) -> str:
    return {"course": "stype-course", "mentorship": "stype-mentorship",
            "project": "stype-project", "tool": "stype-project"}.get(stype, "stype-course")


def _priority_class(priority: int) -> str:
    return {1: "p1", 2: "p2", 3: "p3"}.get(priority, "p3")


def _service_type_icon(stype: str) -> str:
    return {"course": "📚", "mentorship": "🤝", "project": "🚀", "tool": "🛠️"}.get(stype, "📌")


def _build_dimension_chart(breakdown: dict) -> go.Figure:
    """Horizontal bar chart of the 4 scoring dimensions."""
    labels = {
        "skill_match":          "Skill Match",
        "experience_relevance": "Experience",
        "project_depth":        "Project Depth",
        "github_activity":      "GitHub Activity",
    }
    weights = {
        "skill_match": 0.50, "experience_relevance": 0.25,
        "project_depth": 0.15, "github_activity": 0.10,
    }
    colors = ["#7c3aed", "#2563eb", "#0891b2", "#059669"]

    dims   = list(labels.keys())
    scores = [breakdown.get(d, {}).get("score", 0) for d in dims]
    names  = [labels[d] for d in dims]
    wts    = [f"{weights[d]:.0%}" for d in dims]

    fig = go.Figure()

    # Background track (100%)
    fig.add_trace(go.Bar(
        x=[100] * 4,
        y=names,
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.04)", line=dict(width=0)),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Actual scores
    for i, (name, score, color, wt) in enumerate(zip(names, scores, colors, wts)):
        fig.add_trace(go.Bar(
            x=[score],
            y=[name],
            orientation="h",
            marker=dict(
                color=color,
                opacity=0.85,
                line=dict(width=0),
            ),
            name=name,
            text=f"<b>{score:.1f}</b>",
            textposition="inside",
            textfont=dict(color="white", size=13, family="Space Grotesk"),
            customdata=[[wt, score]],
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Score: {score:.1f} / 100<br>"
                f"Weight: {wt}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=30, t=10, b=10),
        height=220,
        xaxis=dict(
            range=[0, 110],
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#4a5568", size=11),
            showticklabels=True,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#a0aec0", size=12, family="Inter"),
            categoryorder="array",
            categoryarray=list(reversed(names)),
        ),
        font=dict(family="Inter"),
        hoverlabel=dict(
            bgcolor="#1a1a2e",
            bordercolor="#7c3aed",
            font=dict(color="white", family="Inter"),
        ),
    )
    return fig


def _build_radar_chart(breakdown: dict) -> go.Figure:
    """Radar / spider chart alternative view."""
    labels = ["Skill Match", "Experience", "Project Depth", "GitHub Activity"]
    keys   = ["skill_match", "experience_relevance", "project_depth", "github_activity"]
    values = [breakdown.get(k, {}).get("score", 0) for k in keys]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(124,58,237,0.15)",
        line=dict(color="#7c3aed", width=2),
        marker=dict(color="#a78bfa", size=7),
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(color="#4a5568", size=10),
                gridcolor="rgba(255,255,255,0.07)",
                linecolor="rgba(255,255,255,0.07)",
            ),
            angularaxis=dict(
                tickfont=dict(color="#94a3b8", size=11, family="Inter"),
                gridcolor="rgba(255,255,255,0.07)",
                linecolor="rgba(255,255,255,0.1)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=30, b=30),
        height=280,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#1a1a2e",
            bordercolor="#7c3aed",
            font=dict(color="white", family="Inter"),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    file_bytes: bytes,
    file_name: str,
    github_username: str,
    role_name: str,
    gemini_key: str,
    github_token: str,
) -> tuple[dict, dict, list[dict], list[str]]:
    """
    Run the full pipeline and return (candidate, scoring, recommendations, log).

    Cached by Streamlit so identical inputs don't reprocess.
    """
    log: list[str] = []

    # ── 1. Save uploaded bytes to a temp file ─────────────────────────────
    suffix = Path(file_name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        # ── 2. Parse resume ──────────────────────────────────────────────
        log.append("📄 Parsing resume…")
        candidate = parse_resume(tmp_path, api_key=gemini_key)
        log.append(f"✅ Parsed: {candidate.get('name', 'Unknown')}")

        # ── 3. GitHub fetch (optional) ───────────────────────────────────
        if github_username.strip():
            log.append(f"🐙 Fetching GitHub profile for @{github_username}…")
            try:
                gh_data = fetch_github_profile(
                    github_username.strip(),
                    token=github_token or None,
                )
                candidate = merge_github_into_profile(candidate, gh_data)
                commits = gh_data["github"].get("commits_last_year", 0)
                stars   = gh_data["github"].get("total_stars", 0)
                log.append(f"✅ GitHub: {commits} commits/yr  ⭐ {stars} stars")
            except GitHubAPIError as e:
                log.append(f"⚠️ GitHub fetch failed: {e}. Continuing without GitHub data.")
            except Exception as e:
                log.append(f"⚠️ GitHub error: {e}. Continuing without GitHub data.")

        # ── 4. Score ─────────────────────────────────────────────────────
        log.append(f"📊 Scoring against '{role_name}'…")
        scoring = score_profile(candidate, role_name)
        log.append(f"✅ Score: {scoring['total_score']:.1f}/100  [{scoring['grade']}]")

        # ── 5. Recommend ─────────────────────────────────────────────────
        log.append("💡 Generating recommendations…")
        recommendations = generate_recommendations(candidate, role_name, scoring)
        log.append(f"✅ {len(recommendations)} recommendation(s) generated.")

    finally:
        tmp_path.unlink(missing_ok=True)

    return candidate, scoring, recommendations, log


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔑 API Keys")
    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        placeholder="AIza…",
        help="Required. Get yours at https://aistudio.google.com/",
        key="gemini_key_input",
    )
    github_token = st.text_input(
        "GitHub Token *(optional)*",
        type="password",
        value=os.environ.get("GITHUB_TOKEN", ""),
        placeholder="ghp_…",
        help="Personal Access Token for higher rate limits and commit-count access.",
        key="github_token_input",
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
<div style="font-size:0.82rem; color:#718096; line-height:1.6;">
<b>SMARRTIF AI — CV Analyser</b><br>
Upload a resume → get a skill-gap analysis and personalised learning recommendations.<br><br>
Powered by:<br>
• <b>Gemini</b> for resume parsing<br>
• <b>Sentence Transformers</b> for skill matching<br>
• <b>GitHub REST API</b> for activity scoring
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style="font-size:0.75rem;color:#4a5568;text-align:center;">
    SMARRTIF AI &nbsp;•&nbsp; v1.0.0</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <p class="hero-title">🧠 SMARRTIF AI — CV Analyser</p>
    <p class="hero-sub">
        Upload a resume, select a target role, and get an AI-powered score with
        personalised skill-gap recommendations in seconds.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Input section
# ─────────────────────────────────────────────────────────────────────────────

col_a, col_b, col_c = st.columns([2, 1.2, 1.2], gap="large")

with col_a:
    st.markdown('<p class="section-label">📎 Resume</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a PDF or DOCX resume",
        type=["pdf", "docx"],
        label_visibility="collapsed",
        key="resume_uploader",
    )
    if uploaded_file:
        st.markdown(
            f'<div class="info-box">📄 <b>{uploaded_file.name}</b> — '
            f'{uploaded_file.size / 1024:.1f} KB</div>',
            unsafe_allow_html=True,
        )

with col_b:
    st.markdown('<p class="section-label">🐙 GitHub</p>', unsafe_allow_html=True)
    github_username = st.text_input(
        "GitHub username",
        placeholder="e.g. torvalds",
        label_visibility="collapsed",
        key="github_username_input",
    )
    if github_username and not github_token:
        st.markdown(
            '<div class="warn-box">⚠️ No GitHub token — commit counts may be limited.</div>',
            unsafe_allow_html=True,
        )

with col_c:
    st.markdown('<p class="section-label">🎯 Target Role</p>', unsafe_allow_html=True)
    try:
        available_roles = list_available_roles()
    except Exception:
        available_roles = ["ML Engineer", "Data Analyst", "BI Developer",
                           "Backend Developer", "AI Product Associate"]
    selected_role = st.selectbox(
        "Target role",
        options=available_roles,
        label_visibility="collapsed",
        key="role_selector",
    )

st.markdown("<br>", unsafe_allow_html=True)

# Validation
can_submit = bool(uploaded_file and gemini_key and selected_role)
if not gemini_key:
    st.markdown(
        '<div class="warn-box">🔑 Add your Gemini API key in the sidebar to proceed.</div>',
        unsafe_allow_html=True,
    )

analyse_btn = st.button(
    "🚀  Analyse Resume",
    disabled=not can_submit,
    key="analyse_btn",
    use_container_width=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────

if analyse_btn and can_submit:
    st.session_state["results"] = None   # clear previous results

    # Progress stages
    stages = [
        ("📄 Parsing resume with Gemini…",          0.20),
        ("🐙 Fetching GitHub profile…",              0.45),
        ("📊 Running scoring engine…",               0.70),
        ("💡 Generating recommendations…",           0.90),
        ("✅ Done!",                                 1.00),
    ]

    prog_bar = st.progress(0.0)
    status   = st.empty()

    with st.spinner(""):
        for msg, pct in stages:
            status.markdown(
                f'<div class="info-box">{msg}</div>', unsafe_allow_html=True
            )
            prog_bar.progress(pct)
            if pct < 1.0:
                time.sleep(0.3)   # UI feedback before blocking on real work

        try:
            candidate, scoring, recommendations, log = run_pipeline(
                file_bytes      = uploaded_file.getvalue(),
                file_name       = uploaded_file.name,
                github_username = github_username or "",
                role_name       = selected_role,
                gemini_key      = gemini_key,
                github_token    = github_token or "",
            )
            st.session_state["results"] = (candidate, scoring, recommendations, log)

        except Exception as e:
            import traceback
            st.session_state["pipeline_error"] = traceback.format_exc()

    prog_bar.empty()
    status.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Error display
# ─────────────────────────────────────────────────────────────────────────────

import traceback

if "pipeline_error" in st.session_state and st.session_state["pipeline_error"]:
    st.markdown(
        f'<div class="err-box">❌ <b>Error:</b><pre>{st.session_state["pipeline_error"]}</pre></div>',
        unsafe_allow_html=True,
    )
    st.session_state["pipeline_error"] = None


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.get("results"):
    candidate, scoring, recommendations, log = st.session_state["results"]
    bd = scoring["breakdown"]
    summary = recommendations_summary(recommendations)

    st.markdown("---")
    st.markdown('<p class="section-label">📊 Analysis Results</p>', unsafe_allow_html=True)

    # ── Pipeline log (expander) ───────────────────────────────────────────
    with st.expander("🔍 Pipeline log", expanded=False):
        for line in log:
            st.markdown(f"`{line}`")

    # ── Row 1: Score card + candidate info + chart ────────────────────────
    r1_left, r1_mid, r1_right = st.columns([1, 1.6, 2], gap="large")

    with r1_left:
        grade        = scoring["grade"]
        total        = scoring["total_score"]
        grade_col    = _grade_color(grade)
        score_lbl    = _score_label(total)
        candidate_nm = candidate.get("name", "Candidate")
        target_role  = scoring["role"]

        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="card-title">Overall Score</div>
            <div class="score-ring-wrap">
                <div>
                    <span class="score-number">{total:.1f}</span>
                    <span class="score-denom"> / 100</span>
                </div>
                <div class="grade-badge"
                     style="background:rgba(0,0,0,0.25);
                            border:2px solid {grade_col};
                            color:{grade_col};">
                    {grade}
                </div>
                <div class="score-label">{score_lbl}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r1_mid:
        exp_bd  = bd.get("experience_relevance", {})
        proj_bd = bd.get("project_depth", {})
        gh_bd   = bd.get("github_activity", {})
        sm_bd   = bd.get("skill_match", {})

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Candidate</div>
            <p style="font-size:1.1rem;font-weight:700;margin:0 0 0.3rem 0;
                      color:#e2e8f0;">{candidate_nm}</p>
            <p style="color:#718096;font-size:0.85rem;margin:0 0 1rem 0;">
                🎯 Target: <b style="color:#a78bfa;">{target_role}</b>
            </p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;">
                <div style="background:rgba(124,58,237,0.08);border-radius:10px;
                            padding:0.6rem;text-align:center;">
                    <div style="font-size:1.4rem;font-weight:700;
                                color:#a78bfa;">{exp_bd.get("total_years",0):.0f}</div>
                    <div style="font-size:0.7rem;color:#718096;
                                text-transform:uppercase;">Yrs Exp</div>
                </div>
                <div style="background:rgba(37,99,235,0.08);border-radius:10px;
                            padding:0.6rem;text-align:center;">
                    <div style="font-size:1.4rem;font-weight:700;
                                color:#60a5fa;">{proj_bd.get("project_count",0)}</div>
                    <div style="font-size:0.7rem;color:#718096;
                                text-transform:uppercase;">Projects</div>
                </div>
                <div style="background:rgba(5,150,105,0.08);border-radius:10px;
                            padding:0.6rem;text-align:center;">
                    <div style="font-size:1.4rem;font-weight:700;
                                color:#34d399;">
                        {gh_bd.get("commits_last_year", "—") if gh_bd.get("available") else "—"}
                    </div>
                    <div style="font-size:0.7rem;color:#718096;
                                text-transform:uppercase;">Commits/yr</div>
                </div>
                <div style="background:rgba(8,145,178,0.08);border-radius:10px;
                            padding:0.6rem;text-align:center;">
                    <div style="font-size:1.4rem;font-weight:700;
                                color:#38bdf8;">{sm_bd.get("match_rate",0):.0f}%</div>
                    <div style="font-size:0.7rem;color:#718096;
                                text-transform:uppercase;">Skill Hit</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r1_right:
        st.markdown('<div class="card"><div class="card-title">Dimension Breakdown</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            _build_dimension_chart(bd),
            use_container_width=True,
            config={"displayModeBar": False},
            key="dim_bar_chart",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Radar chart (collapsible) ─────────────────────────────────────────
    with st.expander("📡 Radar view", expanded=False):
        st.plotly_chart(
            _build_radar_chart(bd),
            use_container_width=True,
            config={"displayModeBar": False},
            key="radar_chart",
        )

    # ── Dimension detail metrics ──────────────────────────────────────────
    st.markdown('<p class="section-label">📐 Dimension Detail</p>', unsafe_allow_html=True)
    dm1, dm2, dm3, dm4 = st.columns(4, gap="medium")

    def _dim_metric(col, title, detail_dict, color):
        score = detail_dict.get("score", 0)
        wc    = detail_dict.get("weighted_contribution", 0)
        w     = detail_dict.get("weight", 0)
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="card-title">{title}</div>
                <div style="font-size:2rem;font-weight:800;
                            color:{color};font-family:'Space Grotesk',sans-serif;">
                    {score:.1f}
                </div>
                <div style="font-size:0.75rem;color:#4a5568;margin-top:0.2rem;">
                    ×{w:.0%} weight = <b style="color:#718096;">{wc:.1f} pts</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    _dim_metric(dm1, "Skill Match",    bd.get("skill_match", {}),          "#a78bfa")
    _dim_metric(dm2, "Experience",     bd.get("experience_relevance", {}),  "#60a5fa")
    _dim_metric(dm3, "Project Depth",  bd.get("project_depth", {}),         "#38bdf8")
    _dim_metric(dm4, "GitHub Activity",bd.get("github_activity", {}),       "#34d399")

    # ── Row 2: Skill gaps | Recommendations ──────────────────────────────
    st.markdown('<p class="section-label">🔎 Skill Gaps & Recommendations</p>',
                unsafe_allow_html=True)

    gap_col, rec_col = st.columns([1, 1.6], gap="large")

    with gap_col:
        st.markdown('<div class="card"><div class="card-title">Skill Gap Map</div>',
                    unsafe_allow_html=True)
        sm = bd.get("skill_match", {})
        matched = sm.get("matched_skills", [])
        missing = sm.get("missing_skills", [])
        per_skill = sm.get("per_skill_detail", [])

        # Weak = matched but low similarity
        weak_skills = [
            d["required_skill"] for d in per_skill
            if d.get("matched") and d.get("similarity", 1) < 0.65
        ]

        if matched:
            st.markdown("**✅ Covered skills**")
            covered_html = "".join(
                f'<span class="gap-pill" style="background:rgba(16,185,129,0.1);'
                f'color:#34d399;border:1px solid rgba(16,185,129,0.3);">'
                f'{m["required"]}</span>'
                for m in matched if m["required"] not in weak_skills
            )
            st.markdown(covered_html or "*(none)*", unsafe_allow_html=True)

        if weak_skills:
            st.markdown("**⚠️ Weak coverage**")
            weak_html = "".join(
                f'<span class="gap-pill pill-weak">{s}</span>' for s in weak_skills
            )
            st.markdown(weak_html, unsafe_allow_html=True)

        if missing:
            st.markdown("**❌ Missing skills**")
            miss_html = "".join(
                f'<span class="gap-pill pill-missing">{s}</span>' for s in missing
            )
            st.markdown(miss_html, unsafe_allow_html=True)

        if not missing and not weak_skills:
            st.success("No significant skill gaps detected!")

        # Summary bar
        total_req = len(per_skill) or 1
        matched_count = len(matched)
        st.markdown(
            f"""<div style="margin-top:1rem;padding:0.75rem;
                            background:rgba(124,58,237,0.06);border-radius:10px;
                            font-size:0.82rem;color:#94a3b8;">
                ✅ {matched_count} matched &nbsp;|&nbsp;
                ⚠️ {len(weak_skills)} weak &nbsp;|&nbsp;
                ❌ {len(missing)} missing &nbsp;|&nbsp;
                📋 {total_req} total required
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with rec_col:
        st.markdown('<div class="card"><div class="card-title">Recommended Services</div>',
                    unsafe_allow_html=True)

        if not recommendations:
            st.markdown(
                '<div class="info-box">🎉 No significant gaps — candidate is a strong fit '
                'for this role!</div>',
                unsafe_allow_html=True,
            )
        else:
            # Summary bar
            crit = summary["critical_count"]
            miss_c = summary["missing_count"]
            weak_c = summary["weak_count"]
            st.markdown(
                f"""<div style="font-size:0.82rem;color:#94a3b8;
                                margin-bottom:0.8rem;">
                    🔴 {crit} critical &nbsp;|&nbsp; ❌ {miss_c} missing &nbsp;|&nbsp;
                    ⚠️ {weak_c} weak
                </div>""",
                unsafe_allow_html=True,
            )

            for rec in recommendations:
                p_cls  = _priority_class(rec["priority"])
                s_cls  = _service_type_class(rec["service_type"])
                s_icon = _service_type_icon(rec["service_type"])
                sim    = rec["similarity_score"]
                imp_lbl = rec["importance_label"]
                gtype  = rec["gap_type"]
                g_icon = "❌" if gtype == "missing" else "⚠️"



                st.markdown(f"""
                <div class="rec-card {p_cls}">
                    <div class="rec-skill">
                        {g_icon} {rec['gap_skill']}
                        <span style="font-size:0.72rem;color:#4a5568;font-weight:400;">
                            — {imp_lbl} &nbsp;·&nbsp; sim {sim:.0%}
                        </span>
                    </div>
                    <div class="rec-service">
                        {s_icon}&nbsp;
                        <b style="color:#e2e8f0;">{rec['recommended_service']}</b>
                        &nbsp;<span class="{s_cls}">{rec['service_type']}</span>
                    </div>
                    <div class="rec-reason">{rec['reason']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Candidate profile expander ────────────────────────────────────────
    with st.expander("🗂️ Full parsed candidate profile", expanded=False):
        import json
        st.json(candidate)

    # ── GitHub data expander ──────────────────────────────────────────────
    if candidate.get("github"):
        with st.expander("🐙 GitHub data", expanded=False):
            gh = candidate["github"]
            g1, g2, g3 = st.columns(3)
            g1.metric("Commits / yr", gh.get("commits_last_year", "—"))
            g2.metric("Total ⭐ Stars", gh.get("total_stars", "—"))
            g3.metric("Public Repos", gh.get("public_repos_count", "—"))

            top_langs = gh.get("top_languages", [])
            if top_langs:
                st.markdown(
                    "**Top languages:** " +
                    " &nbsp;".join(
                        f'<span class="gap-pill" style="background:rgba(99,102,241,0.12);'
                        f'color:#818cf8;border:1px solid rgba(99,102,241,0.3);">{l}</span>'
                        for l in top_langs[:8]
                    ),
                    unsafe_allow_html=True,
                )

else:
    # ── Empty state ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#4a5568;">
        <div style="font-size:3.5rem;margin-bottom:1rem;">📋</div>
        <p style="font-size:1.1rem;font-weight:500;color:#718096;">
            Upload a resume and click <b style="color:#a78bfa;">Analyse Resume</b> to begin.
        </p>
        <p style="font-size:0.85rem;margin-top:0.5rem;">
            Supports <b>.pdf</b> and <b>.docx</b> formats
        </p>
    </div>
    """, unsafe_allow_html=True)
