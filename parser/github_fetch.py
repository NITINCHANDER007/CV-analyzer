"""
github_fetch.py
---------------
Fetches a GitHub user's public profile data via the GitHub REST API and
returns a structured dict that can be merged into a candidate profile
produced by parse_resume.py.

Data collected
~~~~~~~~~~~~~~
- Public repositories (name, description, language, stars, forks, topics, URL)
- Primary language per repo
- Aggregated language statistics across all repos
- Total commit count authored in the last 365 days  (via Search Commits API)
- Per-repo star counts + total star count

Environment variable
~~~~~~~~~~~~~~~~~~~~
    GITHUB_TOKEN   — Personal Access Token with `public_repo` / read scope.
                     Required for higher rate limits (5 000 req/hr vs 60 req/hr).
                     Strongly recommended for the commit-search endpoint.

Usage
~~~~~
    from parser.github_fetch import fetch_github_profile

    profile = fetch_github_profile("torvalds")            # uses env var for token
    profile = fetch_github_profile("torvalds", token="ghp_xxx")
"""

from __future__ import annotations

import os
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("github_fetch")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_URL = "https://api.github.com"
_TIMEOUT  = 15          # seconds per HTTP request
_PER_PAGE = 100         # GitHub max page size for repos
_MAX_REPO_PAGES = 10    # safety cap: 1 000 repos max

# Accept header required by the Commits Search endpoint
_COMMIT_SEARCH_ACCEPT = "application/vnd.github+json"


# ---------------------------------------------------------------------------
# Low-level HTTP helper
# ---------------------------------------------------------------------------

class GitHubAPIError(Exception):
    """Raised when the GitHub API returns an error response."""
    def __init__(self, status_code: int, message: str, url: str = ""):
        super().__init__(f"GitHub API error {status_code} on {url!r}: {message}")
        self.status_code = status_code
        self.url = url


def _build_session(token: Optional[str]) -> requests.Session:
    """Return a requests.Session pre-configured with auth and headers."""
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "cv-analyser-github-fetch/1.0",
    })
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def _get(session: requests.Session, url: str, params: dict | None = None) -> dict | list:
    """
    Execute a GET request and handle common GitHub API responses.

    Automatically retries once on 202 (async computation) and raises
    GitHubAPIError on 4xx / 5xx responses.
    """
    for attempt in range(2):
        try:
            resp = session.get(url, params=params, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            raise GitHubAPIError(0, str(exc), url) from exc

        if resp.status_code == 202:
            # GitHub is computing stats asynchronously; wait and retry once
            logger.debug("Got 202 on %s — waiting 2s then retrying…", url)
            time.sleep(2)
            continue

        if resp.status_code == 204:
            return {}

        if not resp.ok:
            try:
                detail = resp.json().get("message", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            raise GitHubAPIError(resp.status_code, detail, url)

        return resp.json()

    raise GitHubAPIError(202, "GitHub async computation did not complete in time.", url)


def _paginate(session: requests.Session, url: str, params: dict | None = None) -> list:
    """
    Fetch all pages of a paginated GitHub endpoint.

    Returns a flat list of all items across pages.
    """
    params = dict(params or {})
    params["per_page"] = _PER_PAGE
    all_items: list = []

    for page in range(1, _MAX_REPO_PAGES + 1):
        params["page"] = page
        data = _get(session, url, params=params)

        if not isinstance(data, list):
            # Single-object response — nothing to paginate
            all_items.append(data)
            break

        if not data:
            break  # empty page → we've fetched everything

        all_items.extend(data)
        logger.debug("  page %d: +%d items (total so far: %d)", page, len(data), len(all_items))

        if len(data) < _PER_PAGE:
            break  # partial page → last page

    return all_items


# ---------------------------------------------------------------------------
# Individual data-fetching functions
# ---------------------------------------------------------------------------

def _fetch_user_meta(session: requests.Session, username: str) -> dict:
    """Fetch basic user metadata (avatar, bio, followers, etc.)."""
    logger.info("Fetching user metadata for: %s", username)
    url = f"{_BASE_URL}/users/{username}"
    data = _get(session, url)
    return {
        "name":        data.get("name"),
        "bio":         data.get("bio"),
        "location":    data.get("location"),
        "company":     data.get("company"),
        "blog":        data.get("blog"),
        "followers":   data.get("followers", 0),
        "following":   data.get("following", 0),
        "public_repos": data.get("public_repos", 0),
        "github_url":  data.get("html_url"),
        "avatar_url":  data.get("avatar_url"),
        "created_at":  data.get("created_at"),
    }


def _fetch_repos(session: requests.Session, username: str) -> list[dict]:
    """
    Fetch all public repositories for the user.

    Returns a list of dicts with the fields we care about.
    """
    logger.info("Fetching public repos for: %s", username)
    url = f"{_BASE_URL}/users/{username}/repos"
    raw_repos = _paginate(session, url, params={"type": "public", "sort": "updated"})

    repos: list[dict] = []
    for r in raw_repos:
        if r.get("fork"):
            # Skip forks — we want original work; set fork=False to include them
            continue
        repos.append({
            "name":         r.get("name", ""),
            "description":  r.get("description"),
            "language":     r.get("language"),       # primary language (GitHub's guess)
            "stars":        r.get("stargazers_count", 0),
            "forks":        r.get("forks_count", 0),
            "watchers":     r.get("watchers_count", 0),
            "topics":       r.get("topics", []),
            "url":          r.get("html_url", ""),
            "created_at":   r.get("created_at"),
            "updated_at":   r.get("pushed_at"),
            "is_fork":      r.get("fork", False),
            "size_kb":      r.get("size", 0),
        })

    logger.info("  → %d original public repos found", len(repos))
    return repos


def _aggregate_languages(repos: list[dict]) -> dict[str, int]:
    """
    Count how many repos use each language.

    Returns {language: repo_count} sorted descending by count.
    """
    lang_count: dict[str, int] = {}
    for repo in repos:
        lang = repo.get("language")
        if lang:
            lang_count[lang] = lang_count.get(lang, 0) + 1
    return dict(sorted(lang_count.items(), key=lambda x: x[1], reverse=True))


def _fetch_commits_last_year(session: requests.Session, username: str) -> int:
    """
    Return the total number of commits authored by `username` in the
    last 365 days using the GitHub Search Commits API.

    Note
    ----
    The Search API requires authentication to avoid strict rate limits
    (10 req/min unauthenticated, 30 req/min authenticated).
    We only need the `total_count` from the first page — no pagination needed.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
    url = f"{_BASE_URL}/search/commits"
    params = {
        "q": f"author:{username} committer-date:>{since}",
        "per_page": 1,   # we only need total_count, not the items
    }

    logger.info("Fetching commit count for '%s' since %s…", username, since)

    # The commits search endpoint needs a special Accept header
    original_accept = session.headers.get("Accept")
    session.headers["Accept"] = _COMMIT_SEARCH_ACCEPT

    try:
        data = _get(session, url, params=params)
        count = data.get("total_count", 0) if isinstance(data, dict) else 0
    except GitHubAPIError as exc:
        if exc.status_code in (403, 422):
            logger.warning(
                "Commit search unavailable (status %d). "
                "Ensure GITHUB_TOKEN is set and has sufficient scope. "
                "Returning 0.",
                exc.status_code,
            )
            count = 0
        else:
            raise
    finally:
        # Restore original Accept header
        session.headers["Accept"] = original_accept or "application/vnd.github+json"

    logger.info("  → %d commits in the last year", count)
    return count


def _fetch_events_commit_count(session: requests.Session, username: str) -> int:
    """
    Fallback: count PushEvents in GitHub's public events feed.

    Limitations
    -----------
    - GitHub only stores the last ~90 days and at most 300 events per user.
    - Each PushEvent can contain multiple commits (summed via `payload.size`).
    - Use only when the Search Commits API is unavailable.
    """
    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
    url = f"{_BASE_URL}/users/{username}/events/public"
    total_commits = 0
    page = 1

    logger.info("Fallback: counting commits via Events API for '%s'…", username)

    while page <= 10:  # GitHub caps at 10 pages (300 events)
        try:
            events = _get(session, url, params={"per_page": 30, "page": page})
        except GitHubAPIError:
            break

        if not isinstance(events, list) or not events:
            break

        for event in events:
            created_at_str = event.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            if created_at < one_year_ago:
                logger.debug("  Events older than 1 year reached — stopping.")
                return total_commits

            if event.get("type") == "PushEvent":
                payload = event.get("payload", {})
                total_commits += payload.get("size", 0)  # commits in this push

        page += 1

    logger.info("  → %d commits counted via Events API (≤90 days)", total_commits)
    return total_commits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_github_profile(
    username: str,
    token: Optional[str] = None,
    include_forks: bool = False,
) -> dict:
    """
    Fetch a GitHub user's public profile and return a structured dict.

    Parameters
    ----------
    username : str
        GitHub username (e.g. ``"torvalds"``).
    token : str, optional
        Personal Access Token. Falls back to the ``GITHUB_TOKEN``
        environment variable. Strongly recommended — unauthenticated
        requests are capped at 60/hr and the Search API won't work.
    include_forks : bool
        If ``True``, forked repos are included in the repo list and
        language/star aggregates. Default is ``False``.

    Returns
    -------
    dict
        A structured profile ready to be merged into a candidate dict::

            {
                "github": {
                    "username":         str,
                    "github_url":       str,
                    "name":             str | None,
                    "bio":              str | None,
                    "location":         str | None,
                    "company":          str | None,
                    "followers":        int,
                    "public_repos_count": int,
                    "total_stars":      int,
                    "commits_last_year": int,
                    "top_languages":    list[str],         # ordered by repo count
                    "language_breakdown": dict[str, int],  # {lang: repo_count}
                    "repos": [
                        {
                            "name":        str,
                            "description": str | None,
                            "language":    str | None,
                            "stars":       int,
                            "forks":       int,
                            "topics":      list[str],
                            "url":         str,
                            "updated_at":  str | None,
                        }
                    ]
                }
            }

    Raises
    ------
    GitHubAPIError
        On non-recoverable API errors (bad token, user not found, etc.).
    EnvironmentError
        If no token is available and the request fails due to rate-limiting.

    Example
    -------
    >>> profile = fetch_github_profile("octocat")
    >>> print(profile["github"]["commits_last_year"])
    42
    """
    resolved_token = token or os.environ.get("GITHUB_TOKEN")
    if not resolved_token:
        logger.warning(
            "No GITHUB_TOKEN found. Using unauthenticated requests "
            "(60 req/hr limit). Set GITHUB_TOKEN for reliable access."
        )

    session = _build_session(resolved_token)

    # 1. User metadata
    meta = _fetch_user_meta(session, username)

    # 2. Repos (original only, unless include_forks=True)
    raw_repos = _paginate(
        session,
        f"{_BASE_URL}/users/{username}/repos",
        params={"type": "public", "sort": "updated"},
    )

    repos: list[dict] = []
    for r in raw_repos:
        if r.get("fork") and not include_forks:
            continue
        repos.append({
            "name":        r.get("name", ""),
            "description": r.get("description"),
            "language":    r.get("language"),
            "stars":       r.get("stargazers_count", 0),
            "forks":       r.get("forks_count", 0),
            "watchers":    r.get("watchers_count", 0),
            "topics":      r.get("topics", []),
            "url":         r.get("html_url", ""),
            "created_at":  r.get("created_at"),
            "updated_at":  r.get("pushed_at"),
            "is_fork":     r.get("fork", False),
            "size_kb":     r.get("size", 0),
        })

    logger.info("Collected %d repos (%s forks).",
                len(repos), "including" if include_forks else "excluding")

    # 3. Language aggregation
    lang_breakdown = _aggregate_languages(repos)
    top_languages = list(lang_breakdown.keys())

    # 4. Total stars
    total_stars = sum(r["stars"] for r in repos)

    # 5. Commit count — try Search API first, fall back to Events
    commits_last_year: int
    if resolved_token:
        try:
            commits_last_year = _fetch_commits_last_year(session, username)
        except GitHubAPIError as exc:
            logger.warning("Search API failed (%s). Falling back to Events API.", exc)
            commits_last_year = _fetch_events_commit_count(session, username)
    else:
        logger.info("No token — using Events API for commit count (limited to ~90 days).")
        commits_last_year = _fetch_events_commit_count(session, username)

    # 6. Sort repos by stars descending for presentation
    repos_sorted = sorted(repos, key=lambda r: r["stars"], reverse=True)

    github_data = {
        "username":          username,
        "github_url":        meta.get("github_url", f"https://github.com/{username}"),
        "name":              meta.get("name"),
        "bio":               meta.get("bio"),
        "location":          meta.get("location"),
        "company":           meta.get("company"),
        "blog":              meta.get("blog"),
        "followers":         meta.get("followers", 0),
        "following":         meta.get("following", 0),
        "public_repos_count": meta.get("public_repos", len(repos)),
        "account_created_at": meta.get("created_at"),
        "total_stars":       total_stars,
        "commits_last_year": commits_last_year,
        "top_languages":     top_languages,
        "language_breakdown": lang_breakdown,
        "repos":             repos_sorted,
    }

    return {"github": github_data}


def merge_github_into_profile(candidate: dict, github_data: dict) -> dict:
    """
    Merge the dict returned by ``fetch_github_profile`` into an existing
    candidate profile dict (as produced by ``parse_resume``).

    - Appends GitHub skills/languages to ``candidate["skills"]`` (deduped).
    - Adds ``candidate["github"]`` with the full GitHub breakdown.
    - Adds ``candidate["github_projects"]`` listing repos as project-like entries.

    Parameters
    ----------
    candidate : dict
        Output of ``parse_resume.parse_resume()``.
    github_data : dict
        Output of ``fetch_github_profile()``.

    Returns
    -------
    dict
        A new merged dict (original dicts are not mutated).
    """
    import copy
    merged = copy.deepcopy(candidate)
    gh = github_data.get("github", {})

    # Attach raw GitHub data
    merged["github"] = gh

    # Enrich skills with GitHub languages (deduped, case-insensitive)
    existing_skills_lower = {s.lower() for s in merged.get("skills", [])}
    new_langs = [
        lang for lang in gh.get("top_languages", [])
        if lang.lower() not in existing_skills_lower
    ]
    merged["skills"] = merged.get("skills", []) + new_langs

    # Add GitHub repos as supplemental projects
    merged["github_projects"] = [
        {
            "title":       r["name"],
            "description": r["description"] or "",
            "language":    r["language"],
            "stars":       r["stars"],
            "url":         r["url"],
            "topics":      r["topics"],
        }
        for r in gh.get("repos", [])
    ]

    return merged
