"""
Poster Fetcher Utility
======================
Fetches movie poster URLs and details from the TMDB API.

Strategy:
  1. Try fetch by TMDB movie ID (fast, direct)
  2. If that fails, fall back to search by movie title (finds most movies)

- API key is read from Streamlit secrets (st.secrets["TMDB_API_KEY"])
- Results are cached to avoid redundant API calls
- Request timeout is set to 5 seconds for safety
"""

import streamlit as st
import requests

TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"


def _get_api_key() -> str | None:
    """Retrieve TMDB API key from Streamlit secrets."""
    try:
        return st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster(movie_id: int) -> str | None:
    """
    Fetch just the poster URL for a given TMDB movie ID.
    """
    details = fetch_movie_details(movie_id)
    if details:
        return details.get("poster_url")
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_movie_details(movie_id: int, title: str = "") -> dict | None:
    """
    Fetch full movie details from TMDB.

    Uses movie ID first, falls back to title search if ID fails.

    Returns
    -------
    dict or None
        Keys: poster_url, rating, genres, year, overview, runtime.
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    # --- Strategy 1: Direct ID lookup ---
    result = _fetch_by_id(api_key, movie_id)
    if result:
        return result

    # --- Strategy 2: Search by title (fallback) ---
    if title:
        result = _fetch_by_search(api_key, title)
        if result:
            return result

    return None


def _fetch_by_id(api_key: str, movie_id: int) -> dict | None:
    """Fetch movie details by TMDB movie ID."""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {"api_key": api_key, "language": "en-US"}

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return _parse_movie_data(data)
    except requests.exceptions.RequestException:
        return None


def _fetch_by_search(api_key: str, title: str) -> dict | None:
    """Search TMDB by movie title and return the first match."""
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {"api_key": api_key, "language": "en-US", "query": title}

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return None

        # Pick the best match (first result)
        movie = results[0]

        # Search results don't include full genres, so use genre_ids
        poster_path = movie.get("poster_path")
        release_date = movie.get("release_date", "")

        return {
            "poster_url": f"{POSTER_BASE_URL}{poster_path}" if poster_path else None,
            "rating": round(movie.get("vote_average", 0), 1),
            "genres": "—",  # search endpoint doesn't return genre names
            "year": release_date[:4] if release_date else "—",
            "overview": _truncate(movie.get("overview", "")),
            "runtime": 0,
        }

    except requests.exceptions.RequestException:
        return None


def _parse_movie_data(data: dict) -> dict | None:
    """Parse a TMDB movie detail response into a clean dict."""
    poster_path = data.get("poster_path")
    if not poster_path:
        return None  # No poster means this entry isn't useful

    genres = ", ".join([g["name"] for g in data.get("genres", [])][:3])
    release_date = data.get("release_date", "")

    return {
        "poster_url": f"{POSTER_BASE_URL}{poster_path}",
        "rating": round(data.get("vote_average", 0), 1),
        "genres": genres or "—",
        "year": release_date[:4] if release_date else "—",
        "overview": _truncate(data.get("overview", "")),
        "runtime": data.get("runtime", 0),
    }


def _truncate(text: str, max_len: int = 120) -> str:
    """Truncate text with ellipsis."""
    if len(text) > max_len:
        return text[:max_len - 3] + "…"
    return text
