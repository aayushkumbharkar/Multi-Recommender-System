"""
Movie Recommender Module
========================
Uses precomputed cosine similarity matrix (similarity.pkl) and movie metadata
(movies_list.pkl) to recommend the top 5 similar movies for any given title.
"""

import os
import pickle
import streamlit as st

# ──────────────────────────────────────────────────────────────
# Data paths (relative to project root)
# ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MOVIES_PKL = os.path.join(DATA_DIR, "movies_list.pkl")
SIMILARITY_PKL = os.path.join(DATA_DIR, "similarity.pkl")


@st.cache_data(show_spinner=False)
def _load_movies():
    """Load the movies DataFrame from pickle."""
    with open(MOVIES_PKL, "rb") as f:
        movies_df = pickle.load(f)
    return movies_df


@st.cache_data(show_spinner=False)
def _load_similarity():
    """Load the precomputed cosine-similarity matrix."""
    with open(SIMILARITY_PKL, "rb") as f:
        similarity_matrix = pickle.load(f)
    return similarity_matrix


def get_movie_list() -> list[str]:
    """
    Return a sorted list of all available movie titles.
    Used to populate the dropdown selector in the UI.
    """
    movies_df = _load_movies()
    return sorted(movies_df["title"].dropna().unique().tolist())


def recommend(movie_title: str) -> list[dict]:
    """
    Recommend the top 5 movies most similar to *movie_title*.

    Parameters
    ----------
    movie_title : str
        Exact title string (as provided by the dropdown).

    Returns
    -------
    list[dict]
        Each dict contains ``id`` (TMDB movie ID) and ``title``.
        Returns an empty list if the title is not found.
    """
    movies_df = _load_movies()
    similarity = _load_similarity()

    # Locate the movie index
    matches = movies_df[movies_df["title"] == movie_title]
    if matches.empty:
        return []

    idx = matches.index[0]

    # Get pairwise similarity scores, sort descending, skip self (index 0)
    sim_scores = sorted(
        enumerate(similarity[idx]),
        key=lambda x: x[1],
        reverse=True,
    )[1:6]  # Top 5

    # Build result list
    results = []
    for i, _score in sim_scores:
        results.append({
            "id": int(movies_df.iloc[i]["id"]),
            "title": movies_df.iloc[i]["title"],
        })

    return results
