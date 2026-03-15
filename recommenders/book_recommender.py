"""
Book Recommender Module
=======================
Builds a TF-IDF model on book descriptions and uses cosine similarity
to recommend the top 5 similar books.
"""

import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ──────────────────────────────────────────────────────────────
# Data path
# ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
BOOKS_CSV = os.path.join(DATA_DIR, "data.csv")


@st.cache_data(show_spinner=False)
def _load_books():
    """Load books DataFrame and build TF-IDF + cosine sim matrix (cached)."""
    df = pd.read_csv(BOOKS_CSV)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["description"].fillna(""))

    # Precompute full cosine similarity matrix (6810 x 6810 — small enough)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return df, cosine_sim


def get_book_list() -> list[str]:
    """Return a sorted list of unique book titles for the dropdown."""
    df, _ = _load_books()
    return sorted(df["title"].dropna().unique().tolist())


def recommend(book_title: str) -> list[dict]:
    """
    Recommend the top 5 books most similar to *book_title*.

    Parameters
    ----------
    book_title : str
        Exact title string from the dropdown.

    Returns
    -------
    list[dict]
        Each dict has ``title``, ``authors``, ``average_rating``,
        and ``thumbnail``.
    """
    df, cosine_sim = _load_books()

    matches = df[df["title"] == book_title]
    if matches.empty:
        return []

    idx = matches.index[0]

    sim_scores = sorted(
        enumerate(cosine_sim[idx]),
        key=lambda x: x[1],
        reverse=True,
    )[1:6]  # Top 5, skip self

    results = []
    for i, _score in sim_scores:
        row = df.iloc[i]
        results.append({
            "title": row["title"],
            "authors": row.get("authors", "Unknown"),
            "average_rating": float(row.get("average_rating", 0)),
            "thumbnail": row.get("thumbnail", ""),
        })

    return results
