"""
Sentiment Analysis Module
=========================
Two capabilities:
  1. analyze(text)       — TextBlob on any user-typed review (Positive/Neutral/Negative)
  2. analyze_imdb(title) — Aggregated sentiment from the IMDB reviews dataset
"""

import os
import pandas as pd
import streamlit as st
from textblob import TextBlob

# ──────────────────────────────────────────────────────────────
# IMDB Reviews dataset
# ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
IMDB_CSV = os.path.join(DATA_DIR, "imdb_reviews.csv")


@st.cache_data(show_spinner=False)
def _load_imdb_reviews():
    """Load the IMDB reviews dataset (cached)."""
    try:
        return pd.read_csv(IMDB_CSV)
    except Exception as e:
        st.error(f"Error loading IMDB dataset: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# 1. Free-text sentiment (TextBlob)
# ──────────────────────────────────────────────────────────────
def analyze(text: str) -> dict:
    """
    Analyze the sentiment of any user-provided text.

    Returns
    -------
    dict with ``sentiment``, ``polarity``, ``subjectivity``.
    """
    if not text or not text.strip():
        return {"sentiment": "Neutral", "polarity": 0.0, "subjectivity": 0.0}

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "sentiment": sentiment,
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
    }


# ──────────────────────────────────────────────────────────────
# 2. IMDB-based sentiment for a movie title
# ──────────────────────────────────────────────────────────────
def analyze_imdb(movie_title: str) -> dict | None:
    """
    Look up the IMDB reviews dataset for *movie_title* and return
    aggregated sentiment counts.

    Parameters
    ----------
    movie_title : str
        Movie title to search for in reviews.

    Returns
    -------
    dict or None
        ``positive``, ``negative``, ``neutral``, ``total`` counts.
        Returns None if the dataset is unavailable.
    """
    df = _load_imdb_reviews()
    if df is None:
        return None

    # Search for the movie title in review text (case-insensitive)
    mask = df["review"].str.contains(movie_title, case=False, na=False)
    filtered = df[mask]

    if filtered.empty:
        return {"positive": 0, "negative": 0, "neutral": 0, "total": 0}

    positive = int((filtered["sentiment"] == "positive").sum())
    negative = int((filtered["sentiment"] == "negative").sum())
    neutral = int((filtered["sentiment"] == "neutral").sum())

    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "total": len(filtered),
    }
