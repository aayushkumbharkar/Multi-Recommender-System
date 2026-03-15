"""
Song Recommender Module
=======================
Builds a TF-IDF model on Spotify track names + artist names and uses
cosine similarity to recommend the top 5 similar songs.
"""

import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────
# Data path
# ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SONGS_CSV = os.path.join(DATA_DIR, "spotify_songs.csv")


def _preprocess(text: str) -> str:
    """Lowercase and collapse whitespace."""
    if isinstance(text, str):
        return " ".join(text.lower().split())
    return ""


@st.cache_data(show_spinner=False)
def _load_songs():
    """Load songs DataFrame and build TF-IDF matrix (cached)."""
    df = pd.read_csv(SONGS_CSV)

    # Combine track name + artist for richer similarity
    df["_combined"] = (
        df["track_name"].fillna("") + " " + df["track_artist"].fillna("")
    ).apply(_preprocess)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["_combined"])

    return df, vectorizer, tfidf_matrix


def get_song_list() -> list[str]:
    """Return a sorted list of unique song titles for the dropdown."""
    df, _, _ = _load_songs()
    return sorted(df["track_name"].dropna().unique().tolist())


def recommend(song_title: str) -> list[dict]:
    """
    Recommend the top 5 songs most similar to *song_title*.

    Parameters
    ----------
    song_title : str
        The song title to query.

    Returns
    -------
    list[dict]
        Each dict has ``track_name``, ``track_artist``,
        ``track_album_name``, and ``track_popularity``.
    """
    df, vectorizer, tfidf_matrix = _load_songs()

    # Vectorize the input query
    query_vec = vectorizer.transform([_preprocess(song_title)])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Top 6 then drop self-match (index 0 is highest = self if exact match)
    top_indices = scores.argsort()[::-1]

    results = []
    for idx in top_indices:
        # Skip exact self-match by title
        if df.iloc[idx]["track_name"] == song_title and len(results) == 0:
            continue
        results.append({
            "track_name": df.iloc[idx]["track_name"],
            "track_artist": df.iloc[idx]["track_artist"],
            "track_album_name": df.iloc[idx]["track_album_name"],
            "track_popularity": int(df.iloc[idx]["track_popularity"]),
        })
        if len(results) == 5:
            break

    return results
