"""
Album Art Fetcher Utility
=========================
Fetches song/album artwork from the iTunes Search API.

- Completely free — no API key required
- Results are cached to avoid redundant API calls
- Request timeout set to 5 seconds for safety
"""

import streamlit as st
import requests
from urllib.parse import quote


ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_album_art(track_name: str, artist: str = "") -> str | None:
    """
    Fetch album artwork URL for a given track name and optional artist.

    Parameters
    ----------
    track_name : str
        The song title to search for.
    artist : str, optional
        The artist name (improves accuracy).

    Returns
    -------
    str or None
        URL to the album artwork (600x600), or None if not found.
    """
    query = f"{track_name} {artist}".strip()
    if not query:
        return None

    params = {
        "term": query,
        "media": "music",
        "entity": "song",
        "limit": 1,
    }

    try:
        response = requests.get(ITUNES_SEARCH_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if results:
            # Get the artwork URL and bump resolution to 600x600
            art_url = results[0].get("artworkUrl100", "")
            if art_url:
                return art_url.replace("100x100", "600x600")
        return None

    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException:
        return None
