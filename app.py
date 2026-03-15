"""
Multi-Recommender AI System
============================
A premium Streamlit web application featuring:
    🎬 Movie Recommender   – cosine similarity with TMDB posters
    🎵 Song Recommender    – TF-IDF on Spotify data
    📚 Book Recommender    – TF-IDF on book descriptions
    💬 Sentiment Analysis   – TextBlob-powered review classifier

Run with:  streamlit run app.py
"""

import streamlit as st

# ──────────────────────────────────────────────────────────────
# Page configuration (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Recommender AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Premium Design System — Warm Cinematic Theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* =============== IMPORTS =============== */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* =============== GLOBAL =============== */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Warm cinematic background — pure charcoal, NO blue */
    .stApp {
        background: linear-gradient(160deg, #0d0d0d 0%, #141414 40%, #1a1a1a 100%);
    }

    /* Hide Streamlit chrome */
    header[data-testid="stHeader"] { background: transparent !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* =============== CONSISTENT IMAGE HEIGHTS =============== */
    /* Force all images in columns to same height so titles align */
    div[data-testid="stImage"] img {
        height: 280px;
        object-fit: cover;
        border-radius: 12px;
    }

    /* =============== SIDEBAR — Charcoal with warm accent =============== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111111 0%, #1a1a1a 50%, #141414 100%) !important;
        border-right: 1px solid rgba(229, 62, 62, 0.12);
    }
    section[data-testid="stSidebar"] * {
        color: #d4d4d4 !important;
    }
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 4px;
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 14px 18px !important;
        border-radius: 12px;
        transition: all 0.3s ease;
        font-weight: 500;
        font-size: 0.95rem;
        border: 1px solid transparent;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(229, 62, 62, 0.08);
        border-color: rgba(229, 62, 62, 0.2);
    }

    /* Sidebar brand */
    .sidebar-brand {
        text-align: center;
        padding: 28px 16px 24px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 20px;
    }
    .sidebar-brand h2 {
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e53e3e, #ed8936);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 4px 0;
    }
    .sidebar-brand p {
        font-size: 0.72rem;
        color: #666 !important;
        margin: 0;
        letter-spacing: 2.5px;
        text-transform: uppercase;
    }

    /* =============== HERO SECTION =============== */
    .hero-section {
        text-align: center;
        padding: 48px 20px 32px;
        position: relative;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0; left: 50%;
        transform: translateX(-50%);
        width: 500px; height: 400px;
        background: radial-gradient(ellipse, rgba(229,62,62,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-icon {
        font-size: 3.2rem;
        display: inline-block;
        margin-bottom: 12px;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #888;
        font-size: 1rem;
        font-weight: 300;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* =============== IMDB REVIEW STATS =============== */
    .imdb-stats {
        margin-top: 10px;
        padding: 10px 0 0 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        font-size: 0.78rem;
    }
    .imdb-stats .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 3px 0;
    }
    .imdb-stats .stat-label { color: #888; }
    .imdb-stats .stat-pos { color: #68d391; font-weight: 600; }
    .imdb-stats .stat-neg { color: #fc8181; font-weight: 600; }
    .imdb-stats .stat-neu { color: #f6e05e; font-weight: 600; }
    .imdb-total {
        text-align: center;
        font-size: 0.7rem;
        color: #555;
        margin-top: 6px;
    }

    /* =============== SONG ARTWORK =============== */
    .song-art {
        height: 150px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #1db954, #0d7a37, #064a20);
        border-radius: 14px;
        margin-bottom: 12px;
        box-shadow: 0 8px 24px rgba(29,185,84,0.15);
        position: relative;
        overflow: hidden;
    }
    .song-art::after {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.04) 0%, transparent 50%);
    }

    /* =============== POPULARITY BAR =============== */
    .pop-bar {
        height: 4px;
        background: rgba(255,255,255,0.08);
        border-radius: 4px;
        margin-top: 8px;
        overflow: hidden;
    }
    .pop-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #e53e3e, #ed8936);
        transition: width 1s ease;
    }

    /* =============== SENTIMENT RESULTS =============== */
    .sentiment-result {
        border-radius: 20px;
        padding: 36px;
        margin-top: 24px;
        border: 1px solid rgba(255,255,255,0.06);
        position: relative;
        overflow: hidden;
    }
    .sentiment-result::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        right: 0; height: 2px;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, rgba(16,64,36,0.6), rgba(10,40,24,0.7));
    }
    .sentiment-positive::before {
        background: linear-gradient(90deg, transparent, #22c55e, transparent);
    }
    .sentiment-negative {
        background: linear-gradient(135deg, rgba(64,16,16,0.6), rgba(40,10,10,0.7));
    }
    .sentiment-negative::before {
        background: linear-gradient(90deg, transparent, #ef4444, transparent);
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, rgba(64,56,16,0.6), rgba(40,35,10,0.7));
    }
    .sentiment-neutral::before {
        background: linear-gradient(90deg, transparent, #eab308, transparent);
    }
    .sentiment-emoji {
        font-size: 3rem;
        margin-bottom: 8px;
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    .sentiment-label {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f0f0;
        margin: 4px 0;
    }
    .sentiment-metrics {
        display: flex;
        gap: 40px;
        margin-top: 16px;
        justify-content: center;
    }
    .sentiment-metric {
        text-align: center;
    }
    .sentiment-metric .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0f0f0;
        display: block;
    }
    .sentiment-metric .label {
        font-size: 0.78rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* =============== BUTTONS — Warm gradient =============== */
    .stButton > button {
        background: linear-gradient(135deg, #e53e3e 0%, #dd6b20 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 32px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(229, 62, 62, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(229, 62, 62, 0.4) !important;
    }

    /* =============== SELECTBOX =============== */
    div[data-testid="stSelectbox"] label {
        color: #999 !important;
        font-weight: 500;
        font-size: 0.9rem;
    }

    /* =============== TEXT AREA =============== */
    .stTextArea textarea {
        background: rgba(20, 20, 20, 0.8) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        color: #e0e0e0 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 16px !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(229, 62, 62, 0.3) !important;
        box-shadow: 0 0 0 3px rgba(229, 62, 62, 0.08) !important;
    }

    /* =============== DIVIDER =============== */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(229,62,62,0.15), transparent);
        margin: 32px 0;
    }

    /* =============== PROGRESS BAR =============== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #e53e3e, #ed8936) !important;
        border-radius: 8px;
    }

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Module imports (after page config)
# ──────────────────────────────────────────────────────────────
from recommenders import movie_recommender, song_recommender, book_recommender
from sentiment import sentiment_analysis
from utils import poster_fetcher
from utils import album_art_fetcher

# ──────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <h2>🤖 RecommendAI</h2>
            <p>Multi-Recommender System</p>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🎬 Movie Recommender",
            "🎵 Song Recommender",
            "📚 Book Recommender",
            "💬 Sentiment Analysis",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption("v1.0 • Powered by scikit-learn & TextBlob")


# ══════════════════════════════════════════════════════════════
#  🎬  MOVIE RECOMMENDER
# ══════════════════════════════════════════════════════════════
if page == "🎬 Movie Recommender":
    st.markdown("""
        <div class="hero-section">
            <div class="hero-icon">🎬</div>
            <h1 class="hero-title">Movie Recommender</h1>
            <p class="hero-subtitle">
                Discover your next favorite film — powered by cosine similarity
                across 10,000+ titles with TMDB posters & IMDB reviews
            </p>
        </div>
    """, unsafe_allow_html=True)

    movies = movie_recommender.get_movie_list()
    selected = st.selectbox(
        "Pick a movie",
        movies,
        index=None,
        placeholder="🔍  Start typing to search…",
    )

    if st.button("✨ Get Recommendations", width="stretch") and selected:
        with st.spinner("🎬 Finding similar movies…"):
            results = movie_recommender.recommend(selected)

        if not results:
            st.error("No recommendations found. Please try another title.")
        else:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            cols = st.columns(5, gap="medium")
            for col, movie in zip(cols, results):
                with col:
                    # Fetch full TMDB details (poster + metadata)
                    details = poster_fetcher.fetch_movie_details(movie["id"], movie["title"])

                    if details and details.get("poster_url"):
                        st.image(details["poster_url"], width="stretch")
                    else:
                        st.markdown(
                            '<div style="height:240px;display:flex;align-items:center;'
                            'justify-content:center;background:linear-gradient(135deg,'
                            '#1a1a1a,#222);border-radius:14px;font-size:3rem;'
                            'box-shadow:0 8px 20px rgba(0,0,0,0.5);">🎬</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"**{movie['title']}**")

                    # Show TMDB metadata (rating, genres, year)
                    if details:
                        rating = details["rating"]
                        rating_color = "#68d391" if rating >= 7 else "#f6e05e" if rating >= 5 else "#fc8181"
                        st.markdown(
                            f"""<p style="margin:4px 0 2px 0;">
                                <span style="background:{rating_color};color:#000;
                                    font-weight:700;padding:2px 8px;border-radius:6px;
                                    font-size:0.78rem;">⭐ {rating}</span>
                                <span style="color:#666;font-size:0.72rem;margin-left:6px;">
                                    {details['year']}</span>
                            </p>
                            <p style="color:#888;font-size:0.75rem;margin:2px 0;">
                                🎭 {details['genres']}</p>
                            """,
                            unsafe_allow_html=True,
                        )
                        if details.get("overview"):
                            st.caption(details["overview"])

                    # IMDB sentiment for this recommended movie
                    imdb = sentiment_analysis.analyze_imdb(movie["title"])
                    if imdb and imdb["total"] > 0:
                        st.markdown(f"""
                            <div class="imdb-stats">
                                <div class="stat-row">
                                    <span class="stat-label">Positive</span>
                                    <span class="stat-pos">{imdb['positive']}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Negative</span>
                                    <span class="stat-neg">{imdb['negative']}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Neutral</span>
                                    <span class="stat-neu">{imdb['neutral']}</span>
                                </div>
                                <div class="imdb-total">📊 {imdb['total']} reviews</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.caption("No IMDB reviews found")


# ══════════════════════════════════════════════════════════════
#  🎵  SONG RECOMMENDER
# ══════════════════════════════════════════════════════════════
elif page == "🎵 Song Recommender":
    st.markdown("""
        <div class="hero-section">
            <div class="hero-icon">🎵</div>
            <h1 class="hero-title">Song Recommender</h1>
            <p class="hero-subtitle">
                Find your next banger — TF-IDF similarity across 23,000+
                Spotify tracks with artist & album info
            </p>
        </div>
    """, unsafe_allow_html=True)

    songs = song_recommender.get_song_list()
    selected = st.selectbox(
        "Pick a song",
        songs,
        index=None,
        placeholder="🔍  Start typing to search…",
    )

    if st.button("✨ Get Recommendations", width="stretch") and selected:
        with st.spinner("🎵 Finding similar tracks…"):
            results = song_recommender.recommend(selected)

        if not results:
            st.error("No recommendations found. Please try another song.")
        else:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            cols = st.columns(5, gap="medium")
            for col, song in zip(cols, results):
                with col:
                    pop = min(song["track_popularity"], 100)

                    # Fetch real album artwork from iTunes API
                    art_url = album_art_fetcher.fetch_album_art(
                        song["track_name"], song["track_artist"]
                    )
                    if art_url:
                        st.image(art_url, width="stretch")
                    else:
                        st.markdown(
                            '<div class="song-art">🎵</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"**{song['track_name']}**")
                    st.markdown(
                        f"""<p style="color:#aaa;font-size:0.82rem;margin:2px 0;">
                            🎤 {song['track_artist']}</p>
                        <p style="color:#777;font-size:0.78rem;margin:2px 0;">
                            💿 {song['track_album_name']}</p>
                        <div class="pop-bar">
                            <div class="pop-bar-fill" style="width:{pop}%;"></div>
                        </div>
                        <p style="color:#666;font-size:0.7rem;margin:4px 0 0 0;
                                  text-align:right;">🔥 {pop}/100</p>
                        """,
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════
#  📚  BOOK RECOMMENDER
# ══════════════════════════════════════════════════════════════
elif page == "📚 Book Recommender":
    st.markdown("""
        <div class="hero-section">
            <div class="hero-icon">📚</div>
            <h1 class="hero-title">Book Recommender</h1>
            <p class="hero-subtitle">
                Discover your next great read — content-based filtering
                across 6,800+ books with cover art & ratings
            </p>
        </div>
    """, unsafe_allow_html=True)

    books = book_recommender.get_book_list()
    selected = st.selectbox(
        "Pick a book",
        books,
        index=None,
        placeholder="🔍  Start typing to search…",
    )

    if st.button("✨ Get Recommendations", width="stretch") and selected:
        with st.spinner("📚 Finding similar books…"):
            results = book_recommender.recommend(selected)

        if not results:
            st.error("No recommendations found. Please try another title.")
        else:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            cols = st.columns(5, gap="medium")
            for col, book in zip(cols, results):
                with col:
                    thumb = book.get("thumbnail", "")
                    if thumb:
                        st.image(thumb, width="stretch")
                    else:
                        st.markdown(
                            '<div style="height:220px;display:flex;align-items:center;'
                            'justify-content:center;background:linear-gradient(135deg,'
                            '#1a1a1a,#222);border-radius:14px;font-size:3rem;'
                            'box-shadow:0 8px 20px rgba(0,0,0,0.5);">📖</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(f"**{book['title']}**")
                    rating = book["average_rating"]
                    stars = "⭐" * int(rating) + ("✨" if rating % 1 >= 0.5 else "")
                    st.markdown(
                        f"""<p style="color:#aaa;font-size:0.82rem;margin:2px 0;">
                            ✍️ {book['authors']}</p>
                        <p style="color:#f6e05e;font-size:0.82rem;margin:2px 0;">
                            {stars} {rating}</p>
                        """,
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════
#  💬  SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "💬 Sentiment Analysis":
    st.markdown("""
        <div class="hero-section">
            <div class="hero-icon">💬</div>
            <h1 class="hero-title">Sentiment Analysis</h1>
            <p class="hero-subtitle">
                Paste any review or text and instantly detect its emotional tone
                using NLP-powered analysis
            </p>
        </div>
    """, unsafe_allow_html=True)

    review_text = st.text_area(
        "Enter your review text",
        height=180,
        placeholder="e.g. This movie was absolutely fantastic! The acting was superb…",
    )

    if st.button("✨ Analyze Sentiment", width="stretch") and review_text.strip():
        with st.spinner("🔍 Analyzing sentiment…"):
            result = sentiment_analysis.analyze(review_text)

        sentiment = result["sentiment"]
        css_class = f"sentiment-{sentiment.lower()}"
        emoji_map = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
        color_map = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#eab308"}

        st.markdown(f"""
            <div class="sentiment-result {css_class}">
                <div style="text-align:center;">
                    <div class="sentiment-emoji">{emoji_map[sentiment]}</div>
                    <div class="sentiment-label">{sentiment}</div>
                    <div class="sentiment-metrics">
                        <div class="sentiment-metric">
                            <span class="value" style="color:{color_map[sentiment]}">
                                {result['polarity']}</span>
                            <span class="label">Polarity</span>
                        </div>
                        <div class="sentiment-metric">
                            <span class="value">{result['subjectivity']}</span>
                            <span class="label">Subjectivity</span>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        polarity_pct = (result["polarity"] + 1) / 2
        st.progress(polarity_pct, text="Polarity Scale  (−1.0 ← Negative | Positive → +1.0)")
