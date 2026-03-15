<div align="center">
  
# 🤖 RecommendAI: Multi-Recommender System

A premium, production-ready AI recommender system featuring **Movie, Song, Book, and Sentiment Analysis** modules built with Streamlit, Scikit-Learn, and NLP techniques.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-link-here.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🌟 Overview

RecommendAI is a centralized hub for content discovery. Originally an older university capstone project, it has been completely architected and rebuilt into a modern, modular, and performant web application. It uses **Content-Based Filtering (Cosine Similarity)** and **Natural Language Processing (TextBlob)** to provide highly accurate recommendations and emotional tone analysis.

### ✨ Core Modules

1. **🎬 Movie Recommender**
   - Suggests the top 5 similar movies from a dataset of 10,000+ titles.
   - Integrates live with the **TMDB API** to fetch high-res posters, ratings, genres, and overviews.
   - Cross-references titles with an IMDB review dataset to display real-time audience sentiment.
2. **🎵 Song Recommender**
   - Recommends tracks from a 30,000+ Spotify dataset based on audio features and metadata using TF-IDF.
   - Automatically pulls real album artwork via the **iTunes Search API** (zero API key required).
3. **📚 Book Recommender**
   - Content-based filtering across 6,800+ books.
   - Displays cover art and author ratings perfectly aligned with the cinematic UI.
4. **💬 NLP Sentiment Analysis**
   - **Background Model:** The core sentiment engine was originally modeled using a **TF-IDF Vectorizer** trained on IMDB movie reviews, achieving a remarkable **98% accuracy** in classification.
   - **Movie Integration:** When a movie is recommended, the system cross-references the title against the IMDB reviews dataset. It aggregates historical reviews for that specific movie and classifies them into **Positive**, **Negative**, or **Neutral**—displaying the exact sentiment counts directly on the movie card to help users gauge audience reaction.
   - **Live Analysis Studio:** Features a dedicated free-text analysis tool powered by `TextBlob`. It instantly calculates **Polarity** (−1.0 to 1.0) and **Subjectivity**, displaying dynamic emojis and UI color shifts based on the emotional tone of any pasted text.

---

## 🎨 Premium UI/UX Design

The application distances itself from standard data-science wrappers by implementing a **Warm Cinematic Theme** inspired by Netflix and IMDB:

- **Pure Charcoal Backgrounds** (`#0d0d0d` to `#1a1a1a`) with no cold blue tints.
- **Glassmorphism Elements** on result cards.
- **Fluid Micro-animations** (floating hero icons, pulsing emojis, smooth hover states).
- Custom typography using the **Outfit** Google Font.

---

## 🛠️ Architecture & Tech Stack

| Category             | Technology Used                                       |
| :------------------- | :---------------------------------------------------- |
| **Frontend Options** | Streamlit, HTML/CSS inside Markdown                   |
| **Data Processing**  | Pandas, NumPy                                         |
| **Machine Learning** | Scikit-Learn (`CountVectorizer`, `cosine_similarity`) |
| **NLP**              | TextBlob                                              |
| **External APIs**    | TMDB API (Movies), iTunes API (Music)                 |

---

## 🚧 Challenges Faced & Overcome

Building a robust, deployable recommender system came with several structural challenges:

### 1. The 763 MB GitHub Constraint

**Challenge:** The original capstone project relied on a precomputed `similarity.pkl` matrix that weighed **763 MB**, which is strictly blocked by GitHub's 100 MB hard limit.
**Solution:** The architecture was rewritten to compute the cosine similarity matrix **on-the-fly**. Using `CountVectorizer` and `@st.cache_data`, the matrix is generated directly from the movie tags in just ~1.5 seconds on the very first app load, entirely bypassing the need to store or push massive pickle files.

### 2. Streamlit Security Policies (CSP)

**Challenge:** Initially, injected HTML `<img>` tags were used for movie posters, but Streamlit's strict Content Security Policy blocked external image rendering.
**Solution:** The UI was refactored to use native `st.image()` components styled via CSS targeting the `div[data-testid="stImage"]` classes, ensuring perfect rendering without security violations.

### 3. Missing Song Artwork

**Challenge:** The Spotify dataset lacked album artwork links, and the Spotify API requires complex OAuth tokens.
**Solution:** Implemented a lightweight, robust utility (`album_art_fetcher.py`) that queries the public **iTunes Search API**, caching the 600x600 px album covers instantly without requiring an API key.

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Set up the environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Add your TMDB API Key

Create a `.streamlit/secrets.toml` file in the root directory:

```toml
TMDB_API_KEY = "your_v3_api_key_here"
```

### 4. Download Datasets

_(Since GitHub limits file sizes, the `.csv` and `.pkl` datasets are not tracked via Git)._
Place the following files inside the `data/` folder:

- `movies_list.pkl`
- `spotify_songs.csv`
- `data.csv`
- `imdb_reviews.csv`
- `booksummaries.txt`

### 5. Launch the app

```bash
streamlit run app.py
```

---

## ☁️ Deployment

This project is optimized for deployment on **Streamlit Community Cloud**:

1. Push your code to GitHub (datasets under 100MB will push successfully).
2. Connect your repository to Streamlit Cloud.
3. Add your `TMDB_API_KEY` in the Streamlit Cloud Secrets interface.
4. Deploy! The similarity matrix will be computed dynamically on the cloud server.

---

_Built with ❤️ utilizing Python and Machine Learning._
