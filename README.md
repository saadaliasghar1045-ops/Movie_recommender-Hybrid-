# 🎬 CineMatix — Movie Recommender Backend

**FastAPI backend powering CineMatix, a content-based movie discovery and recommendation web application.**

This repository contains the backend services for CineMatix. It connects a machine learning recommendation pipeline with a frontend built using HTML, CSS, and vanilla JavaScript.

The backend processes movie search requests, generates content-based recommendations, matches movie titles, and retrieves additional movie information through The Movie Database (TMDB) API.

## ✨ Features

### 🧠 Content-Based Movie Recommendations
- Generate movie recommendations based on a user's text description.
- Recommend similar movies using a selected movie title.
- Represent movie content numerically using TF-IDF vectorization.
- Compare movie representations to identify relevant recommendations.
- Support configurable recommendation counts and optional release-year filtering.

### 🔎 Movie Title Matching
- Accept a movie title from the frontend.
- Return matching movie titles for the user to select.
- Use the selected title as input to the recommendation workflow.

### 🎞️ Movie Metadata Integration
Integrate recommendation results with TMDB to retrieve supplementary movie information, including available:

- Movie titles and overviews
- Posters and backdrop images
- Release dates and runtime
- Genres and ratings
- Cast and crew
- Production details
- Trailers and additional images

The availability of individual fields depends on the movie's TMDB record.

### ⚡ REST API
- Expose machine learning functionality through FastAPI endpoints.
- Accept JSON requests from the frontend.
- Return structured JSON responses.
- Support integration with a separately deployed static frontend.

### 🔐 Environment-Based Configuration
- Keep API credentials outside source code.
- Configure TMDB integration through environment variables.
- Separate frontend hosting from backend hosting.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend development and ML integration |
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Pandas | Movie dataset processing |
| Scikit-learn | Text vectorization and similarity processing |
| Joblib | Loading saved ML artifacts |
| HTTPX | HTTP requests to external APIs |
| python-dotenv | Loading environment variables from `.env` |
| TMDB API | Movie metadata and media |
| uv | Python dependency and project management |

## 🧠 How the Recommendation Pipeline Works

CineMatix uses content-based filtering to identify movies with content similar to a user's interests.

The simplified workflow is:

1. Movie information is prepared during preprocessing.
2. Movie text is transformed into numerical feature vectors using a fitted TF-IDF vectorizer.
3. A user submits a text description or selects a movie by title.
4. The recommendation pipeline processes the input and compares it with the available movie representations.
5. Relevant movies are selected and returned through the API.
6. The frontend displays the recommendation results.
7. TMDB is used to retrieve additional information and media when needed.

### Description-Based Recommendations

The user describes the type of movie they want to watch. The backend uses the description to identify relevant movie content and returns recommendations.

### Title-Based Recommendations

The user searches for a movie title. After the backend returns matching titles, the selected movie can be used to find similar films.

The recommendation system is content-based: it focuses on movie content rather than requiring a user's rating history.

## 🏗️ Architecture

```text
CineMatix Frontend
HTML + CSS + JavaScript
          |
          | HTTP / JSON
          v
FastAPI Backend
          |
          ├── Title Matching
          |
          ├── Recommendation API
          |        |
          |        v
          |   Content Pipeline
          |        |
          |        ├── TF-IDF Vectorizer
          |        ├── Saved Movie Vectors
          |        └── Movie Dataset
          |
          └── TMDB Integration
                   |
                   v
             Movie Metadata
             Posters and Media
```

The frontend and backend are separate applications. The frontend sends API requests, while the backend performs recommendation processing and retrieves supplementary movie information.

## 📂 Project Structure

The following illustrates the main backend components; filenames and artifact names should match the actual repository.

```text
movie-recommender-backend/
│
├── app/
│   ├── app.py
│   ├── main.py
│   └── schemas/
│       ├── recommendation_input.py
│       └── title_match_inp.py
│
├── src/
│   ├── pipelines/
│   │   └── content_pipeline.py
│   │
│   ├── vectorizers_and_matrices/
│   │
│   └── datasets_for_app/
│
├── .env
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

### Main Components

- `app/app.py` — Defines the FastAPI application and API endpoints.
- `app/main.py` — Application entry point for running the API server, if included in the repository.
- `app/schemas/` — Defines request schemas used for API validation.
- `src/pipelines/content_pipeline.py` — Contains the content-based recommendation pipeline.
- `src/vectorizers_and_matrices/` — Stores the fitted vectorizer and precomputed movie representations.
- `src/datasets_for_app/` — Contains movie datasets used by the application.

## 🔌 API Endpoints

The following endpoints represent the current CineMatix API design.

| Endpoint | Method | Purpose |
|---|---|---|
| `/home` | GET | Retrieve homepage-related data, if enabled |
| `/trending` | Check implementation | Retrieve trending movie data |
| `/match_title` | POST | Find matching movie titles |
| `/recommend` | POST | Generate content-based recommendations |
| `/movie_info/{tmdb_id}` | GET | Retrieve movie information using a TMDB ID |

Check the route decorators in `app/app.py` for the exact HTTP method and response structure of each endpoint.

### 1. Match Movie Titles

**Endpoint:** `POST /match_title`

Example request:

```json
{
  "title": "Inception"
}
```

The backend processes the submitted title and returns matching movie titles for selection.

### 2. Generate Recommendations

**Endpoint:** `POST /recommend`

Example description-based request:

```json
{
  "inp_text": "A science-fiction movie about dreams and mystery",
  "n_recoms": 20,
  "limit_year": null,
  "by_description": true
}
```

Parameter descriptions:

| Parameter | Description |
|---|---|
| `inp_text` | Description or selected movie title, depending on the mode |
| `n_recoms` | Number of recommendations requested |
| `limit_year` | Optional release-year limit |
| `by_description` | `true` for description-based