# MovieLens Latest Small Dataset (`ml-latest-small`)

## Overview
The **MovieLens Latest Small** dataset is a benchmark dataset collected and maintained by **GroupLens Research** at the University of Minnesota. It describes user rating and free-text tagging activity from MovieLens, a non-commercial web-based movie recommendation service.

This dataset is ideal for prototyping **Content-Based**, **Collaborative Filtering**, and **Hybrid Recommendation Engines** due to its high interaction density and lack of structural noise.

---

## Dataset Quick Statistics
* **Total Ratings:** ~100,836 ratings
* **Total Tag Applications:** ~3,683 tags
* **Total Movies:** 9,742 movies
* **Total Users:** 610 users
* **Rating Scale:** 0.5 to 5.0 stars (in 0.5 increments)
* **Time Range:** March 29, 1996 – September 24, 2018

---

## File Structure & Feature Breakdown

The dataset comes as a `.zip` file containing **4 separate CSV files**. All files are UTF-8 encoded and formatted with standard comma delimiters.

### 1. `movies.csv` (Content Metadata)
Contains metadata for all movies available in the dataset. Used primarily for **Content-Based Filtering (TF-IDF)**.

| Column | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`movieId`** | Integer | Unique identifier for each movie. | `1` |
| **`title`** | String | Full movie title, including release year in parentheses. | `Toy Story (1995)` |
| **`genres`** | String | Pipe-separated list of genres (`\|`). | `Adventure\|Animation\|Children\|Comedy\|Fantasy` |

* **Special Value:** If a movie has no categorized genre, it is marked as `(no genres listed)`.
* **Available Genres:** Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western.

---

### 2. `ratings.csv` (User Interactions)
Contains explicit rating logs provided by users. Used primarily for **Collaborative Filtering (SVD / Matrix Factorization)**.

| Column | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`userId`** | Integer | Unique identifier for each user (anonymized). | `1` |
| **`movieId`** | Integer | Unique identifier matching `movies.csv`. | `1` |
| **`rating`** | Float | Star rating given by the user ($0.5$ to $5.0$). | `4.0` |
| **`timestamp`** | Integer | UTC timestamp (seconds since midnight Jan 1, 1970). | `964982703` |

* **User Privacy & Sparsity:** Every user in this dataset has rated **at least 20 movies**.

---

### 3. `tags.csv` (User-Generated Keywords)
Contains free-text metadata applied to movies by users. Excellent for enriching **TF-IDF features**.

| Column | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`userId`** | Integer | ID of the user who created the tag. | `2` |
| **`movieId`** | Integer | ID of the movie being tagged. | `60756` |
| **`tag`** | String | Short user-entered keyword or phrase. | `funny` |
| **`timestamp`** | Integer | UTC timestamp when the tag was submitted. | `1445712298` |

---

### 4. `links.csv` (External Identifiers)
Maps MovieLens IDs to external movie databases for fetching external posters or rich plot details.

| Column | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`movieId`** | Integer | MovieLens unique ID. | `1` |
| **`imdbId`** | String/Int | IMDb identifier (can construct `https://www.imdb.com/title/tt{imdbId}/`). | `0114709` |
| **`tmdbId`** | Integer | The Movie Database (TMDb) identifier. | `862` |

---

## How to Apply ML Techniques to MovieLens

```
                         ┌───► Content-Based (TF-IDF) ───► Feature: `genres` + `tag`
Input Query / User ID ───┤
                         └───► Collaborative (SVD)   ───► Pivot: `userId` x `movieId`
```

### 1. Content-Based Filtering (TF-IDF + Cosine Similarity)
* **Features:** Concatenate `genres` and user-generated `tag` words into a single text vector.
* **Method:** Compute Cosine Similarity between movie vectors.
* **Use Case:** Recommending items similar to a specific movie without requiring user history.

### 2. Collaborative Filtering (SVD / TruncatedSVD)
* **Features:** Pivot `ratings.csv` into a sparse interaction matrix $R$ of shape $(\text{Users} \times \text{Movies})$.
* **Method:** Apply Singular Value Decomposition ($R \approx U \cdot \Sigma \cdot V^T$) to predict unobserved ratings.
* **Use Case:** Personalizing recommendations for existing users based on past ratings.

---

## Python Loading Snippet

```python
import pandas as pd

# 1. Load files
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")
tags = pd.read_csv("ml-latest-small/tags.csv")

# 2. Extract Release Year from Title
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies['title_clean'] = movies['title