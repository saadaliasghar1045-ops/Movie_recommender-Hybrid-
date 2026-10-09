import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_movie_metadata(
    links_path="../datasets/links.csv",
    output_path="../datasets/tmdb_movies.csv",
):
    if not TMDB_API_KEY:
        raise ValueError("TMDB_API_KEY is missing from .env")

    links_df = pd.read_csv(links_path)

    # Keep movies that have a TMDB ID
    links_df = links_df.dropna(subset=["tmdbId"]).copy()
    links_df["tmdbId"] = links_df["tmdbId"].astype(int)

    # Reuse previously fetched records when rerunning
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_ids = set(existing_df["tmdbId"].dropna().astype(int))
        records = existing_df.to_dict("records")
    else:
        existing_ids = set()
        records = []

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {TMDB_API_KEY}",
        "accept": "application/json",
    })
    
    count = 0 
    for row in links_df.itertuples(index=False):
        movie_id = row.movieId
        tmdb_id = row.tmdbId

        if tmdb_id in existing_ids:
            continue

        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"

        try:
            response = session.get(
                url,
                params={
                "api_key": TMDB_API_KEY,
                "language": "en-US",
            },
            timeout=15,
            )

            if response.status_code == 404:
                print(f"Movie not found: TMDB ID {tmdb_id}")
                continue

            response.raise_for_status()
            data = response.json()

            records.append({
                "movieId": movie_id,
                "tmdbId": tmdb_id,
                "overview": data.get("overview") or "",
                "release_date": data.get("release_date") or "",
                "poster_path": data.get("poster_path"),
                "backdrop_path": data.get("backdrop_path"),
            })
            count+=1
            print(count)

            existing_ids.add(tmdb_id)

        except requests.RequestException as error:
            print(f"Request failed for TMDB ID {tmdb_id}: {error}")

        # Save progress so successful requests aren't lost
        pd.DataFrame(records).to_csv(output_path, index=False)
        

    metadata_df = pd.DataFrame(records)
    metadata_df.to_csv(output_path, index=False)

    return metadata_df
    