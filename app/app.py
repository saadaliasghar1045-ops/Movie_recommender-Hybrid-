from fastapi import FastAPI,HTTPException
from src.pipelines.svd_pipe_line import SVD_pipeline
import httpx
import pandas as pd 
import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

pl_paths = {
    'model':'src/models/colab_filt_svd',
    'ratings':'src/datasets/ratings.csv',
    'movies':'src/datasets/movies.csv'
}

my_pipe_line = SVD_pipeline(pl_paths['model'],pl_paths['ratings'],pl_paths['movies'])
links_df = pd.read_csv(r'src/datasets/links.csv')

##############################

app = FastAPI()

# Endpoint to return the data of movie predicted 
@app.get("/movies/{tmdb_id}")
async def get_movie_details(tmdb_id: int):

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"

    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)

    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail="Movie not found"
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text
        )

    movie = response.json()

    poster_path = movie.get("poster_path")
    backdrop_path = movie.get("backdrop_path")

    poster_url = (
        f"https://image.tmdb.org/t/p/w500{poster_path}"
        if poster_path
        else None
    )

    backdrop_url = (
        f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        if backdrop_path
        else None
    )

    return {
        "tmdbId": movie["id"],
        "title": movie["title"],
        "overview": movie["overview"],
        "release_date": movie["release_date"],
        "poster_url": poster_url,
        "backdrop_url": backdrop_url
    }
    
    
@app.get('/predict')
async def greet(uid : int , num : int = 60 , year_filt : int|None = None ):
    recommendations = my_pipe_line.get_n_recommendations(uid=uid,n=num,min_year=year_filt)
    records = recommendations.to_dict(orient="records")
    return records