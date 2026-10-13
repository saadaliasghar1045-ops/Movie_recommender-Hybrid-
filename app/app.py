from fastapi import FastAPI,HTTPException,status
from src.pipelines.content_pipeline import Content_pipeline
from app.schemas.recommendation_input import recommendation_input
from app.schemas.title_match_inp import title_match_inp
from fastapi.middleware.cors import CORSMiddleware

import joblib
import httpx
import pandas as pd 
import os
from dotenv import load_dotenv


load_dotenv()
MY_API_KEY = os.getenv('MY_API_KEY')


pl_paths = {
    'desc_vectorizer':'src/vectorizers_and_matrices/vectorizer_for_description.joblib',
    'title_search_vectorizer':'src/vectorizers_and_matrices/vectorizer_for_title_search.joblib',
    'desc_tfidf':'src/vectorizers_and_matrices/tfidf_matrix_for_description.joblib',
    'title_search_tfidf':'src/vectorizers_and_matrices/tfidf_matrix_for_title_search.joblib',
    'title_tfidf':'src/vectorizers_and_matrices/tfidf_matrix_for_title.joblib',
    'content_df':'src/datasets_for_app/frame_of_content_filt_for_title.csv',
    'movies_info_df':'src/datasets_for_app/movies_info.csv',
    'top_movies':'src/datasets_for_app/top_movies.csv'
}

# vectorizers and matrices 
desc_vectorizer = joblib.load(pl_paths['desc_vectorizer'])
title_search_vectorizer = joblib.load(pl_paths['title_search_vectorizer'])
desc_tfidf = joblib.load(pl_paths['desc_tfidf'])
title_search_tfidf = joblib.load(pl_paths['title_search_tfidf'])
title_tfidf = joblib.load(pl_paths['title_tfidf'])

# data frames 
content_df = pd.read_csv(pl_paths['content_df'])
movie_info_df = pd.read_csv(pl_paths['movies_info_df'])
top_movies = pd.read_csv(pl_paths['top_movies'])


recom_pipeline = Content_pipeline(
    content_df=content_df,
    movies_info_df=movie_info_df,
    vectorizer_for_description=desc_vectorizer,
    vectorizer_for_title_search=title_search_vectorizer,
    tfidf_matrix_for_description=desc_tfidf,
    tfidf_matrix_for_title_search=title_search_tfidf,
    tfidf_matrix_for_title=title_tfidf
)

##############################

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://saadaliasghar1045-ops.github.io",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/home')
async def greet():
    return 'content based movie recommender creaeted by saad ali asghar'
    
@app.post('/recommend')
async def recommend(recommendation_input : recommendation_input):
    if recommendation_input.by_description :
        mov_sim_by_desc = recom_pipeline.find_similar_by_description(
            inpt_desc= recommendation_input.inp_text,
            limit_year=recommendation_input.limit_year,
            n_recoms=recommendation_input.n_recoms
        )
        mov_sim_by_desc_info = recom_pipeline.get_movies_info(mov_sim_by_desc).to_dict(orient="records")
        return mov_sim_by_desc_info
        
    else : 
        mov_sim_by_title = recom_pipeline.find_similar_by_title(
            inp_title=recommendation_input.inp_text,
            limit_year=recommendation_input.limit_year,
            n_recoms=recommendation_input.n_recoms
        )
        mov_sim_by_title_info = recom_pipeline.get_movies_info(mov_sim_by_title).to_dict(orient="records")
        return mov_sim_by_title_info
    
@app.post('/match_title')
async def match_title(title_match_inp : title_match_inp):
    matched_titles = recom_pipeline.get_matched(
        title=title_match_inp.title,
        t_tfidf_matrix=title_search_tfidf,
        title_vectorizer=title_search_vectorizer,
        df=content_df
    )
    return matched_titles

@app.post('/trending')
async def trending():
    return top_movies.to_dict(orient="records")

@app.get('/movie_info/{tmdb_id}')
async def movie_info(tmdb_id : int):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            url=url,
            params={
                'api_key':MY_API_KEY,
                "append_to_response": "videos,credits,images"
            }
            
        )
        
        if response.status_code == 404 :
            raise HTTPException(status_code=404,detail='movie not found at this tmdb_id {tmdb_id}')
        else :
            return response.json()