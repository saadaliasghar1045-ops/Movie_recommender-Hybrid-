from fastapi import FastAPI,HTTPException
from src.pipelines.content_pipeline import Content_pipeline
from app.schemas.recommendation_input import recommendation_input
from app.schemas.title_match_inp import title_match_inp
import joblib
import httpx
import pandas as pd 
import os

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