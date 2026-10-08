from fastapi import FastAPI
from src.pipelines.svd_pipe_line import SVD_pipeline
import pandas as pd 

pl_paths = {
    'model':'src/models/colab_filt_svd',
    'ratings':'src/datasets/ratings.csv',
    'movies':'src/datasets/movies.csv'
}

my_pipe_line = SVD_pipeline(pl_paths['model'],pl_paths['ratings'],pl_paths['movies'])
links_df = pd.read_csv(r'src/datasets/links.csv')
app = FastAPI()

@app.get('/movies')

@app.get('/predict')
async def greet(uid : int , num : int = 60 , year_filt : int|None = None ):
    recommendations = my_pipe_line.get_n_recommendations(uid=uid,n=num,min_year=year_filt)
    merged_records = pd.merge(recommendations,links_df,on='movieId',how='left') 
    records = recommendations.to_dict(orient="records")

    return records