from fastapi import FastAPI
from src.pipelines.svd_pipe_line import SVD_pipeline

pl_paths = {
    'model':'src/models/colab_filt_svd',
    'ratings':'src/datasets/ratings.csv',
    'movies':'src/datasets/movies.csv'
}

my_pipe_line = SVD_pipeline(pl_paths['model'],pl_paths['ratings'],pl_paths['movies'])

app = FastAPI()


@app.get('/predict')
async def greet(uid : int , num : int = 60 , year_filt : int|None = None ):
    recommendations = my_pipe_line.get_n_recommendations(uid=uid,n=num,min_year=year_filt)
    records = recommendations.to_dict(orient="records")

    return records