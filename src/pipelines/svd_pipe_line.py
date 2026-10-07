import pandas as pd 
from surprise import dump

class SVD_pipeline:
    def __init__(self,model_path,ratings_path,movies_path):
        _, self.model = dump.load(model_path)
        
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        self.all_movies_id = set(self.movies_df['movieId'].unique())
        
    def get_user_unrated_movies(self,uid) :
        user_ratings = self.ratings_df[self.ratings_df['userId'] == uid]
        
        if user_ratings.empty:
            return list(self.all_movies_id)
        
        user_rated_movies = set(user_ratings['movieId'].unique())
        return list(self.all_movies_id - user_rated_movies)
    
    def get_n_recommendations(self,uid,n:int = 10):
        unrated_movies_id = self.get_user_unrated_movies(uid)
        predictions = [self.model.predict(uid,movie_id) for movie_id in unrated_movies_id ]
        
        predictions.sort(key = lambda x : x.est , reverse=True ) # key means what value to sort by 
        top_predictions = predictions[:n]
        
        # predictions on movies obtained now we will separate movies id and the estimated score from them 
        top_data = [
            {"movieId": pred.iid, "predicted_rating": round(pred.est, 2)} 
            for pred in top_predictions
        ]
        
        # now we will create new dataframe using the top_data dictionary 
        top_movies_df = pd.DataFrame(top_data)
        # now we will merge it based on movieId so that all features of preicted movies be obtained like author and release year
        final_df = pd.merge(top_movies_df,self.movies_df,on='movieId',how='inner') 
        
        return final_df[['movieId', 'title', 'genres', 'predicted_rating','year']]

        