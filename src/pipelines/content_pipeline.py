import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Content_pipeline :
    def __int__(self,movies_df : pd.DataFrame):
        self.movies_df = movies_df.copy()
        
        # Where there is no title in title column put empty string there 
        self.movies_df['titles'] = (movies_df['title'].fillna(""))
        
        # Where there is no genre in genres column put empty string there 
        self.movies_df['genre'] = (
            movies_df['gen']
        )
        