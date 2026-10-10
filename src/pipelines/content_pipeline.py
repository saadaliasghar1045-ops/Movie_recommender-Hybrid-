import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Content_pipeline :
    def __int__(self,colab_df : pd.DataFrame,org_df :pd.DataFrame , tfidf_matrix):
        self.colab_df = colab_df,
        self.org_df = org_df
        self.tfidf_matrix = tfidf_matrix

    @staticmethod
    # getting titles that match with user entered title
    def matched_titles(df,inp_title : str):

        search_title = inp_title
        matches = df[
            df["title"].str.contains(
                search_title,
                case=False,
                na=False,
                regex=False
            )
        ]

        titles = matches["title"].to_numpy()
        return titles

    @staticmethod
    # getting index of title
    def get_index(df,title):

        match = df.index[    
        df['title'].str.casefold() == title.casefold()
        ].tolist()

        return match

    # function to apply year filter this returns True and False values for every movie_index if movie at index is greater than year_limit it will be true other wise false
    @staticmethod
    def yearly_filtered_movs_idxs (df,limit_year : int):
        filtered_idxs = (
            df['year'].notna() 
            & (df['year'] >= limit_year)
        ).to_numpy()

        return filtered_idxs

    def find_similar_movies(self,limit_year,title):

        matched_titles = matched_titles(title,self.colab_df)
        