import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Content_pipeline :

    def __init__ (self , content_df : pd.DataFrame , org_df :pd.DataFrame , tfidf_matrix ):
        self.content_df = content_df.copy()
        self.org_df = org_df
        self.tfidf_matrix = tfidf_matrix

   
    # getting titles that match with user entered title
    @staticmethod
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

    # getting index of title
    @staticmethod
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

    # getting similar movies
    def find_similar_movies(self,limit_year,inp_title,n_recoms):

        # creating empty recommended movies frame :
        columns = ['movieId','title','similarity_score']
        total_rec_movs_df = pd.DataFrame(columns=columns)

        matched_titles = self.matched_titles(self.content_df,inp_title)
        n_matched_titles = matched_titles.shape[0]
        
        if n_matched_titles != 0:
            recoms_per_title = n_recoms /n_matched_titles
            for title in matched_titles :

                crr_idx = self.get_index(self.content_df,title)
                similarity = cosine_similarity(
                    self.tfidf_matrix[crr_idx],
                    self.tfidf_matrix
                ).ravel()

                # removing similarity of input title, from similarity and movies that are not eligible
                eligible = self.yearly_filtered_movs_idxs(self.content_df,limit_year=limit_year)
                similarity[~eligible] = -1

                # similarity[crr_idx] = -1
                top_movie_indxs = similarity.argsort()[::-1][:n_recoms]
                rec_movs = self.content_df.iloc[top_movie_indxs][
                    ['movieId','title']    
                ]

                rec_movs['similarity_score'] = similarity[top_movie_indxs]
                total_rec_movs_df = pd.concat([total_rec_movs_df,rec_movs],ignore_index=True)
            
        else : 
            return None
        
        return total_rec_movs_df