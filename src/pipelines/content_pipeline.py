import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Content_pipeline :

    def __init__ (
        self,
        content_df : pd.DataFrame,
        org_df :pd.DataFrame,
        vectorizer_for_title_search,
        vectorizer_for_description,
        tfidf_matrix_for_title_search,
        tfidf_matrix_for_description,
        tfidf_matrix_for_title
    ):
        # data frames
        self.content_df = content_df.copy()
        self.org_df = org_df
        # vectorizers
        self.vectorizer_for_title_search = vectorizer_for_title_search
        self.vectorizer_for_description = vectorizer_for_description
        # tfidf_matrices
        self.tfidf_matrix_for_title_search = tfidf_matrix_for_title_search
        self.tfidf_matrix_for_description = tfidf_matrix_for_description
        self.tfidf_matrix_for_title = tfidf_matrix_for_title
        
    # getting index of title
    @staticmethod
    def get_index(df,title):

        match = df.index[    
        df['title'].str.casefold() == title.casefold()
        ].tolist()

        return match[0]

    # function to apply year filter this returns True and False values for every movie_index if movie at index is greater than year_limit it will be true other wise false
    @staticmethod
    def yearly_filtered_movs_idxs (df,limit_year : int | None):
        filtered_idxs = (
            df['year'].notna() 
            & (df['year'] >= limit_year)
        ).to_numpy()

        return filtered_idxs

    # getting titles that match with user entered title
    @staticmethod
    def get_matched(title : str ,t_tfidf_matrix,title_vectorizer,df : pd.DataFrame):
        title_vector = title_vectorizer.transform([title])
        t_similarity = cosine_similarity(
            title_vector,
            t_tfidf_matrix
        ).ravel()
        
        top_matched_idxs = t_similarity.argsort()[::-1][:3]
        
        top_matched_t = df.iloc[top_matched_idxs]['title']
        return top_matched_t.to_numpy()
    
    def temp_find_similar_by_title(self , inp_title , n_recoms : int = 20 , limit_year : int | None = None):

        # creating empty recommended movies frame :
        columns = ['movieId','similarity_score']
        rec_movs_df = pd.DataFrame(columns=columns)

        matched_titles = self.get_matched(
            title=inp_title,
            title_vectorizer=self.vectorizer_for_title_search,
            df=self.content_df,
            t_tfidf_matrix=self.tfidf_matrix_for_title_search
        )
        
        n_matched_titles = matched_titles.shape[0]

        if n_matched_titles != 0:
            
            recoms_per_matched_title = n_recoms
            
            for title in matched_titles :
 
                crr_idx = self.get_index(self.content_df,title)
                similarity = cosine_similarity(
                    self.tfidf_matrix_for_title[crr_idx],
                    self.tfidf_matrix_for_title
                ).ravel()

                # applying year filter
                if limit_year :
                    eligible = self.yearly_filtered_movs_idxs(self.content_df,limit_year=limit_year)
                    similarity[~eligible] = -1
                    
                # removing similarity of input title, from similarity and movies that are below year limit
                similarity[crr_idx] = -1
                top_indxs_per_title = similarity.argsort()[::-1][:recoms_per_matched_title]
                rec_movs = self.content_df.iloc[top_indxs_per_title][
                    ['movieId']    
                ]

                rec_movs['similarity_score'] = similarity[top_indxs_per_title]
                rec_movs_df = pd.concat([rec_movs_df,rec_movs],ignore_index=True)
            
        else : 
            return None
        
        rec_movs_df.drop_duplicates(subset=['movieId'],inplace=True)
        top_recommendations = rec_movs_df.sort_values(
            by='similarity_score',
            ascending=False
        ).head(n_recoms)
        
        return top_recommendations
    
    # getting similar movies on top of title
    def find_similar_by_title(self , inp_title : str , n_recoms : int = 20 , limit_year : int | None = None):

        # Here I am getting movie index and finding its cosine similarity with other movies vector 
        movie_index = self.get_index(self.content_df,inp_title)
        similarity = cosine_similarity(
            self.tfidf_matrix_for_title[movie_index],
            self.tfidf_matrix_for_title
        ).ravel()
                
        # applying year filter if it exist 
        if limit_year :
            eligible = self.yearly_filtered_movs_idxs(self.content_df,limit_year=limit_year)
            similarity[~eligible] = -1
            
        # removing the searched movie from similarity matrix 
        similarity[movie_index] = -1
        
        # here I am getting top n recommended movies index 
        top_indxs_per_title = similarity.argsort()[::-1][:n_recoms]
        
        # Now I am using the index to access the predicted movies Id's from data frame
        rec_movs = self.content_df.iloc[top_indxs_per_title][
        ['movieId']    
        ]
            
        # At last I added a new column in rec_movs data frame that contains the similarity score 
        rec_movs['similarity_score'] = similarity[top_indxs_per_title]
        return rec_movs
        
    # matching movies based on top of description
    def find_similar_by_description(self,inpt_desc : str ,n_recoms : int  = 20, limit_year : int|None = None  ):
        # converting movie description into vector
        inp_desc_vect = self.vectorizer_for_description.transform([inpt_desc.lower()])
        desc_similarity = cosine_similarity(
            inp_desc_vect,
            self.tfidf_matrix_for_description
        ).ravel()
         
         # now applying year_filter 
        if limit_year :
            eligible = self.yearly_filtered_movs_idxs(self.content_df,limit_year)
            desc_similarity[~eligible] = -1
            
        # getting top movies indexes
        top_indexes = desc_similarity.argsort()[::-1][:n_recoms]
        
        # getting top movies
        top_recoms = self.content_df.iloc[top_indexes][['movieId']]
        top_recoms['similarity_score'] = desc_similarity[top_indexes]
        return top_recoms
             
    def get_movies_info(self,predicted):
        movies_info = pd.merge(predicted,self.org_df,on='movieId',how='left')
        return movies_info