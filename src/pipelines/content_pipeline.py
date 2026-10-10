import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Content_pipeline :
    def __int__(self,colab_df : pd.DataFrame,org_df :pd.DataFrame):
        self.colab_df = colab_df,
        self.org_df = org_df

    def get_check_match_title(self,inp_title : str):

        search_title = "xo kitty"
        matches = self.colab_df[
            self.colab_df["title"].str.contains(
            search_title,
            case=False,
            na=False,
            regex=False
        )
        ]
        titles = matches["title"].to_numpy()