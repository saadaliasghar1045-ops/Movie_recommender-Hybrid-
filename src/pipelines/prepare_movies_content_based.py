import pandas as pd 

def create_clean_df_for_Content_pipeline(movies_df,tags_df):
    new_movies_df = movies_df[['movieId','title','genres']].copy()
    # cleaning movies_df
    new_movies_df['title'] = (
        movies_df['title'].fillna("")
        .astype(str)
        .str.strip()
    )
    new_movies_df['genres'] = (
        new_movies_df['genres'].fillna("")
        .astype(str)
        .str.replace("|"," ")
        .str.strip()
    )
    
    # cleaning tags_df
    new_tags_df = tags_df[["movieId","tag"]].copy()
    new_tags_df.dropna(subset=['movieId','tag'])
    
    new_tags_df['tag'] = (
    new_tags_df['tag']
    .astype(str)
    .str.strip()
    .str.lower()
    )
    
    new_tags_df = new_tags_df[new_tags_df['tag'] != ""]
    
    tags_grouped = (
    new_tags_df.groupby('movieId')['tag']
    .apply(lambda crr_mov_tags : " ".join(crr_mov_tags)) # this lambda function input tags of first movie and join them with " " and then moves to next movie tags and do same upto so on
    .reset_index(name='tags') # it cut out the movieId from the resultant series into separate column and the name="tags" changes the name of remaining tag  series to tagss
)
    
    # merging the tags and movies_df
    movies_df_final = pd.merge(
        new_movies_df,
        tags_grouped,
        on='movieId',
        how='left',
        validate='one_to_one' # validating so that each movieId appears one in both data frames otherwise it will raise error 
    )
    
    # cleaning final frame 
    movies_df_final["tags"] = movies_df_final['tags'].fillna("")
    movies_df_final = movies_df_final.drop_duplicates(subset=["movieId"])
    movies_df_final = movies_df_final.reset_index(drop=True)
    
    return movies_df_final