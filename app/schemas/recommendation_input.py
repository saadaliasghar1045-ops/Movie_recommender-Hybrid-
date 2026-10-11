from pydantic import BaseModel

class recommendation_input(BaseModel):
    inp_text : str 
    n_recoms : int = 20
    limit_year : int | None = None
    by_description : bool = True
    