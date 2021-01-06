from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from mallet import MalletLDA, spaCyPreprocess


app = FastAPI()
__version__ = "0.1.0"

LDA = MalletLDA()
LDA_PRE = spaCyPreprocess()

class LDA_input(BaseModel):
    text_input : List[str]
    text_tokenized : Optional[List[str]]

class LDA_response(BaseModel):
    pass

@app.get("/LDA/preprocess")
def preprocess(q: LDA_input) -> LDA_input:
    """
    Preprocesses the text getting it ready for the model. Fills in
    q.text_tokenized from q.text_input.
    """
    q.text_tokenized = list(LDA_PRE(q.text_input))
    return q

@app.get("/LDA/train")
def train(q: LDA_input):
    topics, doc_topics, word_weights = LDA.train(q.text_tokenized)

    print(word_weights.index.dtype)
    
    result = {
        "words" : word_weights.to_json(orient='split'),
        "topics" : topics.to_json(orient='split'),
        "documents" : doc_topics.tolist(),
        
    }
    return result
