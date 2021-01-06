from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from mallet_interface import MalletLDA, spaCyPreprocess


app = FastAPI()
__version__ = "0.1.0"

LDA = MalletLDA()
LDA_PRE = spaCyPreprocess()


class LDA_input(BaseModel):
    text_input: Optional[List[str]]
    text_tokenized:Optional[List[str]]
    n_topics: int=10


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
    topics, doc_topics, word_weights = LDA.train(
        q.text_tokenized, n_topics = q.n_topics)

    result = {
        "words": word_weights.to_json(orient="split"),
        "topics": topics.to_json(orient="split"),
        "documents": doc_topics.tolist(),
    }
    return result
