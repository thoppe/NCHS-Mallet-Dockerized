import pandas as pd
from src.api import MalletLDA

df = pd.read_csv("pp_raw_documents.csv", nrows=200)

LDA = MalletLDA()

tokens = LDA.preprocess(df["text"])
doc_topics, topics, word_weights = LDA.train(tokens)

dx = word_weights
print(dx)

