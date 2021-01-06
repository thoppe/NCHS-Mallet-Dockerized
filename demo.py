import pandas as pd
import numpy as np

import requests

df = pd.read_csv("pp_raw_documents.csv", nrows=200)

url = "http://127.0.0.1:8000/LDA/preprocess"
params = {"text_input": df["text"].values.tolist()}

r = requests.get(url, json=params)

js = r.json()


url = "http://127.0.0.1:8000/LDA/train"
params = {"text_input": df["text"].values.tolist()}
r = requests.get(url, json=js)
js = r.json()

words = pd.read_json(js["words"], orient="split")
topics = pd.read_json(js["topics"], orient="split")
docs = np.array(js["documents"])
print(words)
print(topics)
print(docs.shape)
