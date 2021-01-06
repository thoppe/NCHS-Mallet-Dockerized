import streamlit as st
import pandas as pd
import requests
import numpy as np

#import numpy as np
#import wordcloud

# Try this?
# https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21  ... pyLDAvis

num_words = 150
document_limit = 5000

# Display options
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title=("Topic-Model Explorer"),
)

st.title("Topic model explorer")

st.sidebar.title("Options")
n_topics = st.sidebar.slider("Number of Topics", 5, 20, value=10)

# Custom fileupload
st.sidebar.markdown("## Custom dataset")
f_upload = st.sidebar.file_uploader('Upload a CSV, with the target column named "text"')

if f_upload is None:
    f_dataset = "example_data/reddit_suicide_data.csv"
else:
    f_dataset = f_dataset

df = pd.read_csv(f_dataset, nrows=document_limit)

n_documents = len(df)
st.write(f"Loaded {n_documents:,} documents into memory.")

@st.cache
def preprocess_input(f_dataset):
    with st.spinner("*Preprocessing text with spaCy*"):
        url = "http://127.0.0.1:8000/LDA/preprocess"
        params = {"text_input": df["text"].values.tolist()}

    r = requests.get(url, json=params)
    js = r.json()
    
    return js

@st.cache
def train_tokenized(tokenized, n_topics):

    data_input = {
        "text_tokenized" : tokenized['text_tokenized'],
        'n_topics' : n_topics,
    }
    
    with st.spinner("*Running MALLET*"):
        url = "http://127.0.0.1:8000/LDA/train"
        r = requests.get(url, json=data_input)
        js = r.json()

        words = pd.read_json(js["words"], orient="split")
        topics = pd.read_json(js["topics"], orient="split")
        docs = np.array(js["documents"])

    return words, topics, docs


tokenized = preprocess_input(f_dataset)
words, topics, docs = train_tokenized(tokenized, n_topics)


import wordcloud
import itertools

imgs = []
cols = st.beta_columns(3)
col = itertools.cycle(cols)

for topicID, dx in words.groupby("topicID"):
    dx = dx.sort_values("weight", ascending=False)
    
    freq = {k: v for k, v in zip(dx.word, dx.weight)}
    WC = wordcloud.WordCloud()
    w = WC.generate_from_frequencies(freq)
    img = w.to_array()
    
    active_column = next(col)
    active_column.image(img, f"Topic {topicID}", use_column_width=True)



tmp='''

status = [st.empty()] * 4

text = interface.preprocess_text(df)
status[1].write(f"Finished preprocessing text.")

words = interface.train_lda(text, num_topics, num_words=num_words)
status[3].write(f"Finished LDA")

# Clean the computation window
for item in status:
    item.empty()

dx = pd.DataFrame()

for i in range(num_topics):
    dx[f"topic{i}"] = list(zip(*words[i]))[0]
    dx[f"w{i}"] = list(zip(*words[i]))[1]
    dx[f"w{i}"] *= 100

labels = [f"w{i}" for i in range(num_topics)]
tableviz = (
    dx[:10]
    .style.background_gradient(cmap="Blues", subset=labels)
    .format("{:0.2f}", subset=labels)
)
st.write(tableviz, index=False)

import itertools

cols = st.beta_columns(2)
col = itertools.cycle(cols)

for i, wordset in enumerate(words):
    freq = {k: v for k, v in wordset}
    WC = wordcloud.WordCloud()
    w = WC.generate_from_frequencies(freq)
    img = w.to_array()

    active_column = next(col)
    active_column.image(img, f"Topic {i}", use_column_width=True)
'''
