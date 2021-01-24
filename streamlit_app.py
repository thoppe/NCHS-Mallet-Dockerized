import streamlit as st
import pandas as pd
import requests
import numpy as np
import wordcloud
import itertools


# Try this?
# https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21  ... pyLDAvis

num_words = 150
document_limit = 5000
st_time_to_live = 4 * 3600

# Display options
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title=("Topic-Model Explorer"),
)

st.title("Topic model explorer")

st.sidebar.title("Options")
n_topics = st.sidebar.slider("Number of Topics", 5, 40, value=30)

n_sort_topic = st.sidebar.slider("Topic sort order by", 0, n_topics - 1)

# Custom fileupload
st.sidebar.markdown("## Custom dataset")
f_upload = st.sidebar.file_uploader('Upload a CSV, with the target column named "text"')

if f_upload is None:
    f_dataset = "example_data/reddit_suicide_data.csv"
else:
    f_dataset = f_upload

df = pd.read_csv(f_dataset, nrows=document_limit)

n_documents = len(df)


if f_upload:
    st.write(f"Loaded {n_documents:,} documents from `{f_dataset.name}`")
else:
    st.write(f"Loaded {n_documents:,} documents from `{f_dataset}`")


@st.cache(ttl=st_time_to_live)
def preprocess_input(f_dataset):
    with st.spinner("*Preprocessing text with spaCy*"):
        url = "http://127.0.0.1:8000/LDA/preprocess"
        params = {"text_input": df["text"].values.tolist()}

    r = requests.get(url, json=params)
    js = r.json()

    return js


@st.cache(ttl=st_time_to_live)
def train_tokenized(tokenized, n_topics):

    data_input = {
        "text_tokenized": tokenized["text_tokenized"],
        "n_topics": n_topics,
    }

    with st.spinner("*Running MALLET*"):
        url = "http://127.0.0.1:8000/LDA/train"
        r = requests.get(url, json=data_input)
        js = r.json()

        words = pd.read_json(js["words"], orient="split")
        topics = pd.read_json(js["topics"], orient="split")
        docs = np.array(js["documents"])

    return words, topics, docs


@st.cache(ttl=st_time_to_live)
def compute_wordclouds(words):
    imgs = []

    for topicID, dx in words.groupby("topicID"):
        dx = dx.sort_values("weight", ascending=False)

        freq = {k: v for k, v in zip(dx.word, dx.weight)}
        WC = wordcloud.WordCloud(
            max_words=2000,
            prefer_horizontal=1,
            relative_scaling=0.75,
            max_font_size=60,
            random_state=13,
        )
        w = WC.generate_from_frequencies(freq)
        img = w.to_array()

        imgs.append(img)

    return imgs


tokenized = preprocess_input(f_dataset)
words, topics, docs = train_tokenized(tokenized, n_topics)

WC = compute_wordclouds(words)


for topic_idx in range(n_topics):
    col0, col1 = st.beta_columns((1, 2))

    col0.image(WC[topic_idx], f"Topic {topic_idx}", use_column_width=True)

    dx = pd.DataFrame(docs).sort_values(topic_idx, ascending=False)[:5]
    dx *= 100
    dx.insert(loc=0, column="text", value=df["text"])
    dx = dx[["text", topic_idx]]
    dx["text"] = dx["text"].str[:300]
    labels = list(range(n_topics))
    labels = [topic_idx]
    tableviz = dx.style.background_gradient(cmap="Blues", subset=labels).format(
        "{:0.0f}", subset=labels
    )

    col1.table(tableviz)


with st.beta_expander(label="Word Clouds", expanded=True):
    cols = st.beta_columns(3)
    col = itertools.cycle(cols)

    for i, img in enumerate(compute_wordclouds(words)):
        active_column = next(col)
        active_column.image(img, f"Topic {i}", use_column_width=True)

with st.beta_expander(label="Document Labels (top 200)", expanded=True):
    dx = pd.DataFrame(docs).sort_values(n_sort_topic, ascending=False)[:200]
    dx *= 100

    dx.insert(loc=0, column="text", value=df["text"])
    labels = list(range(n_topics))
    tableviz = dx.style.background_gradient(cmap="Blues", subset=labels).format(
        "{:0.0f}", subset=labels
    )
    st.table(tableviz)
