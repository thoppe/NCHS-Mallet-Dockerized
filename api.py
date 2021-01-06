from pathlib import Path
import tempfile
import os
import re

import spacy
import pandas as pd
from tqdm import tqdm

# Part of the install
# import nltk
# nltk.download('stopwords')

# Default to NLTK stopwords

# 7.4 seconds
# for doc in nlp.pipe(df.text):
#    print(doc)

# 12.8 seconds
# for text in tqdm(df.text):
#    doc = nlp(text)


class PreprocessLDA:
    def __init__(self):

        self.nlp = spacy.load("en_core_web_sm", disable=[])

        self.kept_ents = set(["PERSON", "FACILITY", "GPE", "LOC"])

        # Hardwiring max phrase size
        self.max_chunk_length = 3

        # Default to NLTK stopwords
        with open("stopwords/NLTK_stopwords.txt") as FIN:
            self.stopwords = FIN.read().split("\n")

    def extract_tokens_and_phrases(self, doc):
        # Given a spaCy document, returns list of tokens, list of phrases

        tokenlist = [
            token.lower_
            for token in doc
            if re.search(r"[^\s_]", token.orth_) is not None
        ]
        entities = [
            entity.text.lower().split()
            for entity in doc.ents
            if entity.label_ in self.kept_ents
            and re.search(r"[^\s_]", entity.text) is not None
        ]

        entitylist = ["_".join(e) for e in entities if len(e) > 1]
        chunklist = self.normalize_phrases(doc.noun_chunks)
        phraselist = chunklist + [e for e in entitylist if e not in chunklist]
        return tokenlist, phraselist

    def normalize_phrases(self, chunks):
        """
        Takes list of chunks from spacy NP chunker, maximum phrase length,
        stoplist. Strips stopwords at start of each phrase (e.g. strings of
        determiners).
        Returns resulting phrases in lowercase that are at most
        max_chunk_length words long (after removing leading stoplist words
        like determiners, e.g. a_small_dog becomes small_dog)
        """

        result = []
        for phrase in chunks:
            tokenlist = [token.lower() for token in phrase.text.split()]
            while len(tokenlist) > 0:
                if tokenlist[0] not in self.stopwords:
                    break
                tokenlist.pop(0)
            if len(tokenlist) > 1 and len(tokenlist) <= self.max_chunk_length:
                result.append("_".join(tokenlist))
        return result

    def filter_token(self, token):
        return (
            token in self.stopwords
            or re.search(r"[^\w_\-]", token) is not None
            or re.search(r"[^\d]", token) is None
            or token == "-"
            or token == "_"
        )

    def encode(self, doc):
        tokens, phrases = self.extract_tokens_and_phrases(doc)
        terms = tokens + phrases
        filtered_terms = [t for t in terms if not self.filter_token(t)]
        return " ".join(filtered_terms)

    def __call__(self, texts):
        for doc in self.nlp.pipe(texts, batch_size=100):
            yield self.encode(doc)


class MalletLDA:
    def __init__(self, model_name="MalletLDA"):

        self.mallet_EXEC = "~/src/mallet-2.0.8/bin/mallet"
        self.model_name = model_name
        self.workdir = tempfile.TemporaryDirectory()

    def f(self, name):
        return str(Path(self.workdir.name) / f"{self.model_name}.{name}")

    def preprocess(self, lines):
        """
        SpaCy preprocessing and NLTK stopword removal. Takes in a list of 
        strings and returns a three-column dataframe of [docID, model_name,
        tokenized_text]. docID is sequentially assigned starting at 1.
        """

        clf = PreprocessLDA()

        dx = pd.DataFrame()

        ITR = tqdm(lines, total=len(lines))
        dx["tokenized_text"] = [clf(line) for line in ITR]

        # Convert to mallet 3-column format: docID<tab>label<tab>text
        dx = df[["tokenized_text"]].fillna("")
        dx["docID"] = range(1, len(df) + 1)
        dx["model_name"] = self.model_name
        dx = dx[["docID", "model_name", "tokenized_text"]]
        dx = dx.set_index("docID")

        return dx

    def _import_file(self, df):
        """
        Takes a dataframe from .preprocess and saves the import file
        into the temp working directory.
        """
        cmd = (
            f"{self.mallet_EXEC} import-file"
            f" --input {self.f('input')}"
            f" --output {self.f('vocab')}"
            r" --token-regex '\S+'"
            " --preserve-case"
            " --keep-sequence"
        )

        df.to_csv(self.f("input"), sep="\t", encoding="utf-8", header=None)
        os.system(cmd)

    def train(self, df, n_topics=10, n_optimize_interval=10, n_interations=1000):
        """
        Runs MALLET against the imported data.
        """

        self._import_file(df)

        cmd = (
            f"{self.mallet_EXEC} train-topics"
            f" --input {self.f('vocab')}"
            f" --num-topics {n_topics}"
            f" --optimize-interval {n_optimize_interval}"
            f" --num-iterations {n_interations}"
            f" --output-model             {self.f('model')}"
            f" --output-doc-topics        {self.f('doc-topics')}"
            f" --output-topic-keys        {self.f('topic-keys')}"
            f" --output-state             {self.f('topic-state.gz')}"
            f" --inferencer-filename      {self.f('inferencer')}"
            f" --word-topic-counts-file   {self.f('word-topic-counts')}"
            f" --topic-word-weights-file  {self.f('topic-word-weights')}"
        )
        os.system(cmd)
        os.system(f"ls -lh {self.workdir.name}")

        doc_topics = pd.read_csv(self.f("doc-topics"), sep="\t", header=None)
        doc_topics = doc_topics[range(2, n_topics + 2)].values
        assert doc_topics.shape == (len(df), n_topics)

        topics = pd.read_csv(self.f("topic-keys"), sep="\t", header=None)
        topics.columns = ["topicID", "alpha", "top_words"]
        assert len(topics) == n_topics

        word_weights = pd.read_csv(self.f("topic-word-weights"), sep="\t", header=None)
        word_weights.columns = ["topicID", "word", "weight"]

        return topics, doc_topics, word_weights


df = pd.read_csv("pp_raw_documents.csv", nrows=200)

LDA = MalletLDA()

tokens = LDA.preprocess(df["text"])
doc_topics, topics, word_weights = LDA.train(tokens)

dx = word_weights
print(dx)

exit()
