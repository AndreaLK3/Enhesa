import pandas as pd
from enum import Enum
import Filepaths as F
import nltk
from collections import Counter
import os
import pickle

class Split(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"

class Column(Enum):
    CLASS = "class"
    ARTICLE = "article"

EMBEDDINGS_DIM=300
UNK_TOKEN = 'unk'

# Load train.csv and/or test.csv as Pandas dataframes
def load_split(split_enum):
    if split_enum == Split.TRAINING:
        df = pd.read_csv(F.training_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value])
    elif split_enum == Split.VALIDATION:
        df = None
    elif split_enum == Split.TEST:
        df = pd.read_csv(F.test_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value], index_col=False)
    return df


def get_vocabulary(corpus_df, vocab_fpath, min_frequency, new=False, lowercase=True):

    if os.path.exists(vocab_fpath) and not new:
        with open(vocab_fpath, "rb") as vocab_file:
            vocabulary_ls = pickle.load(vocab_file)
    else:

        # german_stopwords_ls = nltk.corpus.stopwords.words('german')

        articles = corpus_df[Column.ARTICLE.value].to_list()
        vocabulary_counter = Counter()

        for article in articles:
            words = nltk.tokenize.word_tokenize(article, language='german')
            if lowercase:
                words = [w.lower() for w in words]
            vocabulary_counter.update(words)

        vocabulary_ls_0 = list(vocabulary_counter.keys())
        vocabulary_ls = [w for w in vocabulary_ls_0 if vocabulary_counter[w] >= min_frequency]
        if UNK_TOKEN not in vocabulary_ls:
            vocabulary_ls.append(UNK_TOKEN)
        with open(vocab_fpath, "wb") as vocab_file:
            pickle.dump(vocabulary_ls, vocab_file)

    return vocabulary_ls


def get_labels(split_df):
    class_names = list(split_df["class"].value_counts().index)
    labels_ls = []
    for index, row in split_df.iterrows():
        labels_ls.append(class_names.index(row["class"]))

    return labels_ls

