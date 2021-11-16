import pandas as pd
from enum import Enum
import Filepaths as F
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class Split(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"

class Column(Enum):
    CLASS = "class"
    ARTICLE = "article"


# Load train.csv and/or test.csv as Pandas dataframes
def load_split(split_enum):
    if split_enum == Split.TRAINING:
        df = pd.read_csv(F.training_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value])
    elif split_enum == Split.VALIDATION:
        df = None
    elif split_enum == Split.TEST:
        df = pd.read_csv(F.test_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value], index_col=False)
    return df

# Green, through yellow, to red
def create_gyr_colormap():
    viridisBig = cm.get_cmap('hsv', 1024)
    newcmp = ListedColormap(viridisBig(np.linspace(0.38, 0.08, 1024)))

    return newcmp

def save_figure(fig, out_fpath):
    fig.tight_layout()
    fig.savefig(out_fpath)

# Auxiliary function: use CountVectorizer's tokenization to get the vocabulary (words, no frequencies) of each class
def get_class_vocabularies(training_df, class_names):

    classes_words_ls = []
    for c_name in class_names:
        class_articles = training_df[training_df[Column.CLASS.value] == c_name][
            Column.ARTICLE.value].to_list()
        vectorizer = CountVectorizer(lowercase=False)
        vectorizer.fit(class_articles)
        words_in_class = set(vectorizer.vocabulary_.keys())
        classes_words_ls.append(words_in_class)
    return classes_words_ls

# Gives the TF-IDF value for all the words of a document in a text corpus
def get_document_tf_idf(document, tfidf_vectorizer):

    count_vectorizer = CountVectorizer(lowercase=False)
    count_vectorizer.fit(document)
    doc_words = list(count_vectorizer.vocabulary_.keys())
    doc_tfidf = []
    for w in doc_words:
        w_idx = tfidf_vectorizer.vocabulary_[w]
        w_tfidf = tfidf_vectorizer.idf_[w_idx]
        doc_tfidf.append(w_tfidf)
    return doc_tfidf


def get_labels(split_df):
    class_names = list(split_df["class"].value_counts().index)
    labels_ls = []
    for index, row in split_df.iterrows():
        labels_ls.append(class_names.index(row["class"]))

    return labels_ls

