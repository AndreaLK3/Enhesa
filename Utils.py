import pandas as pd
from enum import Enum
import Filepaths as F
import nltk
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

class Split(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"

class Column(Enum):
    CLASS = "class"
    ARTICLE = "article"

EMBEDDINGS_DIM=300

# Load train.csv and/or test.csv as Pandas dataframes
def load_split(split_enum):
    if split_enum == Split.TRAINING:
        df = pd.read_csv(F.training_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value])
    elif split_enum == Split.VALIDATION:
        df = None
    elif split_enum == Split.TEST:
        df = pd.read_csv(F.test_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value], index_col=False)
    return df


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


def get_vocabulary(corpus_df):
    articles = corpus_df[Column.ARTICLE.value].to_list()
    vocabulary_counter = Counter()
    for article in articles:
        words = nltk.tokenize.word_tokenize(article, language='german')
        vocabulary_counter.update(words)

    return vocabulary_counter

def get_labels(split_df):
    class_names = list(split_df["class"].value_counts().index)
    labels_ls = []
    for index, row in split_df.iterrows():
        labels_ls.append(class_names.index(row["class"]))

    return labels_ls

