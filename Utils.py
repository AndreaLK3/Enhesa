import pandas as pd
from enum import Enum
import Filepaths as F
import nltk
from collections import Counter
import os
import pickle
import logging
import sys

class Split(Enum):
    TRAIN = "train"  # refers to the original train.csv file in the 10kGNAD dataset
    VALIDATION = "validation"
    TEST = "test"

class Column(Enum):
    CLASS = "class"
    ARTICLE = "article"

EMBEDDINGS_DIM=300
UNK_TOKEN = 'unk'



# Load train.csv and/or test.csv as Pandas dataframes
def load_split(split_enum):
    if split_enum == Split.TRAIN:
        df = pd.read_csv(F.train_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value])
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


def info_max_words_in_article(corpus_df):
    articles = corpus_df[Column.ARTICLE.value].to_list()

    max_words = 0
    for article in articles:
        words = nltk.tokenize.word_tokenize(article, language='german')
        num_words = len(words)
        if num_words > max_words:
            max_words = num_words
    return max_words



# Invoked to write a message to a text logfile and also print it
def init_logging(logfilename, loglevel=logging.INFO):
  for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
  logging.basicConfig(level=loglevel, filename=logfilename, filemode="w",
                      format='%(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

  if len(logging.getLogger().handlers) < 2:
      outlog_h = logging.StreamHandler(sys.stdout)
      outlog_h.setLevel(loglevel)
      logging.getLogger().addHandler(outlog_h)