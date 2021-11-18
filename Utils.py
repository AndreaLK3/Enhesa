import pandas as pd
from enum import Enum
import Filepaths as F
import nltk
import logging
import sys
import torch
import os

# Enums
class Split(Enum):
    TRAIN = "train"  # refers to the original train.csv file in the 10kGNAD dataset
    VALIDATION = "validation"
    TEST = "test"

class Column(Enum):
    CLASS = "class"
    ARTICLE = "article"

# Constants
EMBEDDINGS_DIM=200
UNK_TOKEN = 'unk'
CLASS_NAMES = ['Etat', 'Inland', 'International', 'Kultur', 'Panorama', 'Sport', 'Web', 'Wirtschaft', 'Wissenschaft']
              # sorted list of class names, in case we can not retrieve it from a dataset

# Load train.csv and/or test.csv as Pandas dataframes
def load_split(split_enum):
    if split_enum == Split.TRAIN:
        df = pd.read_csv(F.train_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value])
    elif split_enum == Split.VALIDATION:
        df = None
    elif split_enum == Split.TEST:
        df = pd.read_csv(F.test_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value], index_col=False)
    return df


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

# Round the numbers in a list
def round_list_elems(ls, precision=2):
    rounded_ls = [round(elem, precision) for elem in ls]
    return rounded_ls

# Side info: in a given corpus, how many words are in the longest article?
def info_max_words_in_article(corpus_df):
    articles = corpus_df[Column.ARTICLE.value].to_list()

    max_words = 0
    for article in articles:
        words = nltk.tokenize.word_tokenize(article, language='german')
        num_words = len(words)
        if num_words > max_words:
            max_words = num_words
    return max_words


def load_model(lr):
    model_fname = "Model_" + "lr" + str(lr) + ".pt"
    saved_model_fpath = os.path.join(F.models_folder, F.saved_models_subfolder, model_fname)
    model = torch.load(saved_model_fpath)
    return model