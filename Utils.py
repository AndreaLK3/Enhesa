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

# Load train.csv and/or test.csv as Pandas dataframes
def load_split(split_enum):
    if split_enum == Split.TRAINING:
        df = pd.read_csv(F.training_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value])
    elif split_enum == Split.VALIDATION:
        df = None
    elif split_enum == Split.TEST:
        df = pd.read_csv(F.test_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value], index_col=False)
    return df

# with open('/tmp/demo.pickle', 'wb') as outputfile:
# ...     pickle.dump(counts, outputfile)
# ...
# >>> del counts
# >>> with open('/tmp/demo.pickle', 'rb') as inputfile:
# ...     print(pickle.load(inputfile))

def get_vocabulary(corpus_df, vocab_fpath):
    if os.path.exists(vocab_fpath):
        vocabulary_counter = pickle.load(F.vocabulary_fpath)
    else:
        articles = corpus_df[Column.ARTICLE.value].to_list()
        vocabulary_counter = Counter()
        for article in articles:
            words = nltk.tokenize.word_tokenize(article, language='german')
            vocabulary_counter.update(words)
        pickle.dump(vocabulary_counter, vocab_fpath)

    return vocabulary_counter



def get_labels(split_df):
    class_names = list(split_df["class"].value_counts().index)
    labels_ls = []
    for index, row in split_df.iterrows():
        labels_ls.append(class_names.index(row["class"]))

    return labels_ls

