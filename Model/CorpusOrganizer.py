import os
import pickle
from collections import Counter

import nltk
import numpy as np
import pandas as pd

import Filepaths as F

# From the original training corpus in train.csv, split off 10% to be the validation set,
# and keep the rest in Models/training.csv to be used as the actual training set
from Utils import Column, UNK_TOKEN


def organize_training_corpus(train_corpus_df):

    new_training_rows_ls = []
    validation_rows_ls = []

    num_articles = train_corpus_df.index.stop
    random_assignment_arr = np.random.rand(num_articles)
    validation_threshold = 0.9

    for index, row in train_corpus_df.iterrows():
        if random_assignment_arr[index] > validation_threshold:
            validation_rows_ls.append(row)
        else:
            new_training_rows_ls.append(row)

    new_training_df = pd.DataFrame(new_training_rows_ls)
    validation_df = pd.DataFrame(validation_rows_ls)

    new_training_df.to_csv(F.training_set_file, index=False, sep=";", header=False)
    validation_df.to_csv(F.validation_set_file, index=False, sep=";", header=False)

    return (new_training_df, validation_df)

# Retrieves the vocabulary, or creates it if not present
# Source: train.csv. Tokenizer: NLTK's word_tokenize(language='german'). Default: lowercase
def get_vocabulary(corpus_df, vocab_fpath, min_frequency, new=False, lowercase=True):

    if os.path.exists(vocab_fpath) and not new:
        with open(vocab_fpath, "rb") as vocab_file:
            vocabulary_ls = pickle.load(vocab_file)
    else:
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