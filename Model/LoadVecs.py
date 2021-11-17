import os

import Model.CorpusOrganizer
import Model.CorpusReader
import Utils
import Filepaths
import pandas as pd
import Filepaths as F
import numpy as np

# Compute the randomly initialized ~N(0,0.01) word vectors
def compute_word_vectors(vocab_words_ls):

    vocab_len = len(vocab_words_ls)
    print("Vocabulary size |V|=" + str(vocab_len))

    word_embeddings = np.random.normal(loc=0.0, scale=0.01, size=(vocab_len, Utils.EMBEDDINGS_DIM))
    np.save(F.random_wordEmb_fpath, word_embeddings)

    return word_embeddings

# Entry level function: if the word embeddings were already computed, load them.
# Otherwise, compute them
def get_word_vectors():
    corpus_df = Utils.load_split(Utils.Split.TRAIN)
    vocab_words_ls = Model.CorpusOrganizer.get_vocabulary(corpus_df, F.vocabulary_fpath, min_frequency=2, new=False)

    if os.path.exists(F.random_wordEmb_fpath):
        word_embeddings = np.load(F.random_wordEmb_fpath)
    else:
        word_embeddings = compute_word_vectors(vocab_words_ls)

    return word_embeddings







