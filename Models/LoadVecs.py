import os
import re
import Utils
import Filepaths
import pandas as pd
import Filepaths as F
import numpy as np


def assign_pretrained_embs(vocabulary_words_ls, vecs_fpath):

    pretrained_embs_arr = np.random.normal(loc=0.0, scale=0.01, size=(len(vocabulary_words_ls), Utils.EMBEDDINGS_DIM))
    # If some words of the vocabulary are not found in the pre-trained Word2Vec vectors,
    # they remain initialized according to N(0,0.01)s

    with open(vecs_fpath, "r") as vecs_file:
        for line_num, line in enumerate(vecs_file):
            line_ls = line.split()
            word_str = line_ls[0]

            word_1 = re.sub(r"^b'", '', word_str)
            word_2 = re.sub(r"'$", '', word_1)
            try:
                word_idx = vocabulary_words_ls.index(word_2)

                vector_str = line_ls[1:]
                vector_ls = []
                for num_str in vector_str:
                    if num_str.startswith("'"):
                        num_str = num_str[1:-1]
                    num = float(num_str)
                    vector_ls.append(num)
                vector_arr = np.array(vector_ls)
                pretrained_embs_arr[word_idx] = vector_arr
            except ValueError:
                pass
                # print(str(word_2) + " not present in corpus vocabulary")

            if (line_num) % 10000 == 0:
                print("Reading line n."+str(line_num) + " of pretrained vectors file...")

    return pretrained_embs_arr


def compute_word_vectors(pretrained_or_random):
    corpus_df = Utils.load_split(Utils.Split.TRAINING)

    vocab_words_ls = Utils.get_vocabulary(corpus_df, F.vocabulary_fpath, min_frequency=2, new=False)
    vocab_len = len(vocab_words_ls)
    print("Vocabulary size |V|=" + str(vocab_len))

    if (pretrained_or_random):
        vecs_fpath = os.path.join(Filepaths.vectors_folder, Filepaths.vectors_fname)
        word_embeddings = assign_pretrained_embs(vocab_words_ls, vecs_fpath)
        np.save(F.pretrained_wordEmb_fpath, word_embeddings)
    else:  # random, i.e. N(0, 0.01)
        word_embeddings = np.random.normal(loc=0.0, scale=0.01, size=(vocab_len, Utils.EMBEDDINGS_DIM))
        np.save(F.random_wordEmb_fpath, word_embeddings)

    return word_embeddings

# Entry level function: if the word embeddings were already computed, load them.
# Otherwise, compute them (i.e. compute random normal vectors, or assign pretrained vectors).
def get_word_vectors(pretrained_or_random):
    if pretrained_or_random:
        if os.path.exists(F.pretrained_wordEmb_fpath):
            word_embeddings = np.load(F.pretrained_wordEmb_fpath)
        else:
            word_embeddings = compute_word_vectors(pretrained_or_random)
    else:
        if os.path.exists(F.random_wordEmb_fpath):
            word_embeddings = np.load(F.random_wordEmb_fpath)
        else:
            word_embeddings = compute_word_vectors(pretrained_or_random)
    return word_embeddings







