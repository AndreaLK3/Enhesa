import os
import re
import Utils
import Filepaths
import pandas as pd
import Filepaths as F
import numpy as np


def save_pretrained_vecs_wordslist(vecs_fpath, out_fpath):

    with open(vecs_fpath, "r") as vecs_file:
        with open(out_fpath, "w") as out_file:
            for line in vecs_file:
                line_ls = line.split()
                word_str = line_ls[0]

                word_1 = re.sub(r"^b'", '', word_str)
                word_2 = re.sub(r"'$", '', word_1)
                out_file.write(word_2)



def get_all_pretrained_embs(vocabulary_words_ls, vecs_fpath):

    pretrained_embs_arr = np.random.normal(loc=0.0, scale=0.01, size=(vocab_len, Utils.EMBEDDINGS_DIM))
    # If some words of the vocabulary are not found in the pre-trained Word2Vec vectors,
    # they remain initialized according to N(0,0.01)s

    with open(vecs_fpath, "r") as vecs_file:
        for line in vecs_file:
            line_ls = line.split()
            word_str = line_ls[0]

            word_1 = re.sub(r"^b'", '', word_str)
            word_2 = re.sub(r"'$", '', word_1)

            vector_str = line_ls[1:]
            vector_ls = []
            for num_str in vector_str:
                if num_str.startswith("'"):
                    num_str = num_str[1:-1]
                num = float(num_str)
                vector_ls.append(num)
            vector_arr = np.array(vector_ls)
            try:
                word_idx = vocabulary_words_ls.index(word_2)
                pretrained_embs_arr[word_idx] = vector_arr
            except ValueError:
                pass
                # print(str(word_2) + " not present in corpus vocabulary")
        print("*")
    return pretrained_embs_arr


def init_word_vectors(pretrained_or_random):
    corpus_df = Utils.load_split(Utils.Split.TRAINING)
    corpus_txt_ls = corpus_df[Utils.Column.ARTICLE.value].to_list()

    vocabulary = Utils.get_vocabulary(corpus_df)
    vocab_words_ls = list(vocabulary.keys())
    vocab_len = len(vocab_words_ls)

    if not (pretrained_or_random):
        word_embeddings = np.random.normal(loc=0.0, scale=0.01, size=(vocab_len, Utils.EMBEDDINGS_DIM))
    else:  # pretrained
        vecs_fpath = os.path.join(Filepaths.vectors_folder, Filepaths.vectors_fname)
        word_embeddings = get_all_pretrained_embs(vocab_words_ls, vecs_fpath)

    return word_embeddings.shape






