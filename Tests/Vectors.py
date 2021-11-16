import Utils
import Filepaths as F
import pandas as pd
import unittest
import Models.LoadVecs as LV
from collections import Counter
import os
import numpy as np

class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.corpus_df = pd.read_csv(F.mytests_file, sep=";",
                                     names=[Utils.Column.CLASS.value, Utils.Column.ARTICLE.value], index_col=False)
        self.corpus_txt_ls = self.corpus_df[Utils.Column.ARTICLE.value].to_list()
        self.vocabulary = Utils.get_vocabulary(self.corpus_df)
        self.vectors_fpath = os.path.join(F.tests_folder, F.mytests_vectors_fname)
        self.vectors_wordslist_fpath = os.path.join(F.tests_folder, F.mytests_vectors_wordslist_fname)
        LV.save_pretrained_vecs_wordslist(self.vectors_fpath, self.vectors_wordslist_fpath)


    def init_word_vectors_pretrained(self):

        vocab_words_ls = list(self.vocabulary.keys())
        vocab_len = len(vocab_words_ls)

        word_embeddings = LV.get_all_pretrained_embs(vocab_words_ls, self.vectors_fpath)
        self.assertEqual(word_embeddings.size, (vocab_len, Utils.EMBEDDINGS_DIM))



if __name__ == '__main__':
    unittest.main()