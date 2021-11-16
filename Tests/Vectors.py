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
        self.vocab_fpath = os.path.join(F.tests_folder, F.vocabulary_fname)
        self.vocabulary = Utils.get_vocabulary(self.corpus_df, self.vocab_fpath)
        self.vectors_fpath = os.path.join(F.tests_folder, F.mytests_vectors_fname)
        self.wordembeddings_fpath = os.path.join(F.tests_folder, F.pretrained_wordEmb_fname)

    def test_pretrained_wordvectors(self):

        vocab_words_ls = list(self.vocabulary.keys())
        # vocab_len = len(vocab_words_ls)

        word_embeddings = LV.assign_pretrained_embs(vocab_words_ls, self.vectors_fpath)
        unk_index = vocab_words_ls.index('UNK')
        der_index = vocab_words_ls.index('der')
        self.assertEqual(word_embeddings[unk_index][0], -0.07903)

        np.save(self.wordembeddings_fpath, word_embeddings)
        word_embeddings = np.load(self.wordembeddings_fpath)
        self.assertEqual(word_embeddings[der_index][0], -0.011639)


if __name__ == '__main__':
    unittest.main()