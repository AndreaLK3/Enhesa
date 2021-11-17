import Model.CorpusOrganizer
import Model.CorpusReader
import Utils
import Filepaths as F
import pandas as pd
import unittest
import Models.LoadVecs as LV
from collections import Counter
import os
import numpy as np

class TestVectors(unittest.TestCase):

    def setUp(self):

        self.corpus_df = pd.read_csv(F.mytests_file, sep=";",
                                     names=[Utils.Column.CLASS.value, Utils.Column.ARTICLE.value], index_col=False)
        self.vocab_fpath = os.path.join(F.tests_folder, F.vocabulary_fname)
        self.vocab_words_ls = Model.CorpusOrganizer.get_vocabulary(self.corpus_df, self.vocab_fpath, min_frequency=1, new=True)
        self.vectors_fpath = os.path.join(F.tests_folder, F.mytests_vectors_fname)
        self.wordembeddings_fpath = os.path.join(F.tests_folder, F.pretrained_wordEmb_fname)

    def test_pretrained_wordvectors(self):

        word_embeddings = LV.assign_pretrained_embs(self.vocab_words_ls, self.vectors_fpath)
        der_index = self.vocab_words_ls.index('der')
        self.assertEqual(word_embeddings[der_index][0], -0.011639)


if __name__ == '__main__':
    unittest.main()