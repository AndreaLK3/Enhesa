import Utils
import Filepaths as F
import pandas as pd
import unittest
from collections import Counter
import os

class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.corpus_df = pd.read_csv(F.mytests_file, sep=";",
                                     names=[Utils.Column.CLASS.value, Utils.Column.ARTICLE.value], index_col=False)
        self.vocab_fpath = os.path.join(F.tests_folder, F.vocabulary_fname)


    def test_get_vocabulary(self):
        self.my_mini_vocab = set(['small', ',', 'experimental', 'hydrogen-powered', 'planes', 'are', 'paving', 'the',
            'way', 'for', 'net-zero', 'carbon', 'aviation', 'by', '2050', '.', 'but', 'route', 'is', 'rocky',
            'northern', 'ireland', '’', 's', '0-0', 'draw', 'with', 'italy', 'means', 'azzurri', 'will', 'have', 'to',
            'rely', 'on', 'play-offs', 'secure', 'a', 'place', 'at', '2022', 'world', 'cup', 'as', 'we', 'debate',
            'negative', 'effects', 'of', 'social', 'media', 'consider', 'earliest', 'and', 'arguably', 'most',
    'prevalent' , 'that', 'use', 'internet', 'connect', 'other', 'people', ':', 'chat', 'der', 'random', 'words', 'unk']
)

        self.assertEqual(set(Utils.get_vocabulary(self.corpus_df, self.vocab_fpath, new=True, min_frequency=1)),
                         self.my_mini_vocab)

    def test_min_frequency(self):
        self.my_mini_vocab_freq2 = set([',', 'the', 'way', '.', 'with', 'to', 'we'])
        self.assertEqual(set(Utils.get_vocabulary(self.corpus_df, self.vocab_fpath, new=True, min_frequency=2)),
                         self.my_mini_vocab_freq2)


if __name__ == '__main__':
    unittest.main()