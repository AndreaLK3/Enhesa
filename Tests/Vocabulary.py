import Utils
import Filepaths as F
import pandas as pd
import unittest
from collections import Counter

class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.corpus_df = pd.read_csv(F.mytests_file, sep=";",
                                     names=[Utils.Column.CLASS.value, Utils.Column.ARTICLE.value], index_col=False)
        self.corpus_txt_ls = self.corpus_df[Utils.Column.ARTICLE.value].to_list()


    def test_get_vocabulary(self):
        self.my_mini_vocab = Counter({'the': 9, '.': 4, 'to': 3, ',': 2, 'way': 2, 'with': 2, 'we': 2, 'Small': 1,
            'experimental': 1, 'hydrogen-powered': 1, 'planes': 1, 'are': 1, 'paving': 1, 'for': 1, 'net-zero': 1,
            'carbon': 1, 'aviation': 1, 'by': 1, '2050': 1, 'But': 1, 'route': 1, 'is': 1, 'rocky': 1, 'Northern': 1,
            'Ireland': 1, '’': 1, 's': 1, '0-0': 1, 'draw': 1, 'Italy': 1, 'means': 1, 'Azzurri': 1, 'will': 1,
            'have': 1, 'rely': 1, 'on': 1, 'play-offs': 1, 'secure': 1, 'a': 1, 'place': 1, 'at': 1, '2022': 1,
            'World': 1, 'Cup': 1, 'As': 1, 'debate': 1, 'negative': 1, 'effects': 1, 'of': 1, 'social': 1,
            'media': 1, 'consider': 1, 'earliest': 1, 'and': 1, 'arguably': 1, 'most': 1, 'prevalent': 1,
            'that': 1, 'use': 1, 'internet': 1, 'connect': 1, 'other': 1, 'people': 1, ':': 1, 'chat': 1,
            'der':1, 'random':1, 'words':1, 'UNK':1})
        self.assertEqual(Utils.get_vocabulary(self.corpus_df),
                         self.my_mini_vocab)


if __name__ == '__main__':
    unittest.main()