import Filepaths as F
import Utils
import nltk
import string
import numpy as np

def get_article_indices(article_tokens, vocabulary_ls):
    indices_ls = []
    for tok in article_tokens:
        try:
            idx = vocabulary_ls.index(tok)
        except ValueError:
            idx = vocabulary_ls.index('unk')
        indices_ls.append(idx)
    return indices_ls


def next_featuresandlabel_article(corpus_df):

    vocabulary_ls = Utils.get_vocabulary(corpus_df, F.vocabulary_fpath, min_frequency=2, new=False)

    article_labels = Utils.get_labels(corpus_df)
    articles_ls = list(corpus_df[Utils.Column.ARTICLE.value])

    for i, article in enumerate(articles_ls):
        # article is a string. E.g.: "21-Jähriger fällt wohl bis Saisonende aus. Wien – Rapid muss wohl bis ..."
        tokens_ls_0 = nltk.tokenize.word_tokenize(article, language='german')
        tokens_ls_lower = [tok.lower() for tok in tokens_ls_0]
        tokens_ls_nopunct = [tok for tok in tokens_ls_lower
                                     if tok not in '"#$%&\'()*+,-/:;<=>@[\\]^_`{|}~']  # keep the hyphen
        article_indices = get_article_indices(tokens_ls_nopunct, vocabulary_ls)

        yield (article_indices, article_labels[i])




