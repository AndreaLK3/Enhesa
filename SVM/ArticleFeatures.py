# Alternative n.1 for the document (i.e. article) features used in the Support Vector Machine and CNN:
# No pre-trained vectors are used. The word vectors are initialized randomly.
# Then, the article vector is computed as the average of the random vectors, weighted by TF-IDF
import Utils
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def init_corpus_features():
    corpus_df = Utils.load_split(Utils.Split.TEST)
    corpus_txt_ls = corpus_df[Utils.Column.ARTICLE.value].to_list()
    # tfidf_vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', stop_words=None,
                                       max_features=None, vocabulary=None)
    # tfidf_vectorizer.fit(corpus_txt_ls)

    for article in corpus_txt_ls:
        pass




