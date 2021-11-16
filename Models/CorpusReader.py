
import Utils

def exe(corpus_df, word_embeddings):

    article_labels = Utils.get_labels(Utils.Split.TRAINING)
    articles_ls = list(corpus_df[Utils.Column.ARTICLE])

    for article in articles_ls:
        print("*")


