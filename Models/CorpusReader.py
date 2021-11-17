import Filepaths as F
import Utils
import nltk
import string
import numpy as np
import pandas as pd

def get_article_indices(article_tokens, vocabulary_ls):
    indices_ls = []
    for tok in article_tokens:
        try:
            idx = vocabulary_ls.index(tok)
        except ValueError:
            idx = vocabulary_ls.index('unk')
        indices_ls.append(idx)
    return indices_ls

# Iterator that gives an article's vocabulary indices and class label
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


# From the original training corpus in train.csv,
# split off 10% of each class to be the validation set,
# and keep the rest in Models/training.csv to be used as the actual training set
def organize_training_corpus(train_corpus_df):
    class_frequencies_train = list(train_corpus_df["class"].value_counts())

    # determine how many articles from train.csv will need to be moved in the validation set
    needed_frequencies_validation = [int(freq/10) for freq in class_frequencies_train]

    class_names = list(train_corpus_df["class"].value_counts().index)
    num_classes = len(class_names)
    class_indices_buckets = dict.fromkeys(list(range(num_classes)), set())

    # divide the indices of the rows, by class
    for index, row in train_corpus_df.iterrows():
        article_class_idx = class_names.index(row[Utils.Column.CLASS.value])
        class_indices_buckets[article_class_idx].add(index)

    indices_validation = []
    indices_training = []
    # for each class,
    for c in class_indices_buckets.keys():
        c_indices = list(class_indices_buckets[c])  # (we have no guarantees on the order, and this is intended)
        # add a random 10% of the articles of the class, from the original train.csv, to the validation set
        c_indices_validation = c_indices[0:needed_frequencies_validation[c]]
        indices_validation = indices_validation + c_indices_validation
        # the rest goes into the training set proper
        c_indices_training = c_indices[0:needed_frequencies_validation[c]]
        indices_training = indices_training + c_indices_training

    new_training_rows_ls = []
    validation_rows_ls = []

    for index, row in train_corpus_df.iterrows():
        if index in indices_validation:
            validation_rows_ls.append(row)
        else:
            new_training_rows_ls.append(row)

    new_training_df = pd.DataFrame(new_training_rows_ls)
    validation_df = pd.DataFrame(validation_rows_ls)

    new_training_df.to_csv(F.training_set_file)
    validation_df.to_csv(F.validation_set_file)

    return (new_training_df, validation_df)




