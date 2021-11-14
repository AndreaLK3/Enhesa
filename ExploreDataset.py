# Use matplotlib, seaborn or other Python libraries for visualization to investigate the dataset
# train.csv and test.csv have a stratified split of 10% for testing and the remaining articles for training
# It is opportune to split another 10% fromm train.csv to get a validation set (or to use N-fold cross-validation)

# - Number of articles per class; bar plot, or pie graph. How much is the dataset imbalanced?
# - Average number of words in an article, per class; bar plot.
#   This visualization and the previous one are already found on the site
# - Numerical table or color-matrix, expressing the overlap in vocabulary between any 2 classes

import Utils
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np

def num_articles():
    training_df = Utils.load_split(Utils.Split.TRAINING)
    num_articles_per_class_trainingdf = training_df.groupby(Utils.Column.CLASS.value).count()
    # print(str(num_articles_per_class_trainingdf))
    test_df = Utils.load_split(Utils.Split.TEST)

    class_frequencies_training = list(training_df["class"].value_counts())
    class_frequencies_test = list(test_df["class"].value_counts())
    class_frequencies_total = [class_frequencies_training[i] + class_frequencies_test[i]
                               for i in range(len(class_frequencies_training))]
    class_names = list(training_df["class"].value_counts().index)
    # print(str(class_frequencies_training))

    fig, ax = plt.subplots()
    xlocs, xlabs = plt.xticks()
    bar_obj = plt.bar(x=class_names, height=class_frequencies_training, width=0.6, color="b")
    # plt.bar(x=class_names, height=class_frequencies_test, bottom=class_frequencies_training, width=0.6, color="g")
    plt.xticks(rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Number of articles')
    plt.title("Articles per class - training set")
    plt.bar_label(bar_obj, labels=class_frequencies_training)
    # plt.legend(["Training set", "Test set"])
    plt.show()

def words_in_articles():
    # For each class, we must reunite the text of the articles, and count the number of tokens.
    # This is viable because the dataset is very small, otherwise we would have to split it and use partial results
    training_df = Utils.load_split(Utils.Split.TRAINING)
    # test_df = Utils.load_split(Utils.Split.TEST)

    training_articles = training_df[Utils.Column.ARTICLE.value].to_list()
    # test_articles = test_df[Utils.Column.ARTICLE.value].to_list()

    class_names = list(training_df["class"].value_counts().index)
    avg_words_per_class = []
    vocabularies_ls = []
    for c_name in class_names:
        class_articles = training_df[training_df[Utils.Column.CLASS.value] == c_name][Utils.Column.ARTICLE.value].to_list()
        num_articles_per_class = len(class_articles)
        total_words_per_class = len(nltk.tokenize.word_tokenize(" ".join(class_articles), language='german'))
        avg_words_per_class.append((int(total_words_per_class / num_articles_per_class)))

    fig, ax = plt.subplots()
    xlocs, xlabs = plt.xticks()
    bar_obj = plt.bar(x=class_names, height=avg_words_per_class, width=0.6, color="y")
    plt.xticks(rotation=45)
    plt.xlabel('Classes')
    plt.ylabel('Words')
    plt.title("Average words per article - training set")
    plt.bar_label(bar_obj, labels=avg_words_per_class)
    # plt.legend(["Training set", "Test set"])
    plt.show()

    return vocabularies_ls

def vocabulary_of_classes():
    # For each class, we must reunite the text of the articles, and count the number of tokens.
    # This is viable because the dataset is very small, otherwise we would have to split it and use partial results
    # training_df = Utils.load_split(Utils.Split.TRAINING)
    test_df = Utils.load_split(Utils.Split.TEST)

    # training_articles = training_df[Utils.Column.ARTICLE.value].to_list()
    test_articles = test_df[Utils.Column.ARTICLE.value].to_list()

    class_names = list(test_df["class"].value_counts().index)
    num_classes = len(class_names)
    classes_words_ls = []
    for c_name in class_names:
        class_articles = test_df[test_df[Utils.Column.CLASS.value] == c_name][
            Utils.Column.ARTICLE.value].to_list()
        vectorizer = CountVectorizer(lowercase=False)
        vectorizer.fit(class_articles)
        words_in_class = set(vectorizer.vocabulary_.keys())
        classes_words_ls.append(words_in_class)

    overlap_matrix = np.zeros((num_classes,num_classes))

    for i in range(num_classes):
        words_class_i = classes_words_ls[i]
        for j in range(num_classes):
            words_class_j = classes_words_ls[j]
            cardinality_overlap_i_j = len(words_class_i.intersection(words_class_j))
            cardinality_smaller_class = min(len(words_class_i), len(words_class_j))
            overlap_i_j = cardinality_overlap_i_j / cardinality_smaller_class # len(words_class_i)
            if (i!=j):
                overlap_matrix[i, j] = overlap_i_j
            print("|" + str(class_names[i]) + "|= " + str(len(words_class_i)) +
                  "; |" + str(class_names[j]) + "|= " + str(len(words_class_j))
            + "; |overlap|= " + str(cardinality_overlap_i_j) + "; overlap=" + str(round(overlap_i_j,2)))



    fig, ax = plt.subplots()
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.imshow(overlap_matrix, cmap='viridis')
    plt.colorbar()
    plt.title("Overlap in vocabulary between classes")

    return overlap_matrix
