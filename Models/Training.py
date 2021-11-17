import torch
import pandas as pd
import Filepaths
import Models.CorpusReader as CorpusReader
import Utils
import os
import Models.LoadVecs as LV
import torch.nn.functional as tfunc
import Models.CNN as CNN
import Models.EvaluationMeasures as EV
import logging

def log_accuracy_measures(measures_obj):

    accuracy = measures_obj.compute_accuracy()
    precision = measures_obj.compute_precision()
    recall = measures_obj.compute_recall()
    f1_score = measures_obj.compute_f1score()
    confusion_matrix = measures_obj.compute_confusion_matrix()

    loss = measures_obj.compute_loss()

    logging.info("Loss=" + str(round(loss,2))+ " ; accuracy=" + str(accuracy))
    logging.info("precision=" + str(precision) + " ; recall=" + str(recall))
    logging.info("F1_score=" + str(f1_score))
    logging.info("confusion_matrix=" + str(confusion_matrix))


def run_train():

    if not os.path.exists(Filepaths.training_set_file):
        train_df = Utils.load_split(Utils.Split.TRAIN)
        training_df, validation_df = CorpusReader.organize_training_corpus(train_df)
    else:
        training_df = pd.read_csv(Filepaths.training_set_file,
                                  sep=";", names=[Utils.Column.CLASS.value, Utils.Column.ARTICLE.value])
        validation_df = pd.read_csv(Filepaths.validation_set_file,
                                  sep=";", names=[Utils.Column.CLASS.value, Utils.Column.ARTICLE.value])

    class_names = list(training_df["class"].value_counts().index)
    num_classes = len(class_names)
    word_embeddings = LV.get_word_vectors(False)
    model = CNN.ConvNet(word_embeddings, num_classes)
    model.train()

    training_iterator = CorpusReader.next_featuresandlabel_article(training_df)
    measures_obj = EV.EvaluationMeasures()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for article_indices, article_label in training_iterator:
        # starting operations on one batch
        optimizer.zero_grad()

        x_indices_t = torch.tensor(article_indices)
        y_t = torch.tensor(article_label)
        label_probabilities = model(x_indices_t, y_t)
        predicted_label = torch.argmax(label_probabilities)

        # loss and step
        loss = tfunc.nll_loss(label_probabilities, y_t.unsqueeze(0))
        loss.backward()
        optimizer.step()

        # stats
        measures_obj.append_label(predicted_label)
        measures_obj.append_correct_label(article_label)
        measures_obj.append_loss(loss)

    # end of epoch: print stats, and reset them
    log_accuracy_measures(measures_obj)
    measures_obj.reset_counters()

    # examine the validation set
    evaluation(validation_df, model)


def evaluation(corpus_df, model):
    model.eval()  # do not train the model now
    samples_iterator = CorpusReader.next_featuresandlabel_article(corpus_df)

    validation_measures_obj = EV.EvaluationMeasures()

    for article_indices, article_label in samples_iterator:

        x_indices_t = torch.tensor(article_indices)
        y_t = torch.tensor(article_label)
        label_probabilities = model(x_indices_t, y_t)
        predicted_label = torch.argmax(label_probabilities)

        loss = tfunc.nll_loss(label_probabilities, y_t.unsqueeze(0))

        # stats
        validation_measures_obj.append_label(predicted_label)
        validation_measures_obj.append_correct_label(article_label)
        validation_measures_obj.append_loss(loss)

    # end of epoch: print stats
    log_accuracy_measures(validation_measures_obj)

    model.train()  # training can resume

    validation_loss = validation_measures_obj.compute_loss()

    return validation_loss

