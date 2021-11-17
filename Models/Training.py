import logging
from math import inf
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
from datetime import datetime

def run_train():

    # initialize log file
    now = datetime.now()
    dt_string = now.strftime("%d_%m-%H_%M")
    Utils.init_logging("Training_" + dt_string + ".log")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Are training and validation set already defined? If not, split them from train.csv
    if not os.path.exists(Filepaths.training_set_file):
        train_df = Utils.load_split(Utils.Split.TRAIN)
        training_df, validation_df = CorpusReader.organize_training_corpus(train_df)
    else:
        training_df = pd.read_csv(Filepaths.training_set_file,
                                  sep=";", names=[Utils.Column.CLASS.value, Utils.Column.ARTICLE.value])
        validation_df = pd.read_csv(Filepaths.validation_set_file,
                                  sep=";", names=[Utils.Column.CLASS.value, Utils.Column.ARTICLE.value])

    # initialize model
    class_names = list(training_df["class"].value_counts().index)
    num_classes = len(class_names)
    word_embeddings = LV.get_word_vectors(False)
    model = CNN.ConvNet(word_embeddings, num_classes)
    model.to(DEVICE)
    model.train()

    # More initialization: object to hold the evaluation measures, optimizer
    measures_obj = EV.EvaluationMeasures()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # Training loop
    best_validation_loss = inf # for early-stopping
    max_epochs = 100
    current_epoch = 1
    num_training_samples = training_df.index.stop

    while current_epoch < max_epochs:
        training_iterator = CorpusReader.next_featuresandlabel_article(training_df) #  the samples' iterator
        sample_num = 0
        for article_indices, article_label in training_iterator:
            # starting operations on one batch
            optimizer.zero_grad()
            sample_num = sample_num +1

            x_indices_t = torch.tensor(article_indices).to(DEVICE)
            y_t = torch.tensor(article_label).to(DEVICE)
            label_probabilities = model(x_indices_t)
            y_predicted = torch.argmax(label_probabilities)

            # loss and step
            loss = tfunc.nll_loss(label_probabilities, y_t.unsqueeze(0))
            loss.backward()
            optimizer.step()

            # stats
            measures_obj.append_label(y_predicted.item())
            measures_obj.append_correct_label(article_label)
            measures_obj.append_loss(loss.item())

            if sample_num % (num_training_samples // 10) == 0:
                logging.info("Training sample: \t " + str(sample_num) + "/ " + str(num_training_samples) + " ...")

        # end of epoch: print stats, and reset them
        EV.log_accuracy_measures(measures_obj)
        measures_obj.reset_counters()
        current_epoch = current_epoch + 1

        # examine the validation set
        #validation_loss = evaluation(validation_df, model)
        #if validation_loss <= best_validation_loss:
        #    best_validation_loss = validation_loss
        #else:
        #    break  # early stop


# Inference only. Used for the validation set, and possibly any test set
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
        validation_measures_obj.append_label(predicted_label.item())
        validation_measures_obj.append_correct_label(article_label)
        validation_measures_obj.append_loss(loss.item())

    # end of epoch: print stats
    EV.log_accuracy_measures(validation_measures_obj)

    model.train()  # training can resume

    validation_loss = validation_measures_obj.compute_loss()

    return validation_loss

