import torch
import Models.CorpusReader as CorpusReader
import Utils
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

    loss = measures_obj.compute_loss()

    logging.info("Loss=" + str(round(loss,2))+ " ; accuracy=" + str(accuracy))
    logging.info("precision=" + str(precision) + " ; recall=" + str(recall))
    logging.info("F1_score=" + str(f1_score))



def run_train():

    corpus_df = Utils.load_split(Utils.Split.TRAIN)
    class_names = list(corpus_df["class"].value_counts().index)
    num_classes = len(class_names)
    word_embeddings = LV.get_word_vectors(False)
    model = CNN.ConvNet(word_embeddings, num_classes)
    model.train()

    training_iterator = CorpusReader.next_featuresandlabel_article(corpus_df)
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



def evaluation(corpus_df, num_classes, model):
    model.eval()  # do not train the model now
    samples_iterator = CorpusReader.next_featuresandlabel_article(corpus_df)

    for article_indices, article_label in samples_iterator:

        x_indices_t = torch.tensor(article_indices)
        y_t = torch.tensor(article_label)
        label_probabilities = model(x_indices_t, y_t)
        predicted_label = torch.argmax(label_probabilities)

        loss = tfunc.nll_loss(label_probabilities, y_t.unsqueeze(0))

        model.train()  # training can resume