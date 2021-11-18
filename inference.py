# Load a model, having specified the learning_rate parameter.
# If such a model is not present, it throws error. It should be trained first.
# Use the model for inference, given a span of text.
import Filepaths
import Filepaths as F
import argparse
import os
import Utils
import torch
import Model.Training as T
import Model.CorpusReader as CR
import nltk
import Model.CorpusOrganizer as CO
import pickle

def parse_inference_arguments():
    parser = argparse.ArgumentParser(description='Load a model and either run inference on a provided sample '
                                                 'or evaluate it on the test set')

    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate used in the model we wish to load')
    parser.add_argument('--text', type=str,
                        help="The text of the article that you wish to classify")
    parser.add_argument('--use_test_set', type=bool, default=False,
                        help='Instead of performing inference on a sample, run an evaluation of the model on the test set')

    args = parser.parse_args()
    return args

def load_model(lr):
    model_fname = "Model_" + "lr" + str(lr) + ".pt"
    saved_model_fpath = os.path.join(F.models_folder, F.saved_models_subfolder, model_fname)
    model = torch.load(saved_model_fpath)
    return model

args = parse_inference_arguments()
test_df = Utils.load_split(Utils.Split.TEST)  # TEMP
if args.use_test_set:  # evaluate the specified model on the test set
    Utils.init_logging("Test_lr" + str(args.learning_rate) + ".log")
    test_df = Utils.load_split(Utils.Split.TEST)
    model = load_model(args.learning_rate)
    T.evaluation(test_df, model)
else:  # inference on the given sample
    class_names = list(test_df["class"].value_counts().index)
    # class_names.sort()
    print(class_names)

    model = load_model(args.learning_rate)
    text = args.text

    # tokenize the text
    tokens_ls_0 = nltk.tokenize.word_tokenize(text, language='german')
    tokens_ls_lower = [tok.lower() for tok in tokens_ls_0]
    tokens_ls_nopunct = [tok for tok in tokens_ls_lower
                         if tok not in '"#$%&\'()*+,-/:;<=>@[\\]^_`{|}~']  # keep the hyphen

    # get the vocabulary
    vocab_fpath = Filepaths.vocabulary_fpath
    with open(vocab_fpath, "rb") as vocab_file:
            vocabulary_ls = pickle.load(vocab_file)

    # get the indices for the text
    article_indices = CR.get_article_indices(tokens_ls_nopunct, vocabulary_ls)
    x_indices_t = torch.tensor(article_indices)

    # predict classification
    label_probabilities = model(x_indices_t)
    y_predicted = torch.argmax(label_probabilities)

    print("Predicted class = " + str(class_names[y_predicted]))
