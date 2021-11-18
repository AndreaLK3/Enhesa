# Set up the necessary pipeline and train a model.
# If needed:
#   - Create the vocabulary from 10kGNAD/train.csv
#   - Compute the word embeddings, ~N(0,0.01)
# Then:
#   - train the model using Model/Resources/training.csv (90% of train.csv) and Model/Resources/validation.csv

import argparse
import Model.LoadVecs as LV
import Model.Training as T
import Utils


def parse_training_arguments():
    parser = argparse.ArgumentParser(description='Train a model, if necessary setting up vocabulary and word embeddings beforehand')

    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate for training the model; it is a parameter of the Adam optimizer')

    args = parser.parse_args()
    return args


args = parse_training_arguments()

word_embeddings = LV.get_word_vectors()

# model = T.run_train(learning_rate=args.learning_rate)

# TEMPORARY
test_df = Utils.load_split(Utils.Split.TEST)
model = T.run_train(learning_rate=5e-4)
Utils.init_logging("Test_lr0.0005.log")
T.evaluation(test_df, model)

model = T.run_train(learning_rate=1e-4)
Utils.init_logging("Test_lr0.0001.log")
T.evaluation(test_df, model)