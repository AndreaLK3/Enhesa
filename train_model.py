# Set up the necessary pipeline and train a model.
# If needed:
#   - Create the vocabulary from 10kGNAD/train.csv
#   - Compute (if random) or assign (if pretrained) the word vectors
# Then:
#   - train the model using Model/Resources/training.csv (90% of train.csv) and Model/Resources/validation.csv

import argparse
import Model.LoadVecs as LV
import Model.Training as T

def parse_training_arguments():
    parser = argparse.ArgumentParser(description='Train a model, if necessary setting up vocabulary and word embeddings beforehand')

    parser.add_argument('--use_pretrained_vectors', type=bool, default=False,
                        help='Whether to use the German Word2Vec vectors from https://www.deepset.ai/german-word-embeddings'
                             'to initialize the word embeddings, or just use the default random initialization ~N(0,0.01)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate for training the model; it is a parameter of the Adam optimizer')

    args = parser.parse_args()
    return args


args = parse_training_arguments()

word_embeddings = LV.get_word_vectors(args.use_pretrained_vectors)

model = T.run_train(pretrained_or_random_embeddings=args.use_pretrained_vectors, learning_rate=args.learning_rate)