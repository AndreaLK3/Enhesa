# Load a model, having specified the learning_rate parameter.
# If such a model is not present, it throws error. It should be trained first.
# Use the model for inference, given a span of text.

import argparse
import Model.LoadVecs as LV
import Model.Training as T

def parse_inference_arguments():
    parser = argparse.ArgumentParser(description='Train a model, if necessary setting up vocabulary and word embeddings beforehand')

    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate for training the model; it is a parameter of the Adam optimizer')
    parser.add_argument('--text', type=str,
                        help="The text of the article that you wish to classify")

    args = parser.parse_args()
    return args


args = parse_inference_arguments()

word_embeddings = LV.get_word_vectors()

model = T.run_train(learning_rate=args.learning_rate)