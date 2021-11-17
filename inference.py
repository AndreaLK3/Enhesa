# Load a model, having specified the learning_rate parameter.
# If such a model is not present, it throws error. It should be trained first.
# Use the model for inference, given a span of text.

import argparse
import Model.LoadVecs as LV
import Model.Training as T
import Utils

def parse_inference_arguments():
    parser = argparse.ArgumentParser(description='Train a model, if necessary setting up vocabulary and word embeddings beforehand')

    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate for training the model; it is a parameter of the Adam optimizer')
    parser.add_argument('--text', type=str,
                        help="The text of the article that you wish to classify")
    parser.add_argument('--use_test_set', type=bool, default=False,
                        help='Instead of performing inference on a sample, run an evaluation of the model on the test set')

    args = parser.parse_args()
    return args


args = parse_inference_arguments()

if args.use_test_set:  # evaluate the specified model on the test set
    test_df = Utils.load_split(Utils.Split.TEST)
    # Utils.init_logging("Test_lr0.0005.log")
    # T.evaluation(test_df, model)
else:  # inference on the given sample
    pass