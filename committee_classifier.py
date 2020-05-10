# committee_classifier.py
# Modified from A1 code of CS 378 NLP course

import argparse
import time
from models import *
from typing import List


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='PERCEPTRON', help='model to run (PERCEPTRON, LSA or BERT)')
    args = parser.parse_args()
    return args


def train_model(args, all_exs: List[BillExample]) -> CommitteeClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle
    :param all_exs: training and testing set combined, List of CommitteeClassifier objects
    :return: trained CommitteeClassifier model, of whichever type is specified
    """
    # Train the model
    if args.model == "PERCEPTRON":
        model = train_perceptron(all_exs)
    elif args.model == "CNN":
        word_embeddings = read_word_embeddings("glove.6B.300d-relativized.txt")
        model = train_cnn_classifier(args, all_exs, word_embeddings)
    else:
        raise Exception("Pass in PERCEPTRON, CNN or BERT to run the appropriate system")
    return model


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # Load all exs and index the words.
    all_exs = read_examples()

    # Train and evaluate
    start_time = time.time()
    model = train_model(args, all_exs)
    print()
    print("=====Train Accuracy=====")
    evaluate(model, model.train_exs, False)
    print("=====Test Accuracy=====")
    evaluate(model, model.test_exs, True)
    print("Time for training and evaluation: %.2f seconds" % (time.time() - start_time))
