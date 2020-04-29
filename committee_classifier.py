# committee_classifier.py
# Modified from A1 code of CS 378 NLP course

import argparse
import time
from models import *
from typing import List
from sklearn.model_selection import KFold


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='PERCEPTRON', help='model to run (PERCEPTRON, LSA or BERT)')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing'
                                                                                                         ' output on '
                                                                                                         'the test set')
    args = parser.parse_args()
    return args


def train_model(args, all_exs: List[BillExample]) -> CommitteeClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    # if args.model == "PERCEPTRON":
    #     feat_extractor = PerceptronExtractor(Indexer())
    # else:
    #     raise Exception("Pass in correct string to run the appropriate system")

    # Train the model
    if args.model == "PERCEPTRON":
        model = train_perceptron(all_exs)
    else:
        raise Exception("Pass in PERCEPTRON, LSA or BERT to run the appropriate system")
    return model


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # Load train, dev, and test exs and index the words.
    all_exs = read_examples()
    # TODO need to split all exs into train/dev/test examples

    # print(repr(len(train_exs)) + " / " + repr(len(test_exs)) + " train/test examples")

    # Train and evaluate
    start_time = time.time()
    model = train_model(args, all_exs)
    print()
    print("=====Train Accuracy=====")
    evaluate(model, model.train_exs)
    print("=====Test Accuracy=====")
    evaluate(model, model.test_exs)
    print("Time for training and evaluation: %.2f seconds" % (time.time() - start_time))

    # TODO double check this
    # if args.run_on_test:
        # test_exs_predicted = [BillExample(words, model.predict(words)) for words in test_exs]
