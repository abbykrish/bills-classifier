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


def evaluate(classifier, exs):
    """
    Evaluates a given classifier on the given examples
    :param classifier: classifier to evaluate
    :param exs: the list of SentimentExamples to evaluate on
    :return: None (but prints output)
    """
    print_evaluation([ex.label for ex in exs], [classifier.predict(ex.words) for ex in exs])


def print_evaluation(golds: List[BillExample], predictions: List[BillExample]):
    """
    Prints evaluation statistics comparing golds and predictions, each of which is a sequence of 0/1 labels.
    Prints accuracy as well as precision/recall/F1 of the positive class, which can sometimes be informative if either
    the golds or predictions are highly biased.

    :param golds: gold SentimentExample objects
    :param predictions: pred SentimentExample objects
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))



def train_model(args, train_exs: List[BillExample]) -> CommitteeClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "PERCEPTRON":
        feat_extractor = PerceptronExtractor(Indexer())
    else:
        raise Exception("Pass in correct string to run the appropriate system")

    # Train the model
    if args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    else:
        raise Exception("Pass in PERCEPTRON, LSA or BERT to run the appropriate system")
    return model


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # Load train, dev, and test exs and index the words.
    all_exs = read_examples()
    # TODO need to split all exs into train/dev/test examples
    kfold = KFold(2, True, 1)
    kf_split = kfold.split(all_exs)
    set1, set2 = kf_split

    train_exs = []
    for i in set1[0]:
        train_exs.append(all_exs[i])

    test_exs = []
    for i in set1[1]:
        test_exs.append(all_exs[i])

    print(repr(len(train_exs)) + " / " + repr(len(test_exs)) + " train/test examples")

    # Train and evaluate
    start_time = time.time()
    model = train_model(args, train_exs)
    print("=====Train Accuracy=====")
    evaluate(model, train_exs)
    print("=====Test Accuracy=====")
    evaluate(model, test_exs)
    print("Time for training and evaluation: %.2f seconds" % (time.time() - start_time))

    # TODO double check this
    # if args.run_on_test:
        # test_exs_predicted = [BillExample(words, model.predict(words)) for words in test_exs]
