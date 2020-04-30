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
    :param args: args bundle
    :param train_exs: training set, List of CommitteeClassifier objects
    :return: trained CommitteeClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    # if args.model == "PERCEPTRON":
    #     feat_extractor = PerceptronExtractor(Indexer())
    # else:
    #     raise Exception("Pass in correct string to run the appropriate system")

    # Train the model
    if args.model == "PERCEPTRON":
        model = train_perceptron(all_exs)
    elif args.model == "CNN":
        word_embeddings = read_word_embeddings("glove.6B.300d-relativized.txt")
        print(word_embeddings)
        model = train_cnn_classifier(args, all_exs, word_embeddings)
    else:
        raise Exception("Pass in PERCEPTRON, CNN or BERT to run the appropriate system")
    return model


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # Load train, dev, and test exs and index the words.
    all_exs = read_examples()
    # TODO need to split all exs into train/dev/test examples
    # kfold = KFold(5, True, 1)
    # kf_split = kfold.split(all_exs)
    # set1, set2, set3, set4, set5 = kf_split

    # train_exs = []
    # for i in set1[0]:
    #     train_exs.append(all_exs[i])

    # test_exs = []
    # for i in set1[1]:
    #     test_exs.append(all_exs[i])

    train_exs = all_exs[:int(len(all_exs) * (4 / 5))]
    test_exs = all_exs[int(len(all_exs) * (4 / 5)):]

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
