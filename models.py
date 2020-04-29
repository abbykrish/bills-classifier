# models.py
# Some code modified from A1 code in CS 378 NLP course

from process_data import *
from utils import *
import numpy as np
from nltk.corpus import stopwords
from nltk.util import *
import re
from sklearn.model_selection import KFold

EN_STOPWORDS = stopwords.words('english')


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, ex_words: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param ex_words: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return:
        """
        raise Exception("Don't call me, call my subclasses")


class PerceptronExtractor(FeatureExtractor):
    """
    extracts features for the multi class perceptron
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, ex_words: List[str], add_to_indexer: bool) -> Counter:
        stop_words = set(EN_STOPWORDS)
        base_filtered = [w.lower() for w in ex_words if w.lower() not in stop_words and w.isalpha()]

        if add_to_indexer:
            for item in base_filtered:
                self.indexer.add_and_get_index(item)

        return Counter(base_filtered)


class CommitteeClassifier(object):
    """
     classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex_words: words (List[str]) in the sentence to classify
        :return: the committee the bill was classified as
        """
        raise Exception("Don't call me, call my subclasses")


class PerceptronClassifier(CommitteeClassifier):
    """
    subclass for multi-class prediction using perceptron
    """

    def __init__(self, feat_extractor: FeatureExtractor, weights, word_idf, train_exs, test_exs):
        self.weights = weights
        self.feat_extractor = feat_extractor
        self.word_idf = word_idf
        self.train_exs = train_exs
        self.test_exs = test_exs

    def predict(self, ex_words: List[str]) -> int:
        feat_cnt_dict = self.feat_extractor.extract_features(ex_words, False)
        feat_cnt_array = dict_to_np_array(feat_cnt_dict, self.feat_extractor.get_indexer(), self.word_idf, get_summary(ex_words))
        dot_prod = np.dot(self.weights, feat_cnt_array)
        y_pred = np.argmax(dot_prod)
        return y_pred


def dict_to_np_array(counter: Counter, indexer: Indexer, word_idf, summary):

    feat_cnt_array = np.zeros(len(indexer))
    # print(len(indexer))
    for word, count in counter.items():
        word_idx = indexer.index_of(word)
        if word_idx > 0:
            # multiplies by the word_idf for each one
            feat_cnt_array[word_idx] = word_idf[word_idx]

            # questionable way of doing this but its helping it and isn't bad for now
            if word in summary:
                feat_cnt_array[word_idx] *= 2
    return feat_cnt_array

def get_summary(bill_example):
    # if it doesn't find a summary it just is empty so it won't affect inputs that aren't this
    summaryRegex = "relating to (.*) BE IT ENACTED BY THE LEGISLATURE"
    summary = re.findall(summaryRegex, " ".join(bill_example))
    if len(summary) > 0:
        summary = summary[0]
    else:
        summary = ""
    return summary

# NOTE: implementing multiclass perceptron using the different weights approach
def train_perceptron(all_exs: List[BillExample]) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    kfold = KFold(5, True)
    maxAccuracy = -1
    bestModel = None
    k = 1
    for train_index, test_index in kfold.split(all_exs):
        train_exs = []
        for i in train_index:
            train_exs.append(all_exs[i])

        test_exs = []
        for i in test_index:
            test_exs.append(all_exs[i])

        # trying out a new model creating a new feature extractor
        feat_extractor = PerceptronExtractor(Indexer())
        indexer = feat_extractor.get_indexer()

        # we make a dict of dicts bc we don't know what the vocabulary of the entire dataset will be
        feat_labels = Counter()
        for ex in train_exs:
            feat_labels[ex] = feat_extractor.extract_features(ex.words, True)


        num_labels = 38  # TODO: get this number auto later

        weights = np.zeros((num_labels, len(indexer)))
        indices = list(range(len(train_exs)))

        # tf-idf calculation
        word_idf = np.zeros(len(indexer))
        for idx in indices:
            curr_example = train_exs[idx]
            feat_dict = feat_labels[curr_example]
            for item in feat_dict:
                word_idx = indexer.index_of(item)
                if word_idx > 0:
                    # shows up in this doc
                    word_idf[word_idx] += 1

        word_idf = np.log((len(train_exs) + 1) / (1 + word_idf).astype(float))

        num_epochs = 20
        for i in range(num_epochs):
            # go through the dataset randomly
            # lowering learning rate a bit each time
            learning_rate = 1 / (num_epochs + i)
            np.random.shuffle(indices)
            for idx in indices:
                curr_example = train_exs[idx]

                feat_dict = feat_labels[curr_example]
                curr_features = dict_to_np_array(feat_dict, indexer, word_idf, get_summary(curr_example.words))

                dot_prod = np.dot(weights, curr_features)
                y_pred = np.argmax(dot_prod)
                if y_pred != curr_example.label:
                    weights[y_pred] -= learning_rate * curr_features
                    weights[curr_example.label] += learning_rate * curr_features

        model = PerceptronClassifier(feat_extractor, weights, word_idf, train_exs, test_exs)
        print("=====Cross Validation Split %d=====" % k)
        k += 1
        accuracy = evaluate(model, test_exs)

        if accuracy > maxAccuracy:
            bestModel = model
            maxAccuracy = accuracy

    return bestModel


def evaluate(classifier, exs):
    """
    Evaluates a given classifier on the given examples
    :param classifier: classifier to evaluate
    :param exs: the list of SentimentExamples to evaluate on
    :return: None (but prints output)
    """
    return print_evaluation([ex.label for ex in exs], [classifier.predict(ex.words) for ex in exs])



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
    return (float(num_correct) / num_total)


