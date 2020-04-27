# models.py
# Some code modified from A1 code in CS 378 NLP course

from process_data import *
from utils import *
import numpy as np
from nltk.corpus import stopwords
from nltk.util import *

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
        raise Exception("needs to be implemented")


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

    def __init__(self, feat_extractor: FeatureExtractor, weights):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        feat_cnt_dict = self.feat_extractor.extract_features(ex_words, False)
        feat_cnt_array = dict_to_np_array(feat_cnt_dict, self.feat_extractor.get_indexer())
        dot_prod = np.dot(self.weights, feat_cnt_array)
        y_pred = np.argmax(dot_prod)
        return y_pred


def dict_to_np_array(counter: Counter, indexer: Indexer):
    feat_cnt_array = np.zeros(indexer.__len__())
    for word, count in counter.items():
        word_idx = indexer.index_of(word)
        if word_idx > 0:
            feat_cnt_array[word_idx] = count
    return feat_cnt_array


# NOTE: implementing multiclass perceptron using the different weights approach
def train_perceptron(train_exs: List[BillExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    indexer = feat_extractor.get_indexer()

    # what exactly was a counter?
    # we make a dict of dicts bc we don't know what the vocabulary of the entire dataset will be
    feat_labels = Counter()
    for ex in train_exs:
        feat_labels[ex] = feat_extractor.extract_features(ex.words, True)

    num_labels = 38  # TODO: get this number auto later
    weights = np.zeros((num_labels, len(indexer)))
    indices = list(range(len(train_exs)))
    num_epochs = 20
    for i in range(num_epochs):
        # go through the dataset randomly
        learning_rate = 1 / num_epochs
        np.random.shuffle(indices)
        for idx in indices:
            curr_example = train_exs[idx]
            feat_dict = feat_labels[curr_example]
            curr_features = dict_to_np_array(feat_dict, indexer)
            dot_prod = np.dot(weights, curr_features)
            y_pred = np.argmax(dot_prod)
            if y_pred != curr_example.label:
                weights[y_pred] -= learning_rate * curr_features
                weights[curr_example.label] += learning_rate * curr_features

    return PerceptronClassifier(feat_extractor, weights)
