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

    def __init__(self, feat_extractor: FeatureExtractor, weights, word_idf):
        self.weights = weights
        self.feat_extractor = feat_extractor
        self.word_idf = word_idf

    def predict(self, ex_words: List[str]) -> int:
        feat_cnt_dict = self.feat_extractor.extract_features(ex_words, False)
        feat_cnt_array = dict_to_np_array(feat_cnt_dict, self.feat_extractor.get_indexer(), self.word_idf)
        dot_prod = np.dot(self.weights, feat_cnt_array)
        y_pred = np.argmax(dot_prod)
        return y_pred


def dict_to_np_array(counter: Counter, indexer: Indexer, word_idf):

    feat_cnt_array = np.zeros(len(indexer))
    # print(len(indexer))
    for word, count in counter.items():
        word_idx = indexer.index_of(word)
        if word_idx > 0:
            # multiplies by the word_idf for each one
            feat_cnt_array[word_idx] = count * word_idf[word_idx]
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

    # print(word_idf[indexer.index_of("section")])
    # print(word_idf[indexer.index_of("prison")])
    # print(word_idf[indexer.index_of("year")])

    num_epochs = 20
    num = 0
    for i in range(num_epochs):
        # go through the dataset randomly
        learning_rate = 1 / num_epochs
        np.random.shuffle(indices)
        for idx in indices:
            num += 1
            curr_example = train_exs[idx]
            feat_dict = feat_labels[curr_example]
            curr_features = dict_to_np_array(feat_dict, indexer, word_idf)
            # print(curr_features)
            dot_prod = np.dot(weights, curr_features)
            y_pred = np.argmax(dot_prod)
            if y_pred != curr_example.label:
                weights[y_pred] -= learning_rate * curr_features
                weights[curr_example.label] += learning_rate * curr_features

    return PerceptronClassifier(feat_extractor, weights, word_idf)
