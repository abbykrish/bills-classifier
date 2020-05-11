from collections import Counter
from utils import *
import numpy as np
import re


# Methods for document cleaning and preprocessing for perceptron
def dict_to_np_array(counter: Counter, indexer: Indexer, word_idf, summary, enhanced_flag):
    feat_cnt_array = np.zeros(len(indexer))
    for word, count in counter.items():
        word_idx = indexer.index_of(word)
        if word_idx > 0:
            # starting feature is word_idf
            if enhanced_flag:
                feat_cnt_array[word_idx] = word_idf[word_idx]
            else:
                feat_cnt_array[word_idx] = count

    if enhanced_flag:
        for word in summary.split(" "):
            word_idx = indexer.index_of(word)
            feat_cnt_array[word_idx] *= 2

    return feat_cnt_array


def get_summary(bill_example):
    """
    Get summary of bill from the document to use as a feature
    :param bill_example: the text of the bill
    :return: the summary string

    """
    summaryRegex = "relating to (.*) be it enacted by the legislature"
    summary = re.findall(summaryRegex, " ".join(bill_example))
    if len(summary) > 0:
        summary = summary[0]
    else:
        summary = ""

    return summary


def get_top_words_per_class(weights, feat_extractor):
    """
    Gets the words with the highest weights
    :param weights:
    :param feat_extractor:
    :return
    """
    w = 0
    for weight in weights:
        print("==top 5 words for %d==" % w)
        w += 1
        for i in sorted(range(len(weight)), key=lambda i: weight[i], reverse=True)[:5]:
            print(feat_extractor.get_indexer().get_object(i))


def tf_idf_calc(indexer, train_exs, feat_labels):
    """
    Calculates the TF-IDF weights for each feature
    :param indexer: the features of the perceptron
    :param train_exs: the examples for training the perceptron
    :param feat_labels: the feature vector for each example
    :returns the TF-IDF weights for each feature
    """
    word_idf = np.zeros(len(indexer))
    for idx in range(len(train_exs)):
        curr_example = train_exs[idx]
        feat_dict = feat_labels[curr_example]
        for item in feat_dict:
            word_idx = indexer.index_of(item)
            if word_idx > 0:
                # shows up in this doc
                word_idf[word_idx] += 1

    word_idf = np.log((len(train_exs) + 1) / (1 + word_idf).astype(float))
    return word_idf
