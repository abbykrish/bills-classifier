from process_data import *
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# Methods for evaluating models
def evaluate(classifier, exs, show_plot):
    """
    Evaluates a given classifier on the given examples
    :param classifier: classifier to evaluate
    :param exs: the list of BillExample to evaluate on
    :return: None (but prints output)
    """
    labels = [ex.label for ex in exs]
    predictions = [classifier.predict(ex.words) for ex in exs]
    return print_evaluation(labels, predictions, show_plot)


def print_evaluation(golds: List[BillExample], predictions: List[BillExample], show_plot):
    """
    Prints evaluation statistics comparing golds and predictions
    Prints accuracy as well as precision/recall/F1 of the positive class, which can sometimes be informative if either
    the golds or predictions are highly biased.

    :param golds: gold BillExample objects
    :param predictions: pred BillExample objects
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
    print_stats(golds, predictions, show_plot)
    return (float(num_correct) / num_total)

def print_stats(golds, predictions, show_plot):
    cm = confusion_matrix(golds, predictions, labels=list(range(30)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.nan_to_num(np.diag(cm) / np.sum(cm, axis = 1))
        recall = np.mean(recall)
        precision = np.nan_to_num(np.diag(cm) / np.sum(cm, axis = 0))
        precision = np.mean(precision)
        f1 = 2 * precision * recall / (precision + recall)

        print("Precision: %f, Recall: %f, F1: %f" % (precision, recall, f1))
        if show_plot == True:
            plot(cm)


def plot(cm):
    df_cm = pd.DataFrame(cm, index = range(cm.shape[0]),
                  columns = range(cm.shape[1]))
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=False, cmap="GnBu")
    plt.show()
