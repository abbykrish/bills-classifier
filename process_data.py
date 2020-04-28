# process_data.py

import os
from typing import List
from utils import *


class BillExample:
    """
    Data wrapper for a single example for multi class classification

    Attributes:
        words (List[string]): list of words
        label (int): the gold label of the bill
    """

    def __init__(self, words, label):
        self.words = words
        self.label = label

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()


def read_examples() -> List[BillExample]:
    """
    Reads examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences and forms
    BillExamples.
    :return: a list of BillExamples parsed from the bill texts
    """
    bill_text_dir = os.path.join(os.getcwd(), "bill-texts")
    sorted_bill_list = sorted(os.listdir(bill_text_dir))
    bill_labels = read_labels()
    exs = []

    for bill_num in range(2, len(bill_labels)):
        bill_file_path = os.path.join(bill_text_dir, sorted_bill_list[bill_num])
        bill_label = bill_labels[bill_num]
        with open(bill_file_path, 'r') as f:
            file_text = f.read().strip().replace('\n', '')
            # TODO any additional clean up to the bill text needed
            tokenized_cleaned_sent = list(filter(lambda x: x != '', file_text.rstrip().split(" ")))
            exs.append(BillExample(tokenized_cleaned_sent, bill_label))
    return exs


def read_labels() -> List[int]:
    all_labels = Indexer()
    all_labels_file_name = os.path.join(os.getcwd(), "all_labels.txt")
    all_labels_file = open(all_labels_file_name)
    for line in all_labels_file:
        all_labels.add_and_get_index(line.strip())
    all_labels_file.close()

    labels = []
    labels_file_name = os.path.join(os.getcwd(), "labels.txt")
    labels_file = open(labels_file_name)
    for line in labels_file:
        label_num = all_labels.index_of(line.strip())
        labels.append(label_num)
    labels_file.close()
    return labels
