# process_data.py

import os
import string

import numpy as np
from typing import List
from utils import *
from nltk.util import *
from nltk.tokenize import word_tokenize


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

    if '.DS_Store' in sorted_bill_list[0]:
        sorted_bill_list = sorted_bill_list[1:]
    for bill_num in range(len(bill_labels)):
        bill_file_path = os.path.join(bill_text_dir, sorted_bill_list[bill_num])
        bill_label = bill_labels[bill_num]

        with open(bill_file_path, 'r') as f:
            file_text = f.read().strip()
            tokenized_cleaned_sent = []
            # todo still not getting rid of all 's: need to fix this
            tokens = re.split('[- /\\n]', file_text)
            for word in tokens:
                word = word.strip(string.punctuation).lower()
                word_clean = re.sub('\'s', '', word)
                if word_clean != '' and not any(i.isdigit() for i in word_clean):
                    tokenized_cleaned_sent.append(word_clean)
            # tokenized_cleaned_sent = list(filter(lambda x: x != '',
            #                                      [word.lower() for word in file_text.strip(' ._:",~;|').split(" ")]))
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


class WordEmbeddings:
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """

    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        """
        Returns the embedding for a given word
        :param word: The word to look up
        :return: The UNK vector if the word is not in the Indexer or the vector otherwise
        """
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]


def read_word_embeddings(embeddings_file: str) -> WordEmbeddings:
    """
    Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
    that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
    word embedding files.
    :param embeddings_file: path to the file containing embeddings
    :return: WordEmbeddings object reflecting the words and their embeddings
    """
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    # Make position 0 a PAD token, which can be useful if you
    word_indexer.add_and_get_index("PAD")
    # Make position 1 the UNK token
    word_indexer.add_and_get_index("UNK")
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx + 1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)
            # Append the PAD and UNK vectors to start. Have to do this weirdly because we need to read the first line
            # of the file to see what the embedding dim is
            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)
    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return WordEmbeddings(word_indexer, np.array(vectors))


#################
# You probably don't need to interact with this code unles you want to relativize other sets of embeddings
# to this data. Relativization = restrict the embeddings to only have words we actually need in order to save memory.
# Very advantageous, though it requires knowing your dataset in advance, so it couldn't be used in a production system
# operating on streaming data.
def relativize(file, outfile, word_counter):
    """
    Relativize the word vectors to the given dataset represented by word counts
    :param file: word vectors file
    :param outfile: output file
    :param word_counter: Counter of words occurring in train/dev/test data
    :return:
    """
    f = open(file, encoding="utf8")
    o = open(outfile, 'w')
    voc = []
    count = 0
    for line in f:
        # print(count)
        count += 1
        word = line[:line.find(' ')]
        if word_counter[word] > 0:
            # print("Keeping word vector for " + word)
            voc.append(word)
            o.write(line)
    for word in word_counter:
        if word not in voc:
            count = word_counter[word]
            if count > 1:
                print("Missing " + word + " with count " + repr(count))
    f.close()
    o.close()


if __name__ == "__main__":
    # Count all words in the train, dev, and *test* sets. Note that this use of looking at the test set is legitimate
    # because we're not looking at the labels, just the words, and it's only used to cache computation that we
    # otherwise would have to do later anyway.
    word_counter = Counter()
    for ex in read_examples():
        for word in ex.words:
            word_counter[word] += 1
    # Uncomment these to relativize vectors to the dataset
    relativize("glove.6B.300d.txt", "glove.6B.300d-relativized.txt", word_counter)
