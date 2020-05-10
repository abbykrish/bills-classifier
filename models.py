# models.py
# Some code modified from A1 code in CS 378 NLP course

from evaluate_utils import *
from doc_process_utils import *

from nltk.corpus import stopwords
from nltk.util import *

from sklearn.model_selection import KFold
import lexnlp.extract.en.regulations as lexnlp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time

EN_STOPWORDS = stopwords.words('english')


class CNN(nn.Module):
    def __init__(self, num_filters, window_sizes, dropout, word_embeddings: WordEmbeddings, num_classes):
        """
        :param num_filters: the number of filters/kernels on each example
        :param window_sizes: the different window sizes for each filter
        :param dropout: the dropout rate
        :param word_embeddings: the pretrained glove vectors
        """
        super(CNN, self).__init__()
        # freeze = False to have non static word embeddings
        self.word_embedder = word_embeddings
        self.word_embedding_obj = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings.vectors),
                                                               freeze=True, padding_idx=0)
        embedding_dim = word_embeddings.get_embedding_length()
        self.drop_out = nn.Dropout(dropout)
        # creating different convolutional layers from window sizes to look at different "n"-grams in network
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=[window_size, embedding_dim],
                      padding=(window_size - 2, 0))
            for window_size in window_sizes
        ])
        self.fc = nn.Linear(num_filters * len(window_sizes), num_classes)

    def forward(self, input):
        embedded_input = self.word_embedding_obj(input).float()  # [batch size, sent len, emb dim]
        x = embedded_input.unsqueeze(1)  # to add a channel dimension [batch size, 1, sent len, emb dim]

        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x.float()).float()).squeeze(3)
            x2 = F.max_pool1d(x2, x2.shape[2]).squeeze(2)
            xs.append(x2)

        x = self.drop_out(torch.cat(xs, 1))
        logits = self.fc(x)
        return logits


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
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are
         encountered. At test time, any unseen features should be discarded, but at train time, we probably want to
         keep growing it.
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

    def extract_features(self, ex_words: List[str], add_to_indexer: bool = False) -> Counter:
        stop_words = set(EN_STOPWORDS)
        regulations = list(lexnlp.get_regulations(" ".join(ex_words)))

        base_filtered = [w for w in ex_words if w not in stop_words and not any(i.isdigit() for i in w)]
        filtered = []

        for item in base_filtered:
            filtered.append(item)
            if add_to_indexer:
                self.indexer.add_and_get_index(item)

        for item in regulations:
            reg = item[1]
            filtered.append(reg)
            if add_to_indexer:
                self.indexer.add_and_get_index(reg)

        return Counter(filtered)


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
        feat_cnt_array = dict_to_np_array(feat_cnt_dict, self.feat_extractor.get_indexer(), self.word_idf,
                                          get_summary(ex_words))
        dot_prod = np.dot(self.weights, feat_cnt_array)
        y_pred = np.argmax(dot_prod)
        return y_pred


class CNNClassifier(CommitteeClassifier):
    """
       subclass for multi-class prediction using CNNs
    """

    def __init__(self, cnn, word_embedder, train_exs, test_exs):
        self.cnn = cnn
        self.word_embedder = word_embedder
        self.train_exs = train_exs
        self.test_exs = test_exs

    def predict(self, ex_words: List[str]) -> int:
        self.cnn.eval()
        # convert words to indices
        indices = [self.word_embedder.word_indexer.index_of(ex_word) for ex_word in ex_words]
        indices = [1 if i == -1 else i for i in indices]
        # add a batch dim
        batched_ex = torch.LongTensor(indices).unsqueeze(0)
        # batched_ex shape is [batch size, sent len]
        predictions = self.cnn.forward(batched_ex)
        predictions = predictions.squeeze(0)

        probs = F.softmax(predictions, dim=0)
        return torch.argmax(probs).item()


# NOTE: implementing multiclass perceptron using the different weights approach
def train_perceptron(all_exs: List[BillExample]) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param all_exs: training and testing sets, List of SentimentExample objects
    :return: trained PerceptronClassifier model
    """

    # we iterate through 5 different folds to cross validate the training/test sets
    kfold = KFold(5, True)
    max_accuracy = -1
    best_model = None
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

        # basic initialization
        num_labels = 30  # TODO: get this number auto later
        weights = np.zeros((num_labels, len(indexer)))
        indices = list(range(len(train_exs)))

        # used for initial weights
        word_idf = tf_idf_calc(indexer, train_exs, feat_labels)

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

                # multiclass perceptron update
                dot_prod = np.dot(weights, curr_features)
                y_pred = np.argmax(dot_prod)
                if y_pred != curr_example.label:
                    weights[y_pred] -= learning_rate * curr_features
                    weights[curr_example.label] += learning_rate * curr_features

        model = PerceptronClassifier(feat_extractor, weights, word_idf, train_exs, test_exs)
        print("=====Cross Validation Split %d=====" % k)
        k += 1
        accuracy = evaluate(model, test_exs, False)

        # storing the accuracy of this model for cross validation
        if accuracy > max_accuracy:
            best_model = model
            max_accuracy = accuracy

    return best_model


def train_cnn_classifier(args, all_exs: List[BillExample], word_embeddings: WordEmbeddings):
    """
    :param args: Command-line args so you can access them here
    :param all_exs: all examples, including training and testing
    :param word_embeddings: set of loaded word embeddings
    :return: an RNNClassifier instance trained on the given data
    """
    dropout = 0.5
    num_epochs = 6
    lr = 0.0004
    batch_size = 1
    window_sizes = (3, 4, 5)
    NUM_FILTERS = 100
    num_labels = 30

    # we iterate through 5 different folds to cross validate the training/test sets
    kfold = KFold(5, True)
    max_accuracy = -1
    best_model = None
    k = 1

    for train_index, test_index in kfold.split(all_exs):
        train_exs = []
        for i in train_index:
            train_exs.append(all_exs[i])

        test_exs = []
        for i in test_index:
            test_exs.append(all_exs[i])

        # instantiating the CNN object
        conv_neural_net = CNN(num_filters=NUM_FILTERS,
                              window_sizes=window_sizes, dropout=dropout, word_embeddings=word_embeddings,
                              num_classes=num_labels)

        # specifying tools for training
        conv_neural_net.train()
        optimizer = optim.Adam(conv_neural_net.parameters(), lr=lr)
        loss_function = nn.CrossEntropyLoss()
        ex_indices = [i for i in range(0, len(train_exs))]
        for epoch in range(num_epochs):
            random.shuffle(ex_indices)
            start_time = time.time()
            total_loss = 0.0
            for i in range(0, len(ex_indices), batch_size):
                indices = ex_indices[i: min(i + batch_size, len(ex_indices))]
                # create batches of documents
                batch_x_words = [train_exs[idx].words for idx in indices]
                max_length = np.amax(np.array([len(x) for x in batch_x_words]))
                upper_bound = 1000
                batch_x_indices = []
                # transforming everything in batch to be indexed words
                for sentence in batch_x_words:
                    bound = min(upper_bound, max_length)
                    indexes = [word_embeddings.word_indexer.index_of(sentence[b]) if b < len(sentence) else 0 for b in
                               range(0, bound)]
                    indexes = [1 if i == -1 else i for i in indexes]
                    batch_x_indices.append(indexes)

                batch_x = torch.LongTensor(batch_x_indices)
                batch_y = torch.LongTensor([train_exs[idx].label for idx in indices])

                conv_neural_net.zero_grad()
                # pass batch through the network
                predictions = conv_neural_net.forward(batch_x)
                loss = loss_function(predictions, batch_y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            print("Total loss on epoch %i: %f. Time to run %f" % (epoch, total_loss, (time.time() - start_time)))

        model = CNNClassifier(conv_neural_net, word_embeddings, train_exs, test_exs)
        print("=====Cross Validation Split %d=====" % k)
        k += 1
        accuracy = evaluate(model, test_exs, False)

        # storing the accuracy of this model for cross validation
        if accuracy > max_accuracy:
            best_model = model
            max_accuracy = accuracy

    return best_model
