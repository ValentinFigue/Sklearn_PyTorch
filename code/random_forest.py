# -*- coding: utf-8 -*-
import random
import torch
from math import sqrt

from .binary_tree import TorchDecisionTreeClassifier


class TorchRandomForestClassifier(torch.nn.Module):

    def __init__(self,  nb_trees, nb_samples, max_depth=-1):
        self.trees = []
        self.trees_features = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth

    def fit(self, vectors, labels):

        for _ in range(self.nb_trees):
            tree = TorchDecisionTreeClassifier(self.max_depth)
            sampled_vectors, sample_labels = sample_vectors(vectors, labels, self.nb_samples)
            list_features = sample_dimensions(sampled_vectors)
            self.trees_features.append(list_features)
            sampled_featured_vectors = torch.index_select(sampled_vectors,1, list_features)
            self.nb_trees.append(tree.fit(sampled_featured_vectors, sample_labels))

    def predict(self, vector):

        predictions = []
        for tree, index_features in zip(self.trees,self.trees_features):
            sampled_vector = torch.index_select(tree,0,index_features)
            predictions.append(tree.predict(sampled_vector))

        return max(set(predictions), key=predictions.count)


def sample_vectors(vectors, labels, nb_samples):

    sampled_indices = torch.LongTensor(random.sample(range(len(vectors)), nb_samples))
    sampled_vectors = torch.index_select(vectors,0, sampled_indices)
    sampled_labels = torch.index_select(labels,0, sampled_indices)

    return sampled_vectors, sampled_labels


def sample_dimensions(vectors) :

    sample_dimension = torch.LongTensor(random.sample(range(len(vectors[0])), int(sqrt(len(vectors[0])))))

    return sample_dimension
#
# class RandomForestClassifier(object):
#
#     """
#     :param  nb_trees:       Number of decision trees to use
#     :param  nb_samples:     Number of samples to give to each tree
#     :param  max_depth:      Maximum depth of the trees
#     :param  max_workers:    Maximum number of processes to use for training
#     """
#     def __init__(self, nb_trees, nb_samples, max_depth=-1, max_workers=1):
#         self.trees = []
#         self.nb_trees = nb_trees
#         self.nb_samples = nb_samples
#         self.max_depth = max_depth
#         self.max_workers = max_workers
#
#     """
#     Trains self.nb_trees number of decision trees.
#     :param  data:   A list of lists with the last element of each list being
#                     the value to predict
#     """
#     def fit(self, data):
#         with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
#             rand_fts = map(lambda x: [x, random.sample(data, self.nb_samples)],
#                            range(self.nb_trees))
#             self.trees = list(executor.map(self.train_tree, rand_fts))
#
#     """
#     Trains a single tree and returns it.
#     :param  data:   A List containing the index of the tree being trained
#                     and the data to train it
#     """
#     def train_tree(self, data):
#         logging.info('Training tree {}'.format(data[0] + 1))
#         tree = DecisionTreeClassifier(max_depth=self.max_depth)
#         tree.fit(data[1])
#         return tree
#
#     """
#     Returns a prediction for the given feature. The result is the value that
#     gets the most votes.
#     :param  feature:    The features used to predict
#     """
#     def predict(self, feature):
#         predictions = []
#
#         for tree in self.trees:
#             predictions.append(tree.predict(feature))
#
#         return max(set(predictions), key=predictions.count)
#
#
# def test_rf():
#     from sklearn.model_selection import train_test_split
#
#     data = CSVReader.read_csv("../data/income.csv")
#     train, test = train_test_split(data, test_size=0.3)
#
#     rf = RandomForestClassifier(nb_trees=60, nb_samples=3000, max_workers=4)
#     rf.fit(train)
#
#     errors = 0
#     features = [ft[:-1] for ft in test]
#     values = [ft[-1] for ft in test]
#
#     for feature, value in zip(features, values):
#         prediction = rf.predict(feature)
#         if prediction != value:
#             errors += 1
#
#     logging.info("Error rate: {}".format(errors / len(features) * 100))
#
#
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     test_rf()
