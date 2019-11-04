# -*- coding: utf-8 -*-
from math import log
import torch

from .decision_node import DecisionNode
from .utils import unique_counts, split_function, divide_set, entropy


class TorchDecisionTreeClassifier(torch.nn.Module):

    def __init__(self, max_depth=-1):
        self.root_node = None
        self.max_depth = max_depth

    def fit(self, vectors, labels, criterion=None):
        if len(vectors) < 1:
            raise ValueError("Not enough samples in the given dataset")
        if len(vectors) != len(labels):
            raise ValueError("Labels and data vectors must have the same number of elements")
        if not criterion:
            criterion = entropy

        self.root_node = self.build_tree(vectors, labels, criterion, self.max_depth)

    def build_tree(self, vectors,labels, func, depth):
        if len(vectors) == 0:
            return DecisionNode()
        if depth == 0:
            return DecisionNode(results=unique_counts(labels))

        current_score = func(vectors)
        best_gain = 0.0
        best_criteria = None
        best_sets = None
        column_count = len(vectors[0])

        for col in range(0, column_count):
            column_values = {}
            for vector in vectors:
                column_values[vector[col]] = 1
            for value in column_values.keys():
                vectors_set_1, label_set_1, vectors_set_2, label_set_2 = divide_set(vectors, labels, col, value)

                p = float(len(vectors_set_1)) / len(vectors)
                gain = current_score - p * func(vectors_set_1) - (1 - p) * func(vectors_set_1)
                if gain > best_gain and len(vectors_set_1) > 0 and len(vectors_set_2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = ((vectors_set_1,label_set_1), (vectors_set_2,label_set_2))

        if best_gain > 0:
            true_branch = self.build_tree(best_sets[0][0], best_sets[0][1], func, depth - 1)
            false_branch = self.build_tree(best_sets[1][0], best_sets[1][1], func, depth - 1)
            return DecisionNode(col=best_criteria[0],
                                value=best_criteria[1],
                                tb=true_branch, fb=false_branch)
        else:
            return DecisionNode(results=unique_counts(labels))

    def predict(self, vector):

        return self.classify(vector, self.root_node)

    def classify(self, vector, node):

        if node.results is not None:
            return list(node.results.keys())[0]
        else:
            if split_function(vector,node.col, node.value):
                branch = node.tb
            else:
                branch = node.fb

            return self.classify(vector, branch)



# class DecisionTreeClassifier:
#
#     """
#     :param  max_depth:          Maximum number of splits during training
#     :param  random_features:    If False, all the features will be used to
#                                 train and predict. Otherwise, a random set of
#                                 size sqrt(nb features) will be chosen in the
#                                 features.
#                                 Usually, this option is used in a random
#                                 forest.
#     """
#     def __init__(self, max_depth=-1, random_features=False):
#         self.root_node = None
#         self.max_depth = max_depth
#         self.features_indexes = []
#         self.random_features = random_features
#
#     """
#     :param  rows:       The data used to rain the decision tree. It must be a
#                         list of lists. The last vaue of each inner list is the
#                         value to predict.
#     :param  criterion:  The function used to split data at each node of the
#                         tree. If None, the criterion used is entropy.
#     """
#     def fit(self, rows, criterion=None):
#         if len(rows) < 1:
#             raise ValueError("Not enough samples in the given dataset")
#
#         if not criterion:
#             criterion = self.entropy
#         if self.random_features:
#             self.features_indexes = self.choose_random_features(rows[0])
#             rows = [self.get_features_subset(row) + [row[-1]] for row in rows]
#
#         self.root_node = self.build_tree(rows, criterion, self.max_depth)
#
#     """
#     Returns a prediction for the given features.
#     :param  features:   A list of values
#     """
#     def predict(self, features):
#         if self.random_features:
#             if not all(i in range(len(features))
#                        for i in self.features_indexes):
#                 raise ValueError("The given features don't match\
#                                  the training set")
#             features = self.get_features_subset(features)
#
#         return self.classify(features, self.root_node)
#
#     """
#     Randomly selects indexes in the given list.
#     """
#     def choose_random_features(self, row):
#         nb_features = len(row) - 1
#         return random.sample(range(nb_features), int(sqrt(nb_features)))
#
#     """
#     Returns the randomly selected values in the given features
#     """
#     def get_features_subset(self, row):
#         return [row[i] for i in self.features_indexes]
#
#     """
#     Divides the given dataset depending on the value at the given column index.
#     :param  rows:   The dataset
#     :param  column: The index of the column used to split data
#     :param  value:  The value used for the split
#     """
#     def divide_set(self, rows, column, value):
#         split_function = None
#         if isinstance(value, int) or isinstance(value, float):
#             split_function = lambda row: row[column] >= value
#         else:
#             split_function = lambda row: row[column] == value
#
#         set1 = [row for row in rows if split_function(row)]
#         set2 = [row for row in rows if not split_function(row)]
#
#         return set1, set2
#
#     """
#     Returns the occurence of each result in the given dataset.
#     :param  rows:   A list of lists with the output at the last index of
#                     each one
#     """
#     def unique_counts(self, rows):
#         results = {}
#         for row in rows:
#             r = row[len(row) - 1]
#             if r not in results:
#                 results[r] = 0
#             results[r] += 1
#         return results
#
#     """
#     Returns the entropy in the given rows.
#     :param  rows:   A list of lists with the output at the last index of
#                     each one
#     """
#     def entropy(self, rows):
#         log2 = lambda x: log(x) / log(2)
#         results = self.unique_counts(rows)
#         ent = 0.0
#         for r in results.keys():
#             p = float(results[r]) / len(rows)
#             ent = ent - p * log2(p)
#         return ent
#
#     """
#     Recursively creates the decision tree by splitting the dataset until no
#     gain of information is added, or until the max depth is reached.
#     :param  rows:   The dataset
#     :param  func:   The function used to calculate the best split and stop
#                     condition
#     :param  depth:  The current depth in the tree
#     """
#     def build_tree(self, rows, func, depth):
#         if len(rows) == 0:
#             return self.DecisionNode()
#         if depth == 0:
#             return self.DecisionNode(results=self.unique_counts(rows))
#
#         current_score = func(rows)
#         best_gain = 0.0
#         best_criteria = None
#         best_sets = None
#         column_count = len(rows[0]) - 1
#
#         for col in range(0, column_count):
#             column_values = {}
#             for row in rows:
#                 column_values[row[col]] = 1
#             for value in column_values.keys():
#                 set1, set2 = self.divide_set(rows, col, value)
#
#                 p = float(len(set1)) / len(rows)
#                 gain = current_score - p * func(set1) - (1 - p) * func(set2)
#                 if gain > best_gain and len(set1) > 0 and len(set2) > 0:
#                     best_gain = gain
#                     best_criteria = (col, value)
#                     best_sets = (set1, set2)
#
#         if best_gain > 0:
#             trueBranch = self.build_tree(best_sets[0], func, depth - 1)
#             falseBranch = self.build_tree(best_sets[1], func, depth - 1)
#             return self.DecisionNode(col=best_criteria[0],
#                                      value=best_criteria[1],
#                                      tb=trueBranch, fb=falseBranch)
#         else:
#             return self.DecisionNode(results=self.unique_counts(rows))
#
#     """
#     Makes a prediction using the given features.
#     :param  observation:    The features to use to predict
#     :param  tree:           The current node
#     """
#     def classify(self, observation, tree):
#         if tree.results is not None:
#             return list(tree.results.keys())[0]
#         else:
#             v = observation[tree.col]
#             branch = None
#             if isinstance(v, int) or isinstance(v, float):
#                 if v >= tree.value:
#                     branch = tree.tb
#                 else:
#                     branch = tree.fb
#             else:
#                 if v == tree.value:
#                     branch = tree.tb
#                 else:
#                     branch = tree.fb
#             return self.classify(observation, branch)
