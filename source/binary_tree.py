# -*- coding: utf-8 -*-
import torch

from .decision_node import DecisionNode
from .utils import unique_counts, split_function, divide_set, entropy, variance, mean


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

    def build_tree(self, vectors, labels, func, depth):
        if len(vectors) == 0:
            return DecisionNode()
        if depth == 0:
            return DecisionNode(results=unique_counts(labels))

        current_score = func(labels)
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
                gain = current_score - p * func(label_set_1) - (1 - p) * func(label_set_2)
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
            if split_function(vector, node.col, node.value):
                branch = node.tb
            else:
                branch = node.fb

            return self.classify(vector, branch)


class TorchDecisionTreeRegressor(torch.nn.Module):

    def __init__(self, max_depth=-1):
        self.root_node = None
        self.max_depth = max_depth

    def fit(self, vectors, values, criterion=None):
        if len(vectors) < 1:
            raise ValueError("Not enough samples in the given dataset")
        if len(vectors) != len(values):
            raise ValueError("Labels and data vectors must have the same number of elements")
        if not criterion:
            criterion = variance

        self.root_node = self.build_tree(vectors, values, criterion, self.max_depth)

    def build_tree(self, vectors, values, func, depth):
        if len(vectors) == 0:
            return DecisionNode()
        if depth == 0:
            return DecisionNode(results=mean(values))

        current_score = func(values)
        best_gain = 0.0
        best_criteria = None
        best_sets = None
        column_count = len(vectors[0])

        for col in range(0, column_count):
            column_values = {}
            for vector in vectors:
                column_values[vector[col]] = 1
            for value in column_values.keys():
                vectors_set_1, values_set_1, vectors_set_2, values_set_2 = divide_set(vectors, values, col, value)

                p = float(len(vectors_set_1)) / len(vectors)
                gain = current_score - p * func(values_set_1) - (1 - p) * func(vectors_set_2)
                if gain > best_gain and len(vectors_set_1) > 0 and len(vectors_set_2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = ((vectors_set_1,values_set_1), (vectors_set_2,values_set_2))

        if best_gain > 0:
            true_branch = self.build_tree(best_sets[0][0], best_sets[0][1], func, depth - 1)
            false_branch = self.build_tree(best_sets[1][0], best_sets[1][1], func, depth - 1)
            return DecisionNode(col=best_criteria[0],
                                value=best_criteria[1],
                                tb=true_branch, fb=false_branch)
        else:
            return DecisionNode(results=mean(values))

    def predict(self, vector):

        return self.regress(vector, self.root_node)

    def regress(self, vector, node):

        if node.results is not None:
            return node.results
        else:
            if split_function(vector, node.col, node.value):
                branch = node.tb
            else:
                branch = node.fb

            return self.regress(vector, branch)
