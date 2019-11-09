# -*- coding: utf-8 -*-
import torch

from .binary_tree import TorchDecisionTreeClassifier, TorchDecisionTreeRegressor
from .utils import sample_vectors, sample_dimensions


class TorchRandomForestClassifier(torch.nn.Module):

    def __init__(self,  nb_trees, nb_samples, max_depth=-1, bootstrap=True):
        self.trees = []
        self.trees_features = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.bootstrap = bootstrap

    def fit(self, vectors, labels):

        for _ in range(self.nb_trees):
            tree = TorchDecisionTreeClassifier(self.max_depth)
            list_features = sample_dimensions(vectors)
            self.trees_features.append(list_features)
            if self.bootstrap:
                sampled_vectors, sample_labels = sample_vectors(vectors, labels, self.nb_samples)
                sampled_featured_vectors = torch.index_select(sampled_vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, sample_labels)
            else:
                sampled_featured_vectors = torch.index_select(vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, labels)
            self.trees.append(tree)

    def predict(self, vector):

        predictions = []
        for tree, index_features in zip(self.trees, self.trees_features):
            sampled_vector = torch.index_select(vector, 0, index_features)
            predictions.append(tree.predict(sampled_vector))

        return max(set(predictions), key=predictions.count)


class TorchRandomForestRegressor(torch.nn.Module):

    def __init__(self,  nb_trees, nb_samples, max_depth=-1, bootstrap=True):
        self.trees = []
        self.trees_features = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.bootstrap = bootstrap

    def fit(self, vectors, values):

        for _ in range(self.nb_trees):
            tree = TorchDecisionTreeRegressor(self.max_depth)
            list_features = sample_dimensions(vectors)
            self.trees_features.append(list_features)
            if self.bootstrap:
                sampled_vectors, sample_labels = sample_vectors(vectors, values, self.nb_samples)
                sampled_featured_vectors = torch.index_select(sampled_vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, sample_labels)
            else:
                sampled_featured_vectors = torch.index_select(vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, values)
            self.trees.append(tree)

    def predict(self, vector):

        predictions_sum = 0
        for tree, index_features in zip(self.trees, self.trees_features):
            sampled_vector = torch.index_select(vector, 0, index_features)
            predictions_sum += tree.predict(sampled_vector)

        return predictions_sum/len(self.trees)
