# -*- coding: utf-8 -*-
import torch

from .binary_tree import TorchDecisionTreeClassifier, TorchDecisionTreeRegressor
from .utils import sample_vectors, sample_dimensions


class TorchRandomForestClassifier(torch.nn.Module):
    """
    Torch random forest object used to solve classification problem. This object implements the fitting and prediction
    function which can be used with torch tensors. The random forest is based on
    :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeClassifier` which are built during the :func:`fit` and called
    recursively during the :func:`predict`.

    Args:
        nb_trees (:class:`int`): Number of :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeClassifier` used to fit the
            classification problem.
        nb_samples (:class:`int`): Number of vector samples used to fit each
            :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeClassifier`.
        max_depth (:class:`int`): The maximum depth which corresponds to the maximum successive number of
            :class:`DecisionNode`.
        bootstrap (:class:`bool`): If set to true, a sample of the dimensions of the input vectors are made during the
            fitting and the prediction.

    """
    def __init__(self,  nb_trees, nb_samples, max_depth=-1, bootstrap=True):
        self.trees = []
        self.trees_features = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.bootstrap = bootstrap

    def fit(self, vectors, labels):
        """
        Function which must be used after the initialisation to fit the random forest and build the successive
        :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeClassifier` to solve a specific classification problem.

        Args:
            vectors(:class:`torch.FloatTensor`): Vectors tensor used to fit the random forest. It represents the data
                and must correspond to the following shape (num_vectors, num_dimensions).
            labels (:class:`torch.LongTensor`): Labels tensor used to fit the decision tree. It represents the labels
                associated to each vectors and must correspond to the following shape (num_vectors).

        """
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
        """
        Function which must be used after the the fitting of the random forest. It calls recursively the different
        :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeClassifier` to classify the vector.

        Args:
            vector(:class:`torch.FloatTensor`): Vectors tensor which must be classified. It represents the data
                and must correspond to the following shape (num_dimensions).

        Returns:
            :class:`torch.LongTensor`: Tensor which corresponds to the label predicted by the random forest.

        """
        predictions = []
        for tree, index_features in zip(self.trees, self.trees_features):
            sampled_vector = torch.index_select(vector, 0, index_features)
            predictions.append(tree.predict(sampled_vector))

        return max(set(predictions), key=predictions.count)


class TorchRandomForestRegressor(torch.nn.Module):
    """
    Torch random forest object used to solve regression problem. This object implements the fitting and prediction
    function which can be used with torch tensors. The random forest is based on
    :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` which are built during the :func:`fit` and called
    recursively during the :func:`predict`.

    Args:
        nb_trees (:class:`int`): Number of :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` used to fit the
            classification problem.
        nb_samples (:class:`int`): Number of vector samples used to fit each
            :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor`.
        max_depth (:class:`int`): The maximum depth which corresponds to the maximum successive number of
            :class:`Sklearn_PyTorch.decision_node.DecisionNode`.
        bootstrap (:class:`bool`): If set to true, a sample of the dimensions of the input vectors are made during the
            fitting and the prediction.

    """
    def __init__(self,  nb_trees, nb_samples, max_depth=-1, bootstrap=True):
        self.trees = []
        self.trees_features = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.bootstrap = bootstrap

    def fit(self, vectors, values):
        """
        Function which must be used after the initialisation to fit the random forest and build the successive
        :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` to solve a specific classification problem.

        Args:
            vectors(:class:`torch.FloatTensor`): Vectors tensor used to fit the decision tree. It represents the data
                and must correspond to the following shape (num_vectors, num_dimensions_vectors).
            values(:class:`torch.FloatTensor`): Values tensor used to fit the decision tree. It represents the values
                associated to each vectors and must correspond to the following shape (num_vectors,
                num_dimensions_values).

        """
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
        """
        Function which must be used after the the fitting of the random forest. It calls recursively the different
        :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` to regress the vector.

        Args:
            vector(:class:`torch.FloatTensor`): Vectors tensor which must be regressed. It represents the data
                and must correspond to the following shape (num_dimensions).

        Returns:
            :class:`torch.FloatTensor`: Tensor which corresponds to the value regressed by the random forest.

        """
        predictions_sum = 0
        for tree, index_features in zip(self.trees, self.trees_features):
            sampled_vector = torch.index_select(vector, 0, index_features)
            predictions_sum += tree.predict(sampled_vector)

        return predictions_sum/len(self.trees)
