# -*- coding: utf-8 -*-
import torch

from .decision_node import DecisionNode
from .utils import unique_counts, split_function, divide_set, entropy, variance, mean


class TorchDecisionTreeClassifier(torch.nn.Module):
    """
    Torch decision tree object used to solve classification problem. This object implements the fitting and prediction
    function which can be used with torch tensors. The binary tree is based on
    :class:`Sklearn_PyTorch.decision_node.DecisionNode` which are built during the :func:`fit` and called recursively during the
    :func:`predict`.

    Args:
        max_depth (:class:`int`): The maximum depth which corresponds to the maximum successive number of
            :class:`Sklearn_PyTorch.decision_node.DecisionNode`.

    """
    def __init__(self, max_depth=-1):
        self._root_node = None
        self.max_depth = max_depth

    def fit(self, vectors, labels, criterion=None):
        """
        Function which must be used after the initialisation to fit the binary tree and build the successive
        :class:`Sklearn_PyTorch.decision_node.DecisionNode` to solve a specific classification problem.

        Args:
            vectors (:class:`torch.FloatTensor`): Vectors tensor used to fit the decision tree. It represents the data
                and must correspond to the following shape [num_vectors, num_dimensions].
            labels (:class:`torch.LongTensor`): Labels tensor used to fit the decision tree. It represents the labels
                associated to each vectors and must correspond to the following shape [num_vectors].
            criterion (:class:`function`): Optional function used to optimize the splitting for each
                :class:`Sklearn_PyTorch.decision_node.DecisionNode`. If none given, the entropy function is used.
        """
        if len(vectors) < 1:
            raise ValueError("Not enough samples in the given dataset")
        if len(vectors) != len(labels):
            raise ValueError("Labels and data vectors must have the same number of elements")
        if not criterion:
            criterion = entropy

        self._root_node = self._build_tree(vectors, labels, criterion, self.max_depth)

    def _build_tree(self, vectors, labels, func, depth):
        """
        Private recursive function used to build the tree.
        """
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
            true_branch = self._build_tree(best_sets[0][0], best_sets[0][1], func, depth - 1)
            false_branch = self._build_tree(best_sets[1][0], best_sets[1][1], func, depth - 1)
            return DecisionNode(col=best_criteria[0],
                                value=best_criteria[1],
                                tb=true_branch, fb=false_branch)
        else:
            return DecisionNode(results=unique_counts(labels))

    def predict(self, vector):
        """
        Function which must be used after the the fitting of the binary tree. It calls recursively the different
        :class:`Sklearn_PyTorch.decision_node.DecisionNode` to classify the vector.

        Args:
            vector(:class:`torch.FloatTensor`): Vectors tensor which must be classified. It represents the data
                and must correspond to the following shape (num_dimensions).

        Returns:
            :class:`torch.LongTensor`: Tensor which corresponds to the label predicted by the binary tree.

        """
        return self._classify(vector, self._root_node)

    def _classify(self, vector, node):
        """
        Private recursive function used to classify with the tree.
        """
        if node.results is not None:
            return list(node.results.keys())[0]
        else:
            if split_function(vector, node.col, node.value):
                branch = node.tb
            else:
                branch = node.fb

            return self._classify(vector, branch)


class TorchDecisionTreeRegressor(torch.nn.Module):
    """
    Torch decision tree object used to solve regression problem. This object implements the fitting and prediction
    function which can be used with torch tensors. The binary tree is based on
    :class:`Sklearn_PyTorch.decision_node.DecisionNode` which are built during the :func:`fit` and called recursively during the
    :func:`predict`.

    Args:
        max_depth (:class:`int`): The maximum depth which corresponds to the maximum successive number of
            :class:`Sklearn_PyTorch.decision_node.DecisionNode`.

    """
    def __init__(self, max_depth=-1):
        self._root_node = None
        self.max_depth = max_depth

    def fit(self, vectors, values, criterion=None):
        """
        Function which must be used after the initialisation to fit the binary tree and build the successive
        :class:`Sklearn_PyTorch.decision_node.DecisionNode` to solve a specific regression problem.

        Args:
            vectors(:class:`torch.FloatTensor`): Vectors tensor used to fit the decision tree. It represents the data
                and must correspond to the following shape (num_vectors, num_dimensions_vectors).
            values(:class:`torch.FloatTensor`): Values tensor used to fit the decision tree. It represents the values
                associated to each vectors and must correspond to the following shape (num_vectors,
                num_dimensions_values).
            criterion(:class:`function`): Optional function used to optimize the splitting for each
                :class:`Sklearn_PyTorch.decision_node.DecisionNode`. If none given, the variance function is used.
        """
        if len(vectors) < 1:
            raise ValueError("Not enough samples in the given dataset")
        if len(vectors) != len(values):
            raise ValueError("Labels and data vectors must have the same number of elements")
        if not criterion:
            criterion = variance

        self._root_node = self._build_tree(vectors, values, criterion, self.max_depth)

    def _build_tree(self, vectors, values, func, depth):
        """
        Private recursive function used to build the tree.
        """
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
            true_branch = self._build_tree(best_sets[0][0], best_sets[0][1], func, depth - 1)
            false_branch = self._build_tree(best_sets[1][0], best_sets[1][1], func, depth - 1)
            return DecisionNode(col=best_criteria[0],
                                value=best_criteria[1],
                                tb=true_branch, fb=false_branch)
        else:
            return DecisionNode(results=mean(values))

    def predict(self, vector):
        """
        Function which must be used after the the fitting of the binary tree. It calls recursively the different
        :class:`Sklearn_PyTorch.decision_node.DecisionNode` to regress the vector.

        Args:
            vector(:class:`torch.FloatTensor`): Vectors tensor which must be regressed. It represents the data
                and must correspond to the following shape (num_dimensions).

        Returns:
            :class:`torch.FloatTensor`: Tensor which corresponds to the value regressed by the binary tree.

        """
        return self._regress(vector, self._root_node)

    def _regress(self, vector, node):
        """
        Private recursive function used to regress on the tree.
        """
        if node.results is not None:
            return node.results
        else:
            if split_function(vector, node.col, node.value):
                branch = node.tb
            else:
                branch = node.fb

            return self._regress(vector, branch)
