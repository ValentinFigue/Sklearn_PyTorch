# -*- coding: utf-8 -*-
from .context import Sklearn_PyTorch

import unittest
import torch


class BinaryTreeClassifierTest(unittest.TestCase):
    """Binary classifier tree test."""

    def test_init(self):
        binary_tree = Sklearn_PyTorch.TorchDecisionTreeClassifier(1)
        self.assertTrue(binary_tree.max_depth == 1)

    def test_fit(self):
        binary_tree = Sklearn_PyTorch.TorchDecisionTreeClassifier(1)
        vectors = torch.FloatTensor([[0,1],[0,2]])
        labels = torch.LongTensor([0,1])
        binary_tree.fit(vectors, labels)
        self.assertTrue(binary_tree._root_node.col == 1)
        self.assertTrue(binary_tree._root_node.value == 2)
        binary_tree = Sklearn_PyTorch.TorchDecisionTreeClassifier(10)
        binary_tree.fit(vectors, labels)
        self.assertTrue(binary_tree._root_node.tb.tb is None)
        self.assertTrue(binary_tree._root_node.fb.fb is None)

    def test_predict(self):
        binary_tree = Sklearn_PyTorch.TorchDecisionTreeClassifier(1)
        vectors = torch.FloatTensor([[0,1],[0,2]])
        labels = torch.LongTensor([0,1])
        binary_tree.fit(vectors, labels)
        result = binary_tree.predict(vectors[0])
        self.assertTrue(result == 0)
        result = binary_tree.predict(vectors[1])
        self.assertTrue(result == 1)
        vector = torch.FloatTensor([-1,8])
        result = binary_tree.predict(vector)
        self.assertTrue(result == 1)


class BinaryTreeRegressorTest(unittest.TestCase):
    """Binary regressor tree test."""

    def test_init(self):
        binary_tree = Sklearn_PyTorch.TorchDecisionTreeRegressor(1)
        self.assertTrue(binary_tree.max_depth == 1)

    def test_fit(self):
        binary_tree = Sklearn_PyTorch.TorchDecisionTreeRegressor(1)
        vectors = torch.FloatTensor([[0,1],[0,2]])
        values = torch.FloatTensor([0,1])
        binary_tree.fit(vectors, values)
        self.assertTrue(binary_tree._root_node.col == 1)
        self.assertTrue(binary_tree._root_node.value == 2)
        binary_tree = Sklearn_PyTorch.TorchDecisionTreeClassifier(10)
        binary_tree.fit(vectors, values)
        self.assertTrue(binary_tree._root_node.tb.tb is None)
        self.assertTrue(binary_tree._root_node.fb.fb is None)

    def test_predict(self):
        binary_tree = Sklearn_PyTorch.TorchDecisionTreeRegressor(1)
        vectors = torch.FloatTensor([[0,1],[0,2]])
        values = torch.FloatTensor([0,1])
        binary_tree.fit(vectors, values)
        result = binary_tree.predict(vectors[0])
        self.assertTrue(result.item() == 0)
        result = binary_tree.predict(vectors[1])
        self.assertTrue(result.item() == 1)
        vector = torch.FloatTensor([-1,8])
        result = binary_tree.predict(vector)
        self.assertTrue(result.item() == 1)


if __name__ == '__main__':
    unittest.main()
