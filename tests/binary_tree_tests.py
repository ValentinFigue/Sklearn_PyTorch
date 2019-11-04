# -*- coding: utf-8 -*-
from .context import code

import unittest
import torch


class BinaryTreeTest(unittest.TestCase):
    """Binary test tree."""

    def test_init(self):
        binary_tree = code.TorchDecisionTreeClassifier(1)
        self.assertTrue(binary_tree.max_depth == 1)

    def test_fit(self):
        binary_tree = code.TorchDecisionTreeClassifier(1)
        vectors = torch.FloatTensor([[0,1],[0,2]])
        labels = torch.LongTensor([0,1])
        binary_tree.fit(vectors, labels)
        self.assertTrue(binary_tree.root_node.col == 1)
        self.assertTrue(binary_tree.root_node.value == 2)

    def test_predict(self):
        binary_tree = code.TorchDecisionTreeClassifier(1)
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


if __name__ == '__main__':
    unittest.main()
