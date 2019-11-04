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
        vectors = torch.FloatTensor([[0,1],[1,2]])
        labels = torch.LongTensor([0,1])
        binary_tree.fit(vectors, labels)

    def test_predict(self):
        binary_tree = code.TorchDecisionTreeClassifier(1)
        vectors = torch.FloatTensor([[0,1],[1,2]])
        labels = torch.LongTensor([0,1])
        binary_tree.fit(vectors, labels)
        binary_tree.predict(vectors[0])


if __name__ == '__main__':
    unittest.main()
