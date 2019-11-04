# -*- coding: utf-8 -*-
from .context import code

import unittest
import torch


class RandomForestTest(unittest.TestCase):
    """Binary test tree."""

    def test_init(self):
        random_forest = code.TorchRandomForestClassifier(10, 2, 3)
        self.assertTrue(random_forest.max_depth == 3)

    def test_fit(self):
        random_forest = code.TorchRandomForestClassifier(10, 2, 3)
        vectors = torch.FloatTensor([[0,1],[1,2],[4,2],[8,3]])
        labels = torch.LongTensor([0,1,0,0])
        random_forest.fit(vectors, labels)

    def test_predict(self):
        random_forest = code.TorchRandomForestClassifier(10, 2, 3)
        vectors = torch.FloatTensor([[0,1],[1,2],[4,2],[8,3]])
        labels = torch.LongTensor([0,1,0,0])
        random_forest.fit(vectors, labels)
        random_forest.predict(vectors[0])


if __name__ == '__main__':
    unittest.main()
