# -*- coding: utf-8 -*-
from .context import Sklearn_PyTorch

import unittest
import torch


class RandomForestClassifierTest(unittest.TestCase):
    """Random forest classifier tests."""

    def test_init(self):
        random_forest = Sklearn_PyTorch.TorchRandomForestClassifier(10, 2, 3)
        self.assertTrue(random_forest.max_depth == 3)

    def test_fit(self):
        random_forest = Sklearn_PyTorch.TorchRandomForestClassifier(10, 2, 3)
        vectors = torch.FloatTensor([[0,1],[1,2],[4,2],[8,3]])
        labels = torch.LongTensor([0,1,0,0])
        random_forest.fit(vectors, labels)
        self.assertTrue(len(random_forest.trees) == 10)
        self.assertTrue(len(random_forest.trees_features) == 10)

    def test_predict(self):
        random_forest = Sklearn_PyTorch.TorchRandomForestClassifier(400, 1, 3)
        vectors = torch.FloatTensor([[0],[1],[4],[8]])
        labels = torch.LongTensor([0,1,0,0])
        random_forest.fit(vectors, labels)
        result = random_forest.predict(vectors[0])
        self.assertTrue(result == 0)


class RandomForestRegressorTest(unittest.TestCase):
    """Random forest regressor tests."""

    def test_init(self):
        random_forest = Sklearn_PyTorch.TorchRandomForestRegressor(10, 2, 3)
        self.assertTrue(random_forest.max_depth == 3)

    def test_fit(self):
        random_forest = Sklearn_PyTorch.TorchRandomForestRegressor(10, 2, 3)
        vectors = torch.FloatTensor([[0,1],[1,2],[4,2],[8,3]])
        values = torch.FloatTensor([0,1,0,0])
        random_forest.fit(vectors, values)
        self.assertTrue(len(random_forest.trees) == 10)
        self.assertTrue(len(random_forest.trees_features) == 10)

    def test_predict(self):
        random_forest = Sklearn_PyTorch.TorchRandomForestRegressor(400, 1, 3)
        vectors = torch.FloatTensor([[0],[1],[4],[8]])
        values = torch.FloatTensor([0,1,0,0])
        random_forest.fit(vectors, values)
        result = random_forest.predict(vectors[0])
        self.assertTrue(result.item() > 0)
        self.assertTrue(result.item() < 1)


if __name__ == '__main__':
    unittest.main()
