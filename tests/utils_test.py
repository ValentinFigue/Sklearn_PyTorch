# -*- coding: utf-8 -*-
from .context import Sklearn_PyTorch

import unittest
import torch


class UtilsTest(unittest.TestCase):
    """Utils functions test."""

    def test_unique_count(self):
        labels = torch.LongTensor([0, 1, 0, 0])
        count = Sklearn_PyTorch.unique_counts(labels)
        self.assertTrue(isinstance(count, dict))
        self.assertTrue(len(count) == 2)
        self.assertTrue(count[0] == 3)
        self.assertTrue(count[1] == 1)

    def test_divide_set(self):
        vectors = torch.FloatTensor([[0,1],[1,2],[4,2],[8,3]])
        labels = torch.LongTensor([0,1,0,0])
        vectors_set_1, label_set_1, vectors_set_2, label_set_2 = Sklearn_PyTorch.divide_set(vectors, labels, 0, 2)
        self.assertTrue(len(vectors_set_1) == 2)
        self.assertTrue(len(label_set_1) == 2)
        self.assertTrue(len(vectors_set_2) == 2)
        self.assertTrue(len(label_set_2) == 2)
        vectors_set_1, label_set_1, vectors_set_2, label_set_2 = Sklearn_PyTorch.divide_set(vectors, labels, 1, 2)
        self.assertTrue(len(vectors_set_1) == 3)
        self.assertTrue(len(label_set_1) == 3)
        self.assertTrue(len(vectors_set_2) == 1)
        self.assertTrue(len(label_set_2) == 1)

    def test_split_function(self):
        vector = torch.FloatTensor([0,1])
        self.assertTrue(Sklearn_PyTorch.split_function(vector, 0, 0))
        self.assertFalse(Sklearn_PyTorch.split_function(vector, 1, 2))

    def test_log(self):
        self.assertTrue(Sklearn_PyTorch.log2(2) == 1)

    def test_entropy(self):
        labels = torch.LongTensor([0, 0, 0, 0])
        self.assertTrue(Sklearn_PyTorch.entropy(labels) == 0)
        self.assertTrue(Sklearn_PyTorch.entropy(torch.LongTensor([0, 0, 0, 0])) < Sklearn_PyTorch.entropy(torch.LongTensor([0, 1, 0, 0])))
        self.assertTrue(Sklearn_PyTorch.entropy(torch.LongTensor([0, 1, 0, 0])) < Sklearn_PyTorch.entropy(torch.LongTensor([0, 1, 2, 0])))
        self.assertTrue(Sklearn_PyTorch.entropy(torch.LongTensor([0, 1, 2, 0])) < Sklearn_PyTorch.entropy(torch.LongTensor([0, 1, 2, 3])))

    def test_sample_vectors(self):
        vectors = torch.FloatTensor([[0,1],[1,2],[4,2],[8,3]])
        labels = torch.LongTensor([0,1,0,0])
        sampled_vectors, sample_labels = Sklearn_PyTorch.sample_vectors(vectors, labels, 1)
        self.assertTrue(len(sampled_vectors) == 1)
        self.assertTrue(len(sample_labels) == 1)
        self.assertTrue(sampled_vectors[0] in vectors)
        self.assertTrue(sample_labels[0] in labels)
        sampled_vectors, sample_labels = Sklearn_PyTorch.sample_vectors(vectors, labels, 2)
        self.assertTrue(len(sampled_vectors) == 2)
        self.assertTrue(len(sample_labels) == 2)
        self.assertTrue(sampled_vectors[0] in vectors)
        self.assertTrue(sample_labels[0] in labels)
        self.assertTrue(sampled_vectors[1] in vectors)
        self.assertTrue(sample_labels[1] in labels)

    def test_sample_dimension(self):
        vectors = torch.FloatTensor([[0,1],[1,2],[4,2],[8,3]])
        list_features = Sklearn_PyTorch.sample_dimensions(vectors)
        self.assertTrue(len(list_features) == 1)
        sampled_vectors = torch.index_select(vectors, 1, list_features)
        self.assertTrue(sampled_vectors.size()[1] == 1)

    def test_mean(self):
        values = torch.FloatTensor([[0, 1], [1, 2], [4, 2], [8, 3]])
        self.assertTrue(isinstance(Sklearn_PyTorch.mean(values), torch.FloatTensor))
        self.assertTrue(len(Sklearn_PyTorch.mean(values).size()) == 1)
        self.assertTrue(len(Sklearn_PyTorch.mean(values)) == 2)
        self.assertTrue(Sklearn_PyTorch.mean(values)[1] == 2)
        values = torch.FloatTensor([[0, 1]])
        self.assertTrue(isinstance(Sklearn_PyTorch.mean(values), torch.FloatTensor))
        self.assertTrue(len(Sklearn_PyTorch.mean(values).size()) == 1)
        self.assertTrue(len(Sklearn_PyTorch.mean(values)) == 2)
        self.assertTrue(Sklearn_PyTorch.mean(values)[1] == 1)

    def test_variance(self):
        values = torch.FloatTensor([[0, 1], [1, 2], [4, 2], [8, 3]])
        self.assertTrue(isinstance(Sklearn_PyTorch.variance(values), float))
        self.assertTrue(Sklearn_PyTorch.variance(values) > 0)
        values = torch.FloatTensor([[0, 1], [1, 1], [1, 1], [2, 1]])
        self.assertTrue(Sklearn_PyTorch.variance(values) == 0.5)
        values = torch.FloatTensor([[1, 1], [1, 1], [1, 1], [1, 1]])
        self.assertTrue(Sklearn_PyTorch.variance(values) == 0)


if __name__ == '__main__':
    unittest.main()
