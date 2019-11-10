# -*- coding: utf-8 -*-
from math import log, sqrt
import torch
import random


def unique_counts(labels):
    """
    Unique count function used to count labels.
    """
    results = {}
    for label in labels:
        value = label.item()
        if value not in results.keys():
            results[value] = 0
        results[value] += 1
    return results


def divide_set(vectors, labels, column, value):
    """
    Divide the sets into two different sets along a specific dimension and value.
    """
    set_1 = [(vector, label) for vector, label in zip(vectors, labels) if split_function(vector, column, value)]
    set_2 = [(vector, label) for vector, label in zip(vectors, labels) if not split_function(vector, column, value)]

    vectors_set_1 = [element[0] for element in set_1]
    vectors_set_2 = [element[0] for element in set_2]
    label_set_1 = [element[1] for element in set_1]
    label_set_2 = [element[1] for element in set_2]

    return vectors_set_1, label_set_1, vectors_set_2, label_set_2


def split_function(vector, column, value):
    """
    Split function
    """
    return vector[column] >= value


def log2(x):
    """
    Log2 function
    """
    return log(x) / log(2)


def sample_vectors(vectors, labels, nb_samples):
    """
    Sample vectors and labels uniformly.
    """
    sampled_indices = torch.LongTensor(random.sample(range(len(vectors)), nb_samples))
    sampled_vectors = torch.index_select(vectors,0, sampled_indices)
    sampled_labels = torch.index_select(labels,0, sampled_indices)

    return sampled_vectors, sampled_labels


def sample_dimensions(vectors):
    """
    Sample vectors along dimension uniformly.
    """
    sample_dimension = torch.LongTensor(random.sample(range(len(vectors[0])), int(sqrt(len(vectors[0])))))

    return sample_dimension


def entropy(labels):
    """
    Entropy function.
    """
    results = unique_counts(labels)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(labels)
        ent = ent - p * log2(p)
    return ent


def variance(values):
    """
    Variance function.
    """
    mean_value = mean(values)
    var = 0.0
    for value in values:
        var = var + torch.sum(torch.sqrt(torch.pow(value-mean_value,2))).item()/len(values)
    return var


def mean(values):
    """
    Mean function.
    """
    m = 0.0
    for value in values:
        m = m + value/len(values)
    return m
