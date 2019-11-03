# -*- coding: utf-8 -*-
from .context import code

import unittest


class BinaryTreeTest(unittest.TestCase):
    """Binary test tree."""

    def test_init(self):
        binary_tree = code.TorchDecisionTreeClassifier(1)
        self.assertTrue(binary_tree.max_depth == 1)


if __name__ == '__main__':
    unittest.main()