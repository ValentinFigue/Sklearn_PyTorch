# -*- coding: utf-8 -*-


class DecisionNode:
    """
    Node decision object which is used to build a binary tree. It groups the splitting function and either the different
    nodes below (left and right) or a result value to return.

    Args:
        col (:class:`int`): The dimension along which the splitting is performed.
        value (:class:`float`): The value which splits the space into two different spaces for the dimension specified
            above.
        results (:class:`dict` or :class:`torch.FloatTensor`): The results value to return if no splitting function is
            given. It can be either a dictionary where the keys correspond to the labels returned and the different
            count associated, either a value tensor to return.
        tb (:class:`DecisionNode`): The node to call recursively in the case the splitting function returns true.
        fb (:class:`DecisionNode`): The node to call recursively in the case the splitting function returns false.
    """
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
