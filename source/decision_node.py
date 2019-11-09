# -*- coding: utf-8 -*-


class DecisionNode:
    """
    The Vehicle object contains a lot of vehicles

    Args:
        col (int): The arg is used for...
        value (float):
        results (dict):
        tb (DecisionNode):
        fb (DecisionNode):

    Attributes:
        col (str): This is where we store arg,
    """
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
