"""Experiment to check the validation of generative BT.
"""

import numpy as np
import pickle

# from joblib import Parallel, delayed

from mdp import GridMDP, GridMDPModfy, orientations, dictmax, create_policy

from ltl2btrevised import (
    Globally, Finally, Negation, PropConditionNode,
    getrandomtrace, And)

from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees.composites import Sequence, Selector, Parallel
from py_trees import common, blackboard
import py_trees

from flloat.parser.ltlf import LTLfParser
import numpy as np
import time


# Just a simple condition node that implements atomic propositions
class AlwaysTrueNode(Behaviour):
    """Always True node for the true proposition.

    Inherits the Behaviors class from py_trees. This
    behavior implements the true node for the true proposition.
    """

    def __init__(self, name):
        """Init method for the true node."""
        super(AlwaysTrueNode, self).__init__(name)
        self.proposition_symbol = name


    # def setup(self, timeout, value=False):
    def setup(self, timeout, trace=None, i=0):
        """Have defined the setup method.

        This method defines the other objects required for the
        condition node.
        Args:
        timeout: property from Behavior super class. Not required
        symbol: Name of the proposition symbol
        value: A dict object with key as the proposition symbol and
               boolean value as values. Supplied by trace.
        """
        self.index = i
        self.trace = trace

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def reset(self, i=0):
        self.index = i

    def increment(self):
        self.index += 1

    def update(self):
        """
        Main function that is called when BT ticks.
        """
        return common.Status.SUCCESS