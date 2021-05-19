"""Experiment fom cheese/fire problem.
Expreriments and reults for cheese problem using
BTPlanningProblem and Q-learning algorithm."""


import numpy as np
import pickle

from joblib import Parallel, delayed

from mdp import GridMDP, GridMDPModfy, orientations

from ltl2btrevised import (
    Globally, Finally, PropConditionNode, getrandomtrace)

from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees import common

from flloat.parser.ltlf import LTLfParser
import numpy as np
import time


# Just a simple condition node that implements atomic propositions
class ActionNode(Behaviour):
    """Action node for the planning atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the action node for the planning LTLf propositions.
    """

    def __init__(self, name):
        """Init method for the condition node."""
        super(ActionNode, self).__init__(name)
        self.action_symbol = name


    # def setup(self, timeout, value=False):
    def setup(self, timeout, env, i=0):
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
        self.env = env

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
        try:
            if self.trace[self.index][self.proposition_symbol]:
                return_value = common.Status.SUCCESS
            else:
                return_value = common.Status.FAILURE
        except IndexError:
            return_value = common.Status.FAILURE

        return return_value

def init_mdp(sloc):
    """Initialized a simple MDP world."""
    grid = np.ones((4, 4)) * -0.04

    # Obstacles
    grid[3][0] = None
    grid[2][2] = None
    grid[1][1] = None

    # Cheese and trap
    grid[0][3] = None
    grid[1][3] = None

    grid = np.where(np.isnan(grid), None, grid)
    grid = grid.tolist()

    # Terminal and obstacles defination
    grid[0][3] = +2
    grid[1][3] = -2

    mdp = GridMDPModfy(
        grid, terminals=[None, None], startloc=sloc)

    return mdp
