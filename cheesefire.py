"""Experiment fom cheese/fire problem.
Expreriments and reults for cheese problem using
BTPlanningProblem and Q-learning algorithm."""


import numpy as np
import pickle

from joblib import Parallel, delayed

from py_trees.trees import BehaviourTree
from py_trees import Status

from mdp import GridMDP, GridMDPModfy, orientations


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
        grid, terminals=[(3, 3), (3, 2)], startloc=sloc)

    return mdp
