"""Experiment fom cheese/fire problem.
Expreriments and reults for cheese problem using
BTPlanningProblem and Q-learning algorithm."""


import numpy as np
import pickle

# from joblib import Parallel, delayed

from mdp import GridMDP, GridMDPModfy, orientations

from ltl2btrevised import (
    Globally, Finally, Negation, PropConditionNode,
    getrandomtrace, And)

from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees.composites import Sequence, Selector, Parallel
from py_trees import common, blackboard

from flloat.parser.ltlf import LTLfParser
import numpy as np
import time


# Just a simple condition node that implements atomic propositions
class ActionNode(Behaviour):
    """Action node for the planning atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the action node for the planning LTLf propositions.
    """

    def __init__(self, name, env):
        """Init method for the condition node."""
        super(ActionNode, self).__init__(name)
        self.action_symbol = name
        self.blackboard = blackboard.Client(name=name)
        self.blackboard.register_key(key='trace', access=common.Access.WRITE)
        self.env = env


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
        self.env.step(self.env.env_action_dict[np.random.choice([0, 1, 2, 3])])
        self.blackboard.trace.append(self.env.generate_default_props())
        return common.Status.SUCCESS


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

def setup_node(nodes, trace, k):
    for node in nodes:
        node.setup(0, trace, k)

def main():
    mdp = init_mdp((1, 3))
    goalspec = 'G (!s32) & F (s33)'
    cnode = PropConditionNode('s32')
    gconstaint = Negation(cnode, 'Invert')
    globallyd = Globally(gconstaint)
    anode = ActionNode('s33', mdp)
    finallya = Finally(anode)
    parll = Parallel('And')
    parll.add_children([globallyd, finallya])
    anddec = And(parll)

    blackboard1 = blackboard.Client(name='cheese')
    blackboard1.register_key(key='trace', access=common.Access.WRITE)
    blackboard1.trace = [mdp.generate_default_props()]

    bt = BehaviourTree(anddec)

    for i in range(5):
        setup_node([anddec], blackboard1.trace, k=0)
        bt.tick()

    print(blackboard1.trace)


if __name__ == '__main__':
    main()