"""Experiment fom cheese/fire problem.
Expreriments and reults for cheese problem using
BTPlanningProblem and Q-learning algorithm."""


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
        self.qtable = dict()
        for state in self.env.states:
            self.qtable[state] = dict(zip(orientations, [0, 0, 0, 0]))
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1

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

        ## Random planning algorithm
        # self.env.step(self.env.env_action_dict[np.random.choice([0, 1, 2, 3])])
        # self.blackboard.trace.append(self.env.generate_default_props())
        # if self.blackboard.trace[-1]['s33']:
        #     return common.Status.SUCCESS
        # else:
        #     return common.Status.FAILURE

        ## Qlearning
        state = self.env.curr_loc
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice([0, 1, 2, 3])
            action = orientations[action]
        else:
            action = dictmax(self.qtable[state], s='key')
        p, s1 = zip(*self.env.T(state, action))
        p, s1 = list(p), list(s1)
        s1 = s1[np.argmax(p)]
        next_state = s1
        reward = self.env.R(next_state)
        old_value = self.qtable[state][action]
        next_max = dictmax(self.qtable[next_state], s='val')
        new_value = (1-self.alpha) * old_value + self.alpha * (
            reward + self.gamma * next_max)
        self.qtable[state][action] = new_value
        state = next_state
        self.env.curr_loc = state
        self.blackboard.trace.append(self.env.generate_default_props())
        if self.blackboard.trace[-1]['s33']:
            return common.Status.SUCCESS
        else:
            return common.Status.FAILURE


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
    mdp = init_mdp((3, 0))
    goalspec = 'G (!s32) & F (s33)'
    cnode = PropConditionNode('s32')
    gconstaint = Negation(cnode, 'Invert')
    globallyd = Globally(gconstaint)
    anode = ActionNode('a33', mdp)
    # finallya = Finally(anode)
    # parll = Parallel('Parll')
    # parll.add_children([globallyd, anode])
    seq = Sequence('Seq')
    seq.add_children([globallyd, anode])
    anddec = And(seq)
    ppa1 = Selector('Selector')
    ppa2 = Sequence('Sequence')
    cnode2 = PropConditionNode('s33')
    finallya = Finally(cnode2)
    ppa2.add_children([anddec])
    ppa1.add_children([finallya, ppa2])
    blackboard1 = blackboard.Client(name='cheese')
    blackboard1.register_key(key='trace', access=common.Access.WRITE)
    blackboard1.trace = [mdp.generate_default_props()]

    bt = BehaviourTree(ppa1)
    print(len(blackboard1.trace))
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    # output = py_trees.display.ascii_tree(bt.root)
    # print(output)
    for i in range(50):
        setup_node([anddec, finallya], blackboard1.trace, k=0)
        bt.tick()
        # print(len(blackboard1.trace), bt.root.status, blackboard1.trace)
        # print(gconstaint.status)
        if gconstaint.status == common.Status.FAILURE:
            mdp.restart()
            blackboard1.trace = [mdp.generate_default_props()]

    print(bt.root.status, blackboard1.trace)
    print(mdp.to_arrows(create_policy(anode.qtable)))


if __name__ == '__main__':
    main()