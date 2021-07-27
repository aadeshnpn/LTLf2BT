"""Experiment to check the validation of generative BT
on two sequential tasks with Until binary operator.
"""

import numpy as np
import pickle

# from joblib import Parallel, delayed

from mdp import GridMDPFire, orientations, dictmax, create_policy

from ltl2btrevised import (
    Globally, Finally, Negation, Next, PropConditionNode,
    getrandomtrace, And, Until, UntilA, UntilB)

from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees.composites import Sequence, Selector, Parallel
from py_trees import common, blackboard
import py_trees
from flloat.parser.ltlf import LTLfParser
import numpy as np
import time
import copy


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


# Just a simple action node that already has an optimal
# policy for the simple grid world.
class ActionNode(Behaviour):
    """Action node for the planning atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the action node for the planning LTLf propositions.
    """

    def __init__(self, name, env, qtable, recbt=None):
        """Init method for the action node."""
        super(ActionNode, self).__init__(name)
        self.action_symbol = name
        self.blackboard = blackboard.Client(name='fire')
        self.blackboard.register_key(key='trace', access=common.Access.WRITE)
        self.env = env
        self.qtable = dict()
        for state in self.env.states:
            self.qtable[state] = dict(zip(orientations, [0, 0, 0, 0]))
        for i in range(0,4):
            self.qtable[(i,3)][(1,0)] = 1

        self.step = 0
        # Update the qtable with a learned policy
        self.qtable.update(qtable)

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
        state = self.env.curr_loc
        action = dictmax(self.qtable[state], s='key')
        # print('action', action)
        p, s1 = zip(*self.env.T(state, action))
        p, s1 = list(p), list(s1)
        s1 = s1[np.argmax(p)]
        # print('State', s1)
        self.blackboard.trace.append(self.env.generate_props_loc(s1))
        self.env.curr_loc = s1
        self.step += 1
        # if self.blackboard.trace[-1][self.action_symbol]:
        # if self.env.state_map[self.action_symbol] == 's'+str(s1[0])+str(s1[1]):
        print(self.step, self.action_symbol, self.blackboard.trace[-1])
        if self.blackboard.trace[-1][self.action_symbol]:
            # self.blackboard.trace.append(self.env.generate_props_loc(s1))
            return common.Status.SUCCESS
        # elif 's'+str(s1[0])+str(s1[1]) == 's32':
        elif self.step >=6:            # self.blackboard.trace.append(self.env.generate_props_loc(s1))
            return common.Status.FAILURE
        else:
            return common.Status.RUNNING
        # return common.Status.RUNNING


def init_mdp_seq(sloc):
    """Initialized a simple MDP world."""
    grid = np.ones((4, 4)) * -0.04

    # Obstacles
    grid[3][0] = None
    grid[2][2] = None
    grid[1][1] = None

    # Cheese and trapfinallya
    grid[0][3] = None
    grid[2][3] = None

    grid = np.where(np.isnan(grid), None, grid)
    grid = grid.tolist()

    # Terminal and obstacles defination
    grid[0][3] = +2
    grid[2][3] = -2

    mdp = GridMDPFire(
        grid, terminals=[None, None], startloc=sloc)

    return mdp


def setup_node(nodes, trace, k):
    for node in nodes:
        # print(node, node.name, trace)
        node.setup(0, trace, k)


def advance_exp():
    mdp = init_mdp_seq((0, 3))
    # goalspec = '(s33)|(true & (X (true U s33)))'
    # goalspec_cheese = '(G(!t) & c) | (G(!t) & (X (G(!t) U (G(!t) & c))))'
    # goalspec_home = '(G(!t) & c & h) | (G(!t) & (X ((G(!t)) U (G(!t) & c & h))))'
    goalspec = '(e & f) | (true & (X ( !f U (e & f) ) ))'
    bboard = blackboard.Client(name='fire')
    bboard.register_key(key='trace', access=common.Access.WRITE)
    bboard.trace = []
    bboard.trace.append(mdp.generate_default_props())
    trace = [
        {'e': False, 'f': False, },
        {'e': False, 'f': False, },
        {'e': False, 'f': False, },
        {'e': True, 'f': False, },
        {'e': True, 'f': False, },
        {'e': True, 'f': True, },
        ]
    # bboard.trace = trace
    recbt = create_rec_bt()
    # for i in range(2):
    #     # print(py_trees.display.ascii_tree(genbt[0].root))
    #     [node.reset() for node in recbt[0].root.children]
    #     recbt[0].root.children[1].reset()
    #     # genbt[0].root.children[0].children[0].children[1].reset()
    #     setup_node(recbt[1:], bboard.trace, k=0)
    #     recbt[0].tick()
    #     print(bboard.trace, mdp.curr_loc, recbt[0].root.status)

    genbt = create_gen_bt(recbt[0], mdp)
    for i in range(5):
        # print(py_trees.display.ascii_tree(genbt[0].root))
        [node.reset() for node in recbt[0].root.children]
        # genbt[0].root.children[0].children[0].children[1].reset()
        setup_node(recbt[1:] + genbt[1:], bboard.trace, k=0)
        genbt[0].tick()
        print('Tick',i, bboard.trace, mdp.curr_loc)
        # print(
        #     i, genbt[0].root.status, bboard.trace,
        #     [(a.name, a.status) for a in genbt[3:]])

    parser = LTLfParser()
    formula = parser(goalspec)
    print(bboard.trace)
    print('GEN', formula.truth(bboard.trace), genbt[0].root.status)
    print('REC', formula.truth(bboard.trace), recbt[0].root.status)


def create_rec_bt():
    # goalspec = '(e & f) | (true & (X ( !f U (e & f) ) ))'
    # Ext and Fire
    main = Selector('RCMain')
    ext = PropConditionNode('e')
    # Trap global constraint
    fire = PropConditionNode('f')

    # Post condition
    pandseq = Sequence('PostCondAnd')
    pandseq.add_children([ext, fire])
    pand = And(pandseq)

    # Until
    # Trap global constraint
    fire1 = PropConditionNode('f')
    negfire = Negation(fire1, 'NegFire')

    parll2 = Sequence('UntilAnd')
    untila = UntilA(negfire)
    untilb = UntilB(copy.copy(pand))
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until = Until(anddec2)
    next = Next(until)
    # next = Finally(until)
    parll1 = Sequence('TrueNext')
    # Trap global constraint
    altrue = AlwaysTrueNode('True')
    parll1.add_children([altrue, next])
    anddec1 = And(parll1)
    # Root node
    main.add_children([pand, anddec1])

    bt = BehaviourTree(main)
    # print(py_trees.display.ascii_tree(bt.root))
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    return bt, next, fire, fire1, ext


def get_qtable_ext(mdp):
    qtable = dict()
    for state in mdp.states:
        qtable[state] = dict(zip(orientations, [0, 0, 0, 0]))
    for i in range(0,4):
        qtable[(i,3)][(1,0)] = 1
    qtable[(3,1)][(0,1)] = 0.8
    qtable[(3,1)][(0,-1)] = 0.2
    qtable[(3,0)][(0,1)] = 0.8
    qtable[(3,0)][(0,-1)] = 0.2
    return qtable


def get_qtable_fire(mdp):
    qtable = dict()
    for state in mdp.states:
        qtable[state] = dict(zip(orientations, [0, 0, 0, 0]))
    for i in range(0,4):
        qtable[(3,i)][(0,-1)] = 1
    # qtable[(3,1)][(0,1)] = 0.8
    # qtable[(3,1)][(0,-1)] = 0.2
    # qtable[(3,0)][(0,1)] = 0.8
    # qtable[(3,0)][(0,-1)] = 0.2
    return qtable


def create_gen_bt(recbt, mdp):
    gensel = Selector('Generator')
    genseq = Sequence('GMain')
    qtable_ext = get_qtable_ext(mdp)
    qtable_fire = get_qtable_fire(mdp)
    extact = ActionNode('e', mdp, qtable=qtable_ext)
    fireact = ActionNode('f', mdp, qtable=qtable_fire)

    # Ext and Fire
    main = Selector('GCMain')
    ext = PropConditionNode('e')
    # Trap global constraint
    fire = PropConditionNode('f')

    # Post condition
    pandseq = Sequence('PostCondAnd')
    pandseq.add_children([ext, fire])
    pand = And(pandseq)

    # Actions
    aandseq = Sequence('ActionAnd')
    aandseq.add_children([extact, fireact])
    aand = And(aandseq)

    # Until
    # Trap global constraint
    fire1 = PropConditionNode('f')
    negfire = Negation(fire1, 'NegFire')

    parll2 = Sequence('UntilAnd')
    untila = UntilA(negfire)
    untilb = UntilB(aand)
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until = Until(anddec2)
    next = Next(until)
    # next = Finally(until)
    parll1 = Sequence('TrueNext')
    # Trap global constraint
    altrue = AlwaysTrueNode('True')
    parll1.add_children([altrue, next])
    anddec1 = And(parll1)
    # Root node
    main.add_children([pand, anddec1])

    genseq.add_children([main])
    gensel.add_children([recbt.root, genseq])
    # gensel.add_children([genseq])
    bt = BehaviourTree(gensel)
    print(py_trees.display.ascii_tree(bt.root))
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    return (
        bt, next, fire, fire1, ext, extact, fireact
    )


def main():
    advance_exp()
    # print(bt.root.status, blackboard1.trace)
    # print(mdp.to_arrows(create_policy(anode.qtable)))


if __name__ == '__main__':
    main()