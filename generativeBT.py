"""Experiment to check the validation of generative BT.
"""

import numpy as np
import pickle

# from joblib import Parallel, delayed

from mdp import GridMDP, GridMDPModfy, orientations, dictmax, create_policy

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
# policy for the simple grid world
class ActionNode(Behaviour):
    """Action node for the planning atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the action node for the planning LTLf propositions.
    """

    def __init__(self, name, env, recbt=None):
        """Init method for the action node."""
        super(ActionNode, self).__init__(name)
        self.action_symbol = name
        self.blackboard = blackboard.Client(name=name)
        self.blackboard.register_key(key='trace', access=common.Access.WRITE)
        self.env = env
        self.qtable = dict()
        for state in self.env.states:
            self.qtable[state] = dict(zip(orientations, [0, 0, 0, 0]))
        for i in range(0,4):
            self.qtable[(i,3)][(1,0)] = 1

        # If started near the trap making it probable to bump into trap
        self.qtable[(3,1)][(0,1)] = 0.8
        self.qtable[(3,1)][(0,-1)] = 0.2

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
        print('State', s1)
        self.env.curr_loc = s1
        # if self.blackboard.trace[-1][self.action_symbol]:
        if 's'+str(s1[0])+str(s1[1]) == self.action_symbol:
            self.blackboard.trace.append(self.env.generate_props_loc(s1))
            return common.Status.SUCCESS
        elif 's'+str(s1[0])+str(s1[1]) == 's32':
            self.blackboard.trace.append(self.env.generate_props_loc(s1))
            return common.Status.FAILURE
        else:
            return common.Status.RUNNING
        # return common.Status.RUNNING


def init_mdp(sloc):
    """Initialized a simple MDP world."""
    grid = np.ones((4, 4)) * -0.04

    # Obstacles
    grid[3][0] = None
    grid[2][2] = None
    grid[1][1] = None

    # Cheese and trapfinallya
    grid[0][3] = None
    # grid[1][3] = None

    grid = np.where(np.isnan(grid), None, grid)
    grid = grid.tolist()

    # Terminal and obstacles defination
    grid[0][3] = +2
    # grid[1][3] = -2

    mdp = GridMDPModfy(
        grid, terminals=[None, None], startloc=sloc)

    return mdp


def setup_node(nodes, trace, k):
    for node in nodes:
        node.setup(0, trace, k)


def create_recognizer_bt():
    main = Selector('RMain')
    cheese = PropConditionNode('s33')
    atrue = AlwaysTrueNode('AT')
    parll2 = Sequence('UntilAnd')
    untila = UntilA(copy.copy(atrue))
    untilb = UntilB(copy.copy(cheese))
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until = Until(anddec2)
    next = Next(until)
    parll1 = Sequence('TrueNext')
    parll1.add_children([atrue, next])
    anddec1 = And(parll1)
    main.add_children([cheese, anddec1])
    bt = BehaviourTree(main)
    # print(py_trees.display.ascii_tree(bt.root))
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    return bt, next, cheese


def create_generator_bt(recbt, env):
    gmain = Selector('Main')
    seqg = Sequence('SeqG')

    main = Selector('GMain')
    cheese = PropConditionNode('s33')
    acheese = ActionNode('s33', env)
    # cheese = ActionNode('s33')
    atrue = AlwaysTrueNode('AT')
    parll2 = Sequence('UntilAnd')
    untila = UntilA(copy.copy(atrue))
    untilb = UntilB(acheese)
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until = Until(anddec2)
    next = Next(until)
    parll1 = Sequence('TrueNext')
    parll1.add_children([atrue, next])
    anddec1 = And(parll1)
    main.add_children([cheese, anddec1])
    seqg.add_children([main])
    gmain.add_children([recbt.root, seqg])
    bt = BehaviourTree(gmain)
    # print(py_trees.display.ascii_tree(bt.root))
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    return bt, next, cheese


def base_exp():
    mdp = init_mdp((3, 3))
    goalspec = '(s33)|(true & (X (true U s33)))'
    # anode = ActionNode('cheese', mdp)
    bboard = blackboard.Client(name='cheese')
    bboard.register_key(key='trace', access=common.Access.WRITE)
    bboard.trace = []
    # bboard.trace.append(mdp.generate_default_props())
    trace = [
        {'s33': False},
        {'s33': True},
        ]
    bboard.trace = trace
    print(bboard.trace)
    bt = create_recognizer_bt()
    andec = bt[1:]
    bt = bt[0]
    for i in range(len(trace)):
        setup_node(andec, bboard.trace, k=0)
        bt.tick()
    parser = LTLfParser()
    formula = parser(goalspec)        # returns a LTLfFormula
    print(formula.truth(trace), bt.root.status)


def simple_exp():
    mdp = init_mdp((0, 3))
    goalspec = '(s33)|(true & (X (true U s33)))'
    bboard = blackboard.Client(name='cheese')
    bboard.register_key(key='trace', access=common.Access.WRITE)
    bboard.trace = []
    bboard.trace.append(mdp.generate_default_props())
    recbt = create_recognizer_bt()
    genbt = create_generator_bt(recbt[0], mdp)
    for i in range(3):
        recbt[0].root.children[1].reset()
        setup_node(recbt[1:] + genbt[1:], bboard.trace, k=0)
        genbt[0].tick()
        print(i, genbt[0].root.status, bboard.trace)
    parser = LTLfParser()
    formula = parser(goalspec)
    print(bboard.trace)
    print(formula.truth(bboard.trace), genbt[0].root.status)


def medium_exp():
    mdp = init_mdp((0, 3))
    # goalspec = '(s33)|(true & (X (true U s33)))'
    goalspec = '(G(!s32) & s33) | (G(!s32) & (X (G(!s32) U (G(!s31) & s32))))'
    bboard = blackboard.Client(name='cheese')
    bboard.register_key(key='trace', access=common.Access.WRITE)
    bboard.trace = []
    bboard.trace.append(mdp.generate_default_props())
    print(bboard.trace)
    def create_rec_bt():
        main = Selector('RMain')
        cheese = PropConditionNode('s33')
        # Trap global constraint
        trap = PropConditionNode('s32')
        negtrap = Negation(copy.copy(trap), 'NegTrap')
        gtrap = Globally(negtrap, 'GTrap')
        gtrapcopy1 = copy.copy(gtrap)
        gtrapcopy2 = copy.copy(gtrap)
        # Post condition
        pandseq = Sequence('PostCondAnd')
        pandseq.add_children([gtrapcopy1, cheese])
        pand = And(pandseq)

        # Until
        parll2 = Sequence('UntilAnd')
        untila = UntilA(gtrap)
        untilb = UntilB(copy.copy(pand))
        parll2.add_children([untilb, untila])
        anddec2 = And(parll2)
        until = Until(anddec2)
        # Next
        next = Next(until)
        parll1 = Sequence('TrueNext')

        parll1.add_children([gtrapcopy2, next])
        anddec1 = And(parll1)
        # Root node
        main.add_children([pand, anddec1])
        bt = BehaviourTree(main)
        # print(py_trees.display.ascii_tree(bt.root))
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        return bt, next, cheese, trap, gtrap, gtrapcopy1, gtrapcopy2

    def create_gen_bt(rbt, mdp):
        gmain = Selector('Main')
        seqg = Sequence('SeqG')

        main = Selector('GMain')
        cheese = PropConditionNode('s33')
        cheeseaction = ActionNode('s33', mdp)
        # Trap global constraint
        trap = PropConditionNode('s32')
        negtrap = Negation(copy.copy(trap), 'NegTrap')
        gtrap = Globally(negtrap, 'GTrap')
        gtrapcopy1 = copy.copy(gtrap)
        gtrapcopy2 = copy.copy(gtrap)
        gtrapcopy3 = copy.copy(gtrap)
        # Post condition
        pandseq = Sequence('PostCondAnd')
        pandseq.add_children([gtrap, cheese])
        pand = And(pandseq)

        # Post action
        aandseq = Sequence('PostActionAnd')
        aandseq.add_children([gtrapcopy1, cheeseaction])
        aand = And(aandseq)

        # Until
        parll2 = Sequence('UntilAnd')
        untila = UntilA(gtrapcopy2)
        untilb = UntilB(aand)
        parll2.add_children([untilb, untila])
        anddec2 = And(parll2)
        until = Until(anddec2)
        # Next
        next = Next(until)
        parll1 = Sequence('TrueNext')
        parll1.add_children([gtrapcopy3, next])
        anddec1 = And(parll1)
        # Root node
        main.add_children([pand, anddec1])
        seqg.add_children([main])
        gmain.add_children([rbt.root, seqg])
        bt = BehaviourTree(gmain)

        # print(py_trees.display.ascii_tree(bt.root))
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        return bt, next, cheese, trap , gtrap, gtrapcopy1, gtrapcopy2, gtrapcopy3

    recbt = create_rec_bt()
    genbt = create_gen_bt(recbt[0], mdp)
    for i in range(3):
        recbt[0].root.children[1].reset()
        setup_node(recbt[1:] + genbt[1:], bboard.trace, k=0)
        genbt[0].tick()
        print(i, genbt[0].root.status, bboard.trace)
    trace = [
        {'s33': False, 's32': False},
        {'s33': True, 's32': False},
        ]
    bboard.trace = trace
    parser = LTLfParser()
    formula = parser(goalspec)
    print(bboard.trace)
    print(formula.truth(bboard.trace), genbt[0].root.status)



def main():
    # base_exp()
    # simple_exp()
    medium_exp()
    # print(bt.root.status, blackboard1.trace)
    # print(mdp.to_arrows(create_policy(anode.qtable)))


if __name__ == '__main__':
    main()