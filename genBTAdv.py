"""Experiment to check the validation of generative BT
on two sequential tasks.
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
        self.blackboard = blackboard.Client(name=name)
        self.blackboard.register_key(key='trace', access=common.Access.WRITE)
        self.env = env
        self.qtable = dict()
        for state in self.env.states:
            self.qtable[state] = dict(zip(orientations, [0, 0, 0, 0]))
        for i in range(0,4):
            self.qtable[(i,3)][(1,0)] = 1

        # If started near the trap making it probable to bump into trap
        # self.qtable[(3,1)][(0,1)] = 0.8
        # self.qtable[(3,1)][(0,-1)] = 0.2
        # self.qtable[(3,0)][(0,1)] = 0.8
        # self.qtable[(3,0)][(0,-1)] = 0.2
        # self.step = 0
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
        if 's'+str(s1[0])+str(s1[1]) == self.action_symbol:
            # self.blackboard.trace.append(self.env.generate_props_loc(s1))
            return common.Status.SUCCESS
        # elif 's'+str(s1[0])+str(s1[1]) == 's32':
        elif self.step >=10:
            # self.blackboard.trace.append(self.env.generate_props_loc(s1))
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
        # print(node, node.name, trace)
        node.setup(0, trace, k)


def advance_exp():
    mdp = init_mdp((3, 0))
    # goalspec = '(s33)|(true & (X (true U s33)))'
    goalspec_cheese = '(G(!t) & c) | (G(!t) & (F (G(!t) U (G(!t) & c))))'
    # goalspec_home = '(G(!t) & c & h) | (G(!t) & (F ((G(!t)) U (G(!t) & c & h))))'
    # goalspec = '('+ goalspec_cheese + ') & X (' + goalspec_home +')'
    goalspec = goalspec_cheese
    bboard = blackboard.Client(name='cheese')
    bboard.register_key(key='trace', access=common.Access.WRITE)
    bboard.trace = []
    # bboard.trace.append(mdp.generate_default_props())
    trace = [
        {'c': False, 't': False, 'h':False},
        {'c': False, 't': False, 'h':False},
        {'c': False, 't': False, 'h':False},
        {'c': True, 't': False, 'h':False},
        # {'c': True, 't': False, 'h':True},
        # {'c': True, 't': False, 'h':True},
        # {'c': True, 't': False, 'h':True},
        # {'c': True, 't': False, 'h':True},
        # {'c': True, 't': False, 'h':True},
        ]
    bboard.trace = trace
    recbt = create_rec_bt()
    # genbt = create_gen_bt(recbt[0], mdp)
    for i in range(1):
        # print(py_trees.display.ascii_tree(genbt[0].root))
        # recbt[0].root.children[1].reset()
        # genbt[0].root.children[0].children[0].children[0].reset()
        setup_node(recbt[1:], bboard.trace, k=0)
        recbt[0].tick()
        # print(
        #     i, genbt[0].root.status, bboard.trace,
        #     [(a.name, a.status) for a in genbt[3:]])

    parser = LTLfParser()
    formula = parser(goalspec)
    print(bboard.trace)
    print(formula.truth(bboard.trace), recbt[0].root.status)


def create_rec_bt():
    # goalspec_cheese = '(G(!t) & c) | (G(!t) & (F (G(!t) U (G(!t) & c))))'
    # Cheese
    main = Selector('RCMain')
    cheese = PropConditionNode('c')
    # Trap global constraint
    trap = PropConditionNode('t')
    negtrap = Negation(trap, 'NegTrap')
    gtrap = Globally(negtrap, 'GTrap')

    # Post condition
    pandseq = Sequence('PostCondAnd')
    pandseq.add_children([gtrap, cheese])
    pand = And(pandseq)

    # Until
    # Trap global constraint
    trap1 = PropConditionNode('t')
    negtrap1 = Negation(trap1, 'NegTrap1')
    gtrap1 = Globally(negtrap1, 'GTrap1')

    parll2 = Sequence('UntilAnd')
    untila = UntilA(gtrap1)
    untilb = UntilB(copy.copy(pand))
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until = Until(anddec2)
    # next = Next(until)
    next = Finally(until)
    parll1 = Sequence('TrueNext')
    # Trap global constraint
    trap2 = PropConditionNode('t')
    negtrap2 = Negation(trap2, 'NegTrap2')
    gtrap2 = Globally(negtrap2, 'GTrap2')

    parll1.add_children([gtrap2, next])
    anddec1 = And(parll1)
    # Root node
    main.add_children([pand, anddec1])

    # From Cheese and home
    # goalspec_cheese = '(G(!t) & c) |   (G(!t) & (F (G(!t) U (G(!t) & c))))'
    # goalspec_home = '(G(!t) & c & h) | (G(!t) & (F ((G(!t)) U (G(!t) & c & h))))'

    mainh = Selector('RHMain')
    cheeseh = PropConditionNode('c')
    # Trap global constraint
    # trap = PropConditionNode('t')
    home = PropConditionNode('h')
    negtraph = Negation(copy.copy(trap), 'NegTrapH')
    gtraph = Globally(negtraph, 'GTrapH')

    # Post condition
    pandseqh = Sequence('PostCondAndH')
    pandseqh.add_children([gtraph, cheeseh, home])
    pandh = And(pandseqh)

    # Until
    # Trap global constraint
    trap1h = PropConditionNode('t')
    negtrap1h = Negation(trap1h, 'NegTrap1H')
    gtrap1h = Globally(negtrap1h, 'GTrap1H')

    parll2h = Sequence('UntilAndH')
    untilah = UntilA(gtrap1h)
    untilbh = UntilB(copy.copy(pandh))
    parll2h.add_children([untilbh, untilah])
    anddec2h = And(parll2h)
    untilh = Until(anddec2h)
    # next = Next(until)
    nexth = Finally(untilh)
    parll1h = Sequence('TrueNextH')
    # Trap global constraint
    trap2h = PropConditionNode('t')
    negtrap2h = Negation(trap2h, 'NegTrap2H')
    gtrap2h = Globally(negtrap2h, 'GTrap2H')

    parll1h.add_children([gtrap2h, nexth])
    anddec1h = And(parll1h)
    # Root node
    mainh.add_children([pandh, anddec1h])
    # goalspec = '('+ goalspec_cheese + ') & X (' + goalspec_home +')'
    nextgoal = Next(mainh)

    join = Sequence('Join')
    join.add_children([main, nextgoal])
    allgoal = And(join)
    bt = BehaviourTree(allgoal)
    print(py_trees.display.ascii_tree(bt.root))
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    return bt, next, cheese, gtrap, gtrap1, gtrap2


def create_gen_bt(rbt, mdp):
    gmain = Selector('Main')
    seqg = Sequence('SeqG')

    main = Selector('GMain')
    cheese = PropConditionNode('s33')
    cheeseaction = ActionNode('s33', mdp)
    # Trap global constraint
    trap = PropConditionNode('s32')
    negtrap = Negation(trap, 'NegTrap')
    gtrap = Globally(negtrap, 'GTrap')
    # Trap global constraint
    trap3 = PropConditionNode('s32')
    negtrap3 = Negation(trap3, 'NegTrap3')
    gtrap3 = Globally(negtrap3, 'GTrap3')
    # Post condition
    pandseq = Sequence('PostCondAnd')
    pandseq.add_children([gtrap3, cheese])
    pand = And(pandseq)

    # Post action
    # Trap global constraint
    trap1 = PropConditionNode('s32')
    negtrap1 = Negation(trap1, 'NegTrap1')
    gtrap1 = Globally(negtrap1, 'GTrap1')

    aandseq = Sequence('PostActionAnd')
    aandseq.add_children([gtrap1, cheeseaction])
    aand = And(aandseq)

    # Until
    parll2 = Sequence('UntilAnd')
    # Trap global constraint
    trap2 = PropConditionNode('s32')
    negtrap2 = Negation(trap2, 'NegTrap2')
    gtrap2 = Globally(negtrap2, 'GTrap2')

    untila = UntilA(gtrap2)
    untilb = UntilB(aand)
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until = Until(anddec2)
    # Next
    # next = Next(until)
    next = Finally(until)
    parll1 = Sequence('TrueNext', memory=False)
    parll1.add_children([gtrap, next])
    anddec1 = And(parll1,'ANDCHECK')
    # Root node
    main.add_children([pand, anddec1])
    seqg.add_children([main])
    # gmain.add_children([rbt.root, seqg])
    gmain.add_children([seqg])
    bt = BehaviourTree(gmain)

    py_trees.logging.level = py_trees.logging.Level.DEBUG
    # return bt, next, cheese, trap, trap1, trap2, trap3, gtrap, gtrap1, gtrap2, gtrap3
    return bt, next, cheese, trap, gtrap, gtrap1, gtrap2, gtrap3, anddec1, anddec2, pand


def main():
    advance_exp()
    # print(bt.root.status, blackboard1.trace)
    # print(mdp.to_arrows(create_policy(anode.qtable)))


if __name__ == '__main__':
    main()