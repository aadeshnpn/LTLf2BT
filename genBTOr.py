"""Experiment to check the validation of generative BT
on two tasks with Or binary operator.
"""

import numpy as np
import pickle

# from joblib import Parallel, delayed

from mdp import GridMDPCheeseBeans, orientations, dictmax, create_policy

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
    grid[3][3] = None

    grid = np.where(np.isnan(grid), None, grid)
    grid = grid.tolist()

    # Terminal and obstacles defination
    grid[0][3] = +2
    grid[3][3] = +1

    if np.random.rand() > 0.5:
        grid[0][2] = -2
    else:
        grid[3][2] = -2

    mdp = GridMDPCheeseBeans(
        grid, terminals=[None, None], startloc=sloc)

    return mdp


def setup_node(nodes, trace, k):
    for node in nodes:
        # print(node, node.name, trace)
        node.setup(0, trace, k)


def advance_exp():
    mdp = init_mdp_seq((0, 1))
    # goalspec = '(e & f) | (true & (X ( !f U (e & f) ) ))'
    goalspec_cheese = '(G(!ct) & c) | (G(!ct) & (X (G(!ct) U (G(!ct) & c))))'
    goalspec_beans = '(G(!bt) & b) | (G(!bt) & (X (G(!bt) U (G(!bt) & b))))'
    goalspec_home = '((c | b) & h) | (true & (X ((F(G(!ct) & c) | F(G(!bt) & b)) U ((c|b) & h))))'
    # goalspec_home = '(G(!bt & !ct) & (c|b) & h) | (G(!bt & !ct) & (X (G(!bt & !ct) U (G(!bt & !ct) & (c|b) & h))))'
    # goalspec = '('+ goalspec_cheese + ') & X (' + goalspec_home +')'
    goalspec = '(('+ goalspec_cheese + ') | (' + goalspec_beans +')) & X (' + goalspec_home +')'
    bboard = blackboard.Client(name='fire')
    bboard.register_key(key='trace', access=common.Access.WRITE)
    bboard.trace = []
    bboard.trace.append(mdp.generate_default_props())
    trace = [
        {'c': False, 'b': False, 'h':False, 'ct': False, 'bt':False},
        {'c': False, 'b': False, 'h':False,'ct': False, 'bt':False},
        {'c': False, 'b': False, 'h':False, 'ct': False, 'bt':False},
        {'c': False, 'b': False, 'h':False, 'ct': False, 'bt':False},
        {'c': False, 'b': True, 'h':False, 'ct': False, 'bt':False},
        {'c': False, 'b': False , 'h':False, 'ct': False, 'bt':False},
        ]
    bboard.trace = trace
    # recbt = create_rec_bt()
    # recbt = cheese_bt()
    # recbt = beans_bt()
    recbt = home_bt()
    for i in range(1):
        # print(py_trees.display.ascii_tree(genbt[0].root))
        [node.reset() for node in recbt[0].root.children]
        recbt[0].root.children[1].reset()
        # genbt[0].root.children[0].children[0].children[1].reset()
        setup_node(recbt[1:], bboard.trace, k=0)
        recbt[0].tick()
        print(bboard.trace, mdp.curr_loc, recbt[0].root.status)

    # genbt = create_gen_bt(recbt[0], mdp)
    # for i in range(5):
    #     # print(py_trees.display.ascii_tree(genbt[0].root))
    #     [node.reset() for node in recbt[0].root.children]
    #     # genbt[0].root.children[0].children[0].children[1].reset()
    #     setup_node(recbt[1:] + genbt[1:], bboard.trace, k=0)
    #     genbt[0].tick()
    #     print('Tick',i, bboard.trace, mdp.curr_loc)
    #     # print(
    #     #     i, genbt[0].root.status, bboard.trace,
    #     #     [(a.name, a.status) for a in genbt[3:]])

    parser = LTLfParser()
    formula = parser(goalspec_beans)
    # print(formula.truth(bboard.trace), formula)
    # print('GEN', formula.truth(bboard.trace), genbt[0].root.status)
    print('REC', formula.truth(bboard.trace), recbt[0].root.status)


def cheese_bt(rec=True, env=None, qtable=None):
    main = Selector('RCMain')
    cheese = PropConditionNode('c')
    # Trap global constraint
    ctrap = PropConditionNode('ct')
    negctrap = Negation(ctrap, 'NegCTrap')
    gctrap = Globally(negctrap, 'GCTrap')

    # Post condition
    pandseq = Sequence('PostCondAndC')
    pandseq.add_children([gctrap, cheese])
    pand = And(pandseq)

    if not rec:
        cheeseact = ActionNode('c', env, qtable)
        ctrapa = PropConditionNode('t')
        negctrapa = Negation(ctrapa, 'NegCTrap')
        gctrapa = Globally(negctrapa, 'GCTrap')
        pandseqa = Sequence('PostCondActionC')
        pandseqa.add_children([gctrapa, cheeseact])
        panda = And(pandseqa)
    # Until
    # Trap global constraint
    ctrap1 = PropConditionNode('ct')
    negctrap1 = Negation(ctrap1, 'NegCTrap1')
    gctrap1 = Globally(negctrap1, 'GTCrap1')

    parll2 = Sequence('UntilAndC')
    untila = UntilA(gctrap1)
    if not rec:
        untilb = UntilB(panda)
    else:
        untilb = UntilB(copy.copy(pand))
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until = Until(anddec2)
    next = Next(until)
    # next = Finally(until)
    parll1 = Sequence('TrueNextC')
    # Trap global constraint
    ctrap2 = PropConditionNode('ct')
    negctrap2 = Negation(ctrap2, 'NegCTrap2')
    gctrap2 = Globally(negctrap2, 'GCTrap2')

    parll1.add_children([gctrap2, next])
    anddec1 = And(parll1)
    # Root node
    main.add_children([pand, anddec1])
    bt = BehaviourTree(main)
    return bt, next, cheese, gctrap, gctrap1, gctrap2


def beans_bt(rec=True, env=None, qtable=None):
    main = Selector('RBMain')
    beans = PropConditionNode('b')
    # Trap global constraint
    btrap = PropConditionNode('bt')
    negbtrap = Negation(btrap, 'NegBTrap')
    gbtrap = Globally(negbtrap, 'GBTrap')

    # Post condition
    pandseq = Sequence('PostCondAndB')
    pandseq.add_children([gbtrap, beans])
    pand = And(pandseq)

    if not rec:
        beansact = ActionNode('b', env, qtable)
        btrapa = PropConditionNode('bt')
        negbtrapa = Negation(btrapa, 'NegBTrap')
        gbtrapa = Globally(negbtrapa, 'GBTrap')
        pandseqa = Sequence('PostCondActionB')
        pandseqa.add_children([gbtrapa, beansact])
        panda = And(pandseqa)
    # Until
    # Trap global constraint
    btrap1 = PropConditionNode('bt')
    negbtrap1 = Negation(btrap1, 'NegBTrap1')
    gbtrap1 = Globally(negbtrap1, 'GTBrap1')

    parll2 = Sequence('UntilAndB')
    untila = UntilA(gbtrap1)
    if not rec:
        untilb = UntilB(panda)
    else:
        untilb = UntilB(copy.copy(pand))
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until = Until(anddec2)
    next = Next(until)
    # next = Finally(until)
    parll1 = Sequence('TrueNextB')
    # Trap global constraint
    btrap2 = PropConditionNode('bt')
    negbtrap2 = Negation(btrap2, 'NegBTrap2')
    gbtrap2 = Globally(negbtrap2, 'GBTrap2')

    parll1.add_children([gbtrap2, next])
    anddec1 = And(parll1)
    # Root node
    main.add_children([pand, anddec1])
    bt = BehaviourTree(main)
    return bt, next, beans, gbtrap, gbtrap1, gbtrap2


def home_bt(rec=True, env=None, qtable=None):
    # goalspec_home = '((c | b) & h) | (true & (X ((F(G(!ct) & c) | F(G(!bt) & b)) U ((c|b) & h))))'
    main = Selector('RHMain')
    home = PropConditionNode('h')
    cheese = [PropConditionNode('c') for i in range(3)]
    beans  = [PropConditionNode('b') for i in range(3)]

    # Post condition
    pandseq = Sequence('PostCondAnd')
    pandsel = Selector('PostCondOr')
    pandsel.add_children([cheese[0], beans[0]])
    pandseq.add_children([pandsel, home])
    pand = And(pandseq)

    if not rec:
        beansact = ActionNode('h', env, qtable)
        pandseqa = Sequence('PostCondActionB')
        pandsela = Selector('PostCondActionOrB')
        pandsela.add_children([cheese[1], beans[1]])
        pandseqa.add_children([pandseqa, beansact])
        panda = And(pandseqa)
    # Until
    # Trap global constraint and task constraint
    btrap1 = PropConditionNode('bt')
    negbtrap1 = Negation(btrap1, 'NegBTrap1')
    gbtrap1 = Globally(negbtrap1, 'GTBrap1')
    taskcb = Sequence('TaskCnstrB')
    taskcb.add_children([gbtrap1, beans[2]])
    fincb = Finally(And(taskcb))
    ctrap1 = PropConditionNode('ct')
    negctrap1 = Negation(ctrap1, 'NegCTrap1')
    gctrap1 = Globally(negctrap1, 'GTCrap1')
    taskcc = Sequence('TaskCnstrC')
    taskcc.add_children([gctrap1, cheese[2]])
    fincc = Finally(And(taskcc))
    taskcnstr = Selector('TaskCnstr')
    taskcnstr.add_children([fincb, fincc])

    parll2 = Sequence('UntilAndB')
    untila = UntilA(taskcnstr)
    if not rec:
        untilb = UntilB(panda)
    else:
        untilb = UntilB(copy.copy(pand))
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until = Until(anddec2)
    # next = Next(until)
    next = Finally(until)
    parll1 = Sequence('TrueNextB')
    atrue = AlwaysTrueNode('ATrue')
    parll1.add_children([atrue, next])
    anddec1 = And(parll1)
    # Root node
    main.add_children([pand, anddec1])
    bt = BehaviourTree(main)
    print(py_trees.display.ascii_tree(bt.root))
    return (bt, *cheese, *beans, home, next)


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