from typing import Type
from flloat.parser.ltlf import LTLfParser

from py_trees.composites import Sequence, Selector, Parallel
from py_trees.decorators import Decorator, Inverter
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
import py_trees
from py_trees import common
import copy
import argparse
import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Method to randomly create traces
def getrandomtrace(n=2, maxtracelen=0):
    t1 = [
        {'a': False, 'b': False, 'c': False, 'd': False},
        {'a': False, 'b': False, 'c': True, 'd': False},
        {'a': False, 'b': False, 'c': False, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': False, 'c': False, 'd': False},
        {'a': True, 'b': False, 'c': True, 'd': False},
        {'a': True, 'b': False, 'c': False, 'd': True},
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': True, 'c': False, 'd': False},
        {'a': False, 'b': True, 'c': True, 'd': False},
        {'a': False, 'b': True, 'c': False, 'd': True},
        {'a': False, 'b': True, 'c': True, 'd': True},
        {'a': True, 'b': True, 'c': False, 'd': False},
        {'a': True, 'b': True, 'c': True, 'd': False},
        {'a': True, 'b': True, 'c': False, 'd': True},
        {'a': True, 'b': True, 'c': True, 'd': True},
        ]

    tracelist = []
    def gettraces():
        if maxtracelen == 0:
            m = np.random.randint(1, 50)
        else:
            m = maxtracelen
        choices = {i:t1[i] for i in range(len(t1))}
        trace = []
        for i in range(m):
            j = np.random.choice(list(range(len(t1))), m)
            trace.append(choices[j[i]])
        return trace
    for k in range(n):
        tracelist.append(gettraces())

    return tracelist


# Method that calls the BT execution node setup method
# This supplies trace at time i for the nodes
def setup_node(nodes, trace, k):
    for node in nodes:
        node.setup(0, trace, k)


# Method executes a BT passed as an argument
def execute_bt(bt, trace, nodes, i=0):
    # Args: bt -> BT to tick
    #       trace -> a trace of lenght m
    #       nodes -> Execution nodes of BT that takes trace as input

    # reset with the i value provided so it is consistent
    bt.root.reset(i)
    # setup_node(nodes, trace[k])
    setup_node(nodes, trace, i)
    bt.tick()
    return bt.root.status


# Execute both BT and Ltlf with same traces for comparision
def execute_both_bt_ltlf(subtree, formula, trace, nodes, i=0, verbos=True):
    # Args:
        # Which BT class to use
        # LTL formual
        # Trace of lenght m
        # BT exeuction nodes which require trace input
    # Trace of length 1

    # Create a BT from the subtree
    root = BehaviourTree(subtree)
    # print(py_trees.display.ascii_tree(root.root))

    # Creating a LTLf parser object
    parser = LTLfParser()
    # Parsed formula
    parsed_formula = parser(formula)
    bt_status = execute_bt(root, trace, nodes, i)
    ltlf_status = parsed_formula.truth(trace, i)
    if verbos:
        if ltlf_status == True and bt_status == common.Status.SUCCESS:
            print("trace: {}', 'BT status: {}', 'LTLf status: {}".format(trace, bt_status, ltlf_status))
        elif ltlf_status == False and bt_status == common.Status.FAILURE:
            print("trace: {}', 'BT status: {}', 'LTLf status: {}".format(trace, bt_status, ltlf_status))
        else:
            print("{} trace: {}, BT status: {}, LTLf status: {} {}".format(bcolors.WARNING, trace, bt_status, ltlf_status, bcolors.ENDC))
    return bt_status, ltlf_status


# Function that compares retsults between LTLf and BT
def count_bt_ltlf_return_values(returnvalues):
    count = 0
    for value in returnvalues:
        if (value[0] == common.Status.SUCCESS and value[1] is True):
            count += 1
        elif (value[0] == common.Status.FAILURE and value[1] is False):
            count += 1
    return count
##  Start of BT sub-tree that implement the LTLf operators


# Just a simple condition node that implements atomic propositions
class PropConditionNode(Behaviour):
    """Condition node for the atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the condition node for the atomic LTLf propositions.
    """

    def __init__(self, name):
        """Init method for the condition node."""
        super(PropConditionNode, self).__init__(name)
        self.proposition_symbol = name


    # def setup(self, timeout, value=False):
    def setup(self, timeout, trace, i=0):
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
        # if the proposition value is true
        ## return Success
        # else
        ## return Failure
        # if self.value[self.proposition_symbol]:
        # print(
        #     'proposition index',
        #     self.name, self.index, self.proposition_symbol,
        #     self.trace[self.index][self.proposition_symbol])
        try:
            if self.trace[self.index][self.proposition_symbol]:
                return_value = common.Status.SUCCESS
            else:
                return_value = common.Status.FAILURE
        except IndexError:
            return_value = common.Status.FAILURE

        return return_value


# Just a simple condition node that implements atomic propositions
class Negation(Decorator):
    """Decorator node for the negation of an atomic proposition.

    Inherits the Decorator class from py_trees. This
    behavior implements the negation of an atomic LTLf propositions.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Negation, self).__init__(name=name, child=child)

    def setup(self, timeout, trace, i=0):
        self.decorated.setup(0, trace, i)

    def reset(self, i=0):
        def reset(children, i):
            for child in children:
                try:
                    child.reset(i)
                except AttributeError:
                    reset(child.children, i)
        reset(self.children, i)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the inverted status
        """
        # if the proposition value is true
        ## return Failure
        # else
        ## return Success
        # This give access to the child class of decorator class
        if self.decorated.status == common.Status.RUNNING:
            return common.Status.RUNNING
        elif self.decorated.status == common.Status.FAILURE:
            return common.Status.SUCCESS
        elif self.decorated.status == common.Status.SUCCESS:
            return common.Status.FAILURE


# Just a simple condition node that implements atomic propositions
class And(Decorator):
    """Decorator node for the and of an atomic proposition.

    Inherits the Decorator class from py_trees. This
    behavior implements the negation of an atomic LTLf propositions.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(And, self).__init__(name=name, child=child)

    def reset(self, i=0):
        def reset(children, i):
            for child in children:
                try:
                    child.reset(i)
                except AttributeError:
                    reset(child.children, i)
        reset(self.children, i)

    def increment(self):
        for child in self.children:
            child.increment()

    def setup(self, timeout, trace, i=0):
        self.idx = i
        self.trace = trace
        # Find all the child nodes and call setup
        childs = self.decorated.children
        # print('from And decorator setup', i, childs)
        for child in childs:
            try:
                child.setup(0, trace, i)
            except TypeError:
                for c in child.children:
                    c.setup(0, trace, i)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the inverted status
        """
        # if the proposition value is true
        ## return Failure
        # else
        ## return Success
        # This give access to the child class of decorator class
        # print(self.parent, self.children[0].children, self.name, self.idx, self.decorated.status)
        return self.decorated.status


# Just a simple condition node that implements Next LTLf operator
class Next(Decorator):
    """Decorator node for the Next operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Next LTLf operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Next, self).__init__(name=name, child=child)
        self.idx = 0
        self.next_status = None
        # self.pchilds = pchilds
        # self.trace = trace

    def reset(self, i=0):
        self.next_status = None
        self.idx = i
        def reset(children, i):
            for child in children:
                try:
                    child.reset(i)
                except AttributeError:
                    reset(child.children, i)
        reset(self.children, i)

    def setup(self, timeout, trace, i=0):
        self.idx = i
        self.trace = trace
        self.decorated.setup(0, self.trace, self.idx+1)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Next operator status
        """
        # This give access to the child class of decorator class
        if self.decorated.status == common.Status.RUNNING:
            self.next_status = common.Status.RUNNING
        elif self.decorated.status == common.Status.SUCCESS:
            self.next_status = common.Status.SUCCESS
        elif self.decorated.status == common.Status.FAILURE:
            self.next_status = common.Status.FAILURE

        return self.next_status


# Just a simple decorator node that implements Globally LTLf operator
class Globally(Decorator):
    """Decorator node for the Globally operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Globally LTLf operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Globally, self).__init__(name=name, child=child)
        self.idx = 0

    def reset(self, i=0):
        self.idx = i
        def reset(children, i):
            for child in children:
                try:
                    child.reset(i)
                except AttributeError:
                    reset(child.children, i)
        reset(self.children, i)

    def setup(self, timeout, trace, i=0):
        self.idx = i
        self.trace = trace
        self.decorated.setup(0, self.trace, self.idx)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Globally operator status
        """
        #  Repeat until logic for decorator
        return_value = self.decorated.status
        for j in range(self.idx+1, len(self.trace)):
            if return_value == common.Status.RUNNING:
                return common.Status.RUNNING
            elif return_value == common.Status.SUCCESS:
                self.decorated.setup(0, self.trace, j)
                return_value = list(self.decorated.tick())[-1].update()
            elif return_value == common.Status.FAILURE:
                break

        return return_value


# Just a simple decorator node that implements Finally LTLf operator
class Finally(Decorator):
    """Decorator node for the Finally operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Finally LTLf operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Finally, self).__init__(name=name, child=child)
        self.idx = 0

    def reset(self, i=0):
        self.idx = i
        def reset(children, i):
            for child in children:
                try:
                    child.reset(i)
                except AttributeError:
                    reset(child.children, i)
        reset(self.children, i)

    def setup(self, timeout, trace, i=0):
        self.idx = i
        self.trace = trace
        self.decorated.setup(0, self.trace, self.idx)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Finally operator status
        """
        #  Repeat until logic for decorator
        return_value = self.decorated.status
        for j in range(self.idx+1, len(self.trace)):
            if self.decorated.status == common.Status.RUNNING:
                return common.Status.RUNNING
            elif return_value == common.Status.SUCCESS:
                break
            elif return_value == common.Status.FAILURE:
                self.decorated.setup(0, self.trace, j)
                return_value = list(self.decorated.tick())[-1].update()

        return return_value


# Just a simple decorator node that implements Until LTLf operator
class Until(Decorator):
    """Decorator node for the Until operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Until LTLf operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Until, self).__init__(name=name, child=child)
        self.idx = 0

    def reset(self, i=0):
        self.idx = i
        def reset(children, i):
            for child in children:
                try:
                    child.reset(i)
                except AttributeError:
                    reset(child.children, i)
        reset(self.children, i)

    def setup(self, timeout, trace, i=0):
        self.idx = i
        self.trace = trace
        self.decorated.setup(0, self.trace, self.idx)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Until operator status

        """
        #  Repeat until logic for decorator
        # print('Until', self.idx, self.decorated.status)
        return_value = self.decorated.status
        if self.decorated.status == common.Status.RUNNING:
            return common.Status.RUNNING
        elif return_value == common.Status.SUCCESS:
            return common.Status.SUCCESS
        else:
            for i in range(self.idx+1, len(self.trace)):
                # print('inside Until', self.name, i, self.decorated.name)
                self.decorated.setup(0, self.trace, i)
                return_value = list(self.decorated.tick())[-1].update()
                if return_value == common.Status.SUCCESS:
                    break
        return return_value


# Just a simple decorator node that implements Until LTLf operator
class UntilB(Decorator):
    """Decorator node for the Until operator for the left sub-tree.

    Inherits the Decorator class from py_trees. This
    behavior implements the Until LTLf operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(UntilB, self).__init__(name=name, child=child)
        self.idx = 0
        self.j = -1

    def reset(self, i=0):
        self.idx = i
        self.j = i-1
        # print('resset from until b', self.name, i)
        def reset(children, i):
            for child in children:
                try:
                    child.reset(i)
                except AttributeError:
                    reset(child.children, i)
        reset(self.children, i)

    def setup(self, timeout, trace, i=0):
        self.idx = i
        self.trace = trace
        self.decorated.setup(0, self.trace, self.idx)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Until operator status

        """
        #  Repeat until logic for decorator
        # print('until b', self.name, self.idx, self.decorated.status)
        return_value = self.decorated.status
        self.j += 1
        return return_value


# Just a simple decorator node that implements Until LTLf operator
class UntilA(Decorator):
    """Decorator node for the Until operator for the right sub-tree.

    Inherits the Decorator class from py_trees. This
    behavior implements the Until LTLf operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(UntilA, self).__init__(name=name, child=child)
        self.idx = 0

    def reset(self, i=0):
        self.idx = i
        def reset(children, i):
            for child in children:
                try:
                    child.reset(i)
                except AttributeError:
                    reset(child.children, i)
        reset(self.children, i)

    def setup(self, timeout, trace, i=0):
        self.idx = i
        self.trace = trace
        self.decorated.setup(0, self.trace, self.idx)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Until operator status

        """
        #  Repeat until logic for decorator
        # return_value = self.decorated.status
        # self.j += 1

        j = self.parent.children[0].j
        # print('print until a', self.name, j)
        if (j == self.parent.parent.parent.idx and self.parent.children[0].status == common.Status.SUCCESS):
            return_value = common.Status.SUCCESS
        elif (j > self.parent.parent.parent.idx and self.parent.children[0].status == common.Status.SUCCESS):
            # print(self.parent.parent.parent)
            for k in range(self.parent.parent.parent.idx, j):
                try:
                    self.decorated.reset(k)
                    self.decorated.setup(0, self.trace, k)
                except AttributeError:
                    # [c.reset(k) for c in self.decorated.children]
                    # [c.setup(0, self.trace[:j], k) for c in self.decorated.children]
                    pass
                    # self.decorated.setup(0, self.trace, k)
                return_value = list(self.decorated.tick())[-1].update()
                if return_value == common.Status.FAILURE:
                    break
        else:
            return_value = common.Status.FAILURE
        return return_value


## Experiments to test each LTLf operator and its BT sub-tree

def expriments(traces, btroot, cnodes, formula, args, i=0, verbos=True):
    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine.
    for trace in traces:
        i = 0 if args.trace =='fixed' else np.random.randint(0, len(trace))
        # i = 0
        if verbos:
            print('--------------')
            print('Experiment no: ', expno, ',i=',i)
        # Call the excute function that will execute both BT and LTLf
        # i = 0 if args.trace =='fixed' else np.random.randint(0, len(trace))
        returnvalueslist.append(execute_both_bt_ltlf(btroot, formula, trace, cnodes, i=i, verbos=True))
        if verbos:
            print('=============')
        expno += 1
    count = count_bt_ltlf_return_values(returnvalueslist)
    print("Total Experiment Runs: {}, BT and LTLf agree: {}".format(expno, count))


# Experiment 1 for simple atomic propositions
def proposition2condition(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 1
        trace1 = [
            {'a': False}
        ]
        # Trace of length 1
        trace2 = [
            {'a': True}
        ]
        # Trace of length 3
        trace3 = [
            {'a': True},
            {'a': False},
            {'a': True}
        ]
        traces =[trace1, trace2, trace3]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    cnode = PropConditionNode('a')
    expriments(traces, cnode, [cnode], 'a', args)


# Experiment 2 for simple negation of atomic propositions
def negation2decorator(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 1
        trace1 = [
            {'a': False}
        ]
        # Trace of length 1
        trace2 = [
            {'a': True}
        ]
        # Trace of length 3
        trace3 = [
            {'a': True},
            {'a': False},
            {'a': True}
        ]
        traces =[trace1, trace2, trace3]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    # Experiment variables
    cnode = PropConditionNode('a')
    ndecorator = Negation(cnode, 'Invert')
    expriments(traces, ndecorator, [ndecorator], '!a', args)


# Experiment 3 for simple and of atomic propositions
def and2sequence(args, verbos=True):
    traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)
    cnode1 = PropConditionNode('c')
    cnode2 = PropConditionNode('d')
    parll = Parallel('And')
    parll.add_children([cnode1, cnode2])
    anddec = And(parll)
    expriments(traces, anddec, [anddec], 'c & d', args)


# Experiment 4 for simple X
def next2decorator(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 1
        trace1 = [
            {'a': True, 'c': True, 'd': True}
        ]
        # Trace of length 1
        trace2 = [
            {'a': False, 'c': False, 'd': False}
        ]
        # Trace of length 2
        trace3 = [
            {'a': False, 'c': False, 'd': True},
            {'a': False, 'c': True, 'd': False}
        ]
        # Trace of length 3
        trace4 = [
            {'a': False, 'c': True, 'd': False},
            {'a': False, 'c': False, 'd': False},
            {'a': False, 'c': True, 'd': False}
        ]
        traces =[trace1, trace2, trace3, trace4]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)
    # traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)
    # traces = [trace1, trace2, trace3, trace4]
    cnode1 = PropConditionNode('c')
    # cnode2 = PropConditionNode('d')
    # parll = Parallel('And')
    # parll.add_children([cnode1, cnode2])
    # nextd = Next(parll)
    nextd = Next(cnode1)
    expriments(traces, nextd, [nextd], '(X c)', args)
    # expriments(traces, nextd, [cnode1, nextd], '(X (c & d))', args)


# Experiment 5 for simple X and &
def composite1_next_and(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 3
        trace1 = [
            {'c': False, 'd': False},
            {'c': True, 'd': True},
            {'c': True, 'd': True}
        ]

        trace2 = [
            {'c': False, 'd': False},
            {'c': False, 'd': True},
            {'c': True, 'd': False}
        ]

        traces =[trace1, trace2]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)
    cnode1 = PropConditionNode('c')
    cnode2 = PropConditionNode('d')
    parll = Parallel('And')
    parll.add_children([cnode1, cnode2])
    anddec = And(parll)
    nextd = Next(anddec)
    expriments(traces, nextd, [nextd], '(X (c & d))', args)


# Experiment 6 for simple X and &
def composite2_next_and(args, verbos=True):
    traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)
    cnode1 = PropConditionNode('c')
    nextd = Next(cnode1)
    cnode2 = PropConditionNode('d')
    parll = Parallel('And')
    parll.add_children([nextd, cnode2])
    anddec = And(parll)

    expriments(traces, anddec, [nextd, anddec], '((X c) & d)', args)


# Experiment 7 for composite X
def composite3_next_next(args, verbos=True):
    traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)
    cnode1 = PropConditionNode('c')
    nextd1 = Next(cnode1)
    nextd2 = Next(nextd1)

    expriments(traces, nextd2, [nextd2], '(X (X c))', args)


# Experiment 8 for composite X and and
def composite4_next_and(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 3
        trace1 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace2 = [
            {'a': False, 'b': False, 'c': False, 'd': False},
            {'a': True, 'b': False, 'c': False, 'd': True},
            {'a': False, 'b': True, 'c': True, 'd': False}
        ]

        traces =[trace1, trace2]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)
    cnode1 = PropConditionNode('a')
    cnode2 = PropConditionNode('b')
    cnode3 = PropConditionNode('c')
    cnode4 = PropConditionNode('d')

    nextc = Next(cnode3)
    nextb = Next(cnode2)
    parll1 = Parallel('And1')
    parll1.add_children([nextc, cnode4])
    and1 = And(parll1)
    parll2 = Parallel('And2')
    parll2.add_children([cnode1, nextb])
    and2 = And(parll2)
    mainand = Parallel('And3')
    mainand.add_children([and1, and2])
    andmain = And(mainand)
    finalnext = Next(andmain)

    expriments(traces, finalnext, [nextc, nextb, finalnext], 'X (((X c) & d) & (a & (X b)))', args)


# Experiment 9 for simple globally operator
def globally2decorator(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 1
        trace1 = [
            {'a': False}
        ]
        # Trace of length 1
        trace2 = [
            {'a': True}
        ]
        # Trace of length 2
        trace3 = [
            {'a': True},
            {'a': True}
        ]
        # Trace of length 3
        trace4 = [
            {'a': True},
            {'a': False},
            {'a': True}
        ]
        # Trace of length 4
        trace5 = [
            {'a': True},
            {'a': True},
            {'a': True},
            {'a': True}
        ]
        # Trace of length 5
        trace6 = [
            {'a': True},
            {'a': True},
            {'a': True},
            {'a': False},
            {'a': True}
        ]
        # Trace of length 5
        trace7 = [
            {'a': True},
            {'a': True},
            {'a': True},
            {'a': True},
            {'a': False}
        ]
        # Trace of length 5
        trace8 = [
            {'a': False},
            {'a': True},
            {'a': True},
            {'a': True},
            {'a': True}
        ]
        traces =[trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    # Experiment variables
    cnode = PropConditionNode('a')
    # ndecorator = Negation(cnode, 'Invert')
    gdecorator = Globally(cnode, 'Globally')
    expriments(traces, gdecorator, [gdecorator], 'G a', args)


# Experiment 10 for simple globally operator
def composite1_globally_and(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 3
        trace1 = [
            {'c': True, 'd': True},
            {'c': True, 'd': False},
            {'c': True, 'd': True}
        ]

        trace2 = [
            {'c': True, 'd': True},
            {'c': True, 'd': False},
            {'c': False, 'd': True}
        ]

        trace3 = [
            {'c': True, 'd': True},
            {'c': True, 'd': True},
            {'c': True, 'd': True}
        ]

        trace4 = [
            {'c': True, 'd': True},
            {'c': False, 'd': False},
            {'c': True, 'd': True}
        ]

        traces =[trace1, trace2, trace3, trace4]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)
    cnode1 = PropConditionNode('c')
    cnode2 = PropConditionNode('d')
    parll = Parallel('And')
    parll.add_children([cnode1, cnode2])
    anddec = And(parll)
    globallyd = Globally(anddec)
    expriments(traces, globallyd, [globallyd], '(G (c & d))', args)


# Experiment 11 for simple globally operator
def composite2_globally_and_next(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 3
        trace1 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace2 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': False},
            {'a': True, 'b': True, 'c': False, 'd': True}
        ]

        trace3 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace4 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]
        traces =[trace1, trace2, trace3, trace4]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    cnode1 = PropConditionNode('c')
    cnode2 = PropConditionNode('d')
    parll1 = Parallel('And1')
    parll1.add_children([cnode1, cnode2])
    anddec1 = And(parll1)
    globallyd = Globally(anddec1)

    cnode3 = PropConditionNode('a')
    cnode4 = PropConditionNode('b')
    parll2 = Parallel('And2')
    parll2.add_children([cnode3, cnode4])
    anddec2 = And(parll2)
    globallya = Globally(anddec2)

    parll3 = Parallel('And3')
    parll3.add_children([globallyd, globallya])
    anddec3 = And(parll3)
    expriments(traces, anddec3, [anddec3], '(G (c & d)) & (G (a & b))', args)



# Experiment 12 for simple finally operator
def finally2decorator(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 1
        trace1 = [
            {'a': False}
        ]
        # Trace of length 1
        trace2 = [
            {'a': True}
        ]
        # Trace of length 2
        trace3 = [
            {'a': False},
            {'a': True}
        ]
        # Trace of length 3
        trace4 = [
            {'a': False},
            {'a': False},
            {'a': True}
        ]
        # Trace of length 4
        trace5 = [
            {'a': False},
            {'a': True},
            {'a': False},
            {'a': False}
        ]
        # Trace of length 5
        trace6 = [
            {'a': False},
            {'a': False},
            {'a': False},
            {'a': False},
            {'a': False}
        ]
        # Trace of length 5
        trace7 = [
            {'a': False},
            {'a': False},
            {'a': False},
            {'a': False},
            {'a': True}
        ]
        # Trace of length 5
        trace8 = [
            {'a': False},
            {'a': True},
            {'a': True},
            {'a': True},
            {'a': True}
        ]
        traces =[trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    # Experiment variables
    cnode = PropConditionNode('a')
    # ndecorator = Negation(cnode, 'Invert')
    fdecorator = Finally(cnode, 'Finally')
    expriments(traces, fdecorator, [fdecorator], 'F a', args)


# Experiment 13 for simple finally operator
def composite1_finally_and(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 1
        trace1 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace2 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': False},
            {'a': True, 'b': True, 'c': False, 'd': True}
        ]

        trace3 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace4 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]
        traces =[trace1, trace2, trace3, trace4]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    # Experiment variables
    cnode1 = PropConditionNode('a')
    cnode2 = PropConditionNode('b')
    parll2 = Parallel('And')
    parll2.add_children([cnode1, cnode2])
    anddec2 = And(parll2)

    fdecorator = Finally(anddec2, 'Finally')
    expriments(traces, fdecorator, [fdecorator], 'F (a & b)', args)



# Experiment 14 for simple globally operator
def composite2_finally_and_next(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 3
        trace1 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace2 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': False},
            {'a': True, 'b': True, 'c': False, 'd': True}
        ]

        trace3 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace4 = [
            {'a': True, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]
        traces = [trace1, trace2, trace3, trace4]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    cnode1 = PropConditionNode('c')
    cnode2 = PropConditionNode('d')
    parll1 = Parallel('And1')
    parll1.add_children([cnode1, cnode2])
    anddec1 = And(parll1)
    globallyd = Finally(anddec1)

    cnode3 = PropConditionNode('a')
    cnode4 = PropConditionNode('b')
    parll2 = Parallel('And2')
    parll2.add_children([cnode3, cnode4])
    anddec2 = And(parll2)
    globallya = Finally(anddec2)

    parll3 = Parallel('And3')
    parll3.add_children([globallyd, globallya])
    anddec3 = And(parll3)
    expriments(traces, anddec3, [anddec3], '(F (c & d)) & (F (a & b))', args)


# Experiment 15 for simple finally operator
def until2subtree(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 1
        t1 = [{'a': True, 'b': True}]
        t2 = [{'a': True, 'b': False}]
        t3 = [{'a': False, 'b': True}]
        t4 = [{'a': False, 'b': False}]
        t0 = [
            {'a': True, 'b': False},
            {'a': True, 'b': True},
            {'a': False, 'b': False}
            ]

        t5 = [
             [t1[0], t1[0]],
             [t1[0], t2[0]],
             [t1[0], t3[0]],
             [t1[0], t4[0]],
             [t2[0], t1[0]],
             [t2[0], t2[0]],
             [t2[0], t3[0]],
             [t2[0], t4[0]],
             [t3[0], t1[0]],
             [t3[0], t2[0]],
             [t3[0], t3[0]],
             [t3[0], t4[0]],
             [t4[0], t1[0]],
             [t4[0], t2[0]],
             [t4[0], t3[0]],
             [t4[0], t4[0]]
             ]

        t6 = [
            [t1[0], t1[0], t1[0]],
            [t1[0], t1[0], t2[0]],
            [t1[0], t1[0], t3[0]],
            [t1[0], t1[0], t4[0]],

            [t1[0], t2[0], t1[0]],
            [t1[0], t2[0], t2[0]],
            [t1[0], t2[0], t3[0]],
            [t1[0], t2[0], t4[0]],


            [t1[0], t3[0], t1[0]],
            [t1[0], t3[0], t2[0]],
            [t1[0], t3[0], t3[0]],
            [t1[0], t3[0], t4[0]],


            [t1[0], t4[0], t1[0]],
            [t1[0], t4[0], t2[0]],
            [t1[0], t4[0], t3[0]],
            [t1[0], t4[0], t4[0]]
        ]

        t7 = [
            [t2[0], t2[0], t1[0]],
            [t2[0], t2[0], t2[0]],
            [t2[0], t2[0], t3[0]],
            [t2[0], t2[0], t4[0]],

            [t2[0], t3[0], t1[0]],
            [t2[0], t3[0], t2[0]],
            [t2[0], t3[0], t3[0]],
            [t2[0], t3[0], t4[0]],


            [t2[0], t4[0], t1[0]],
            [t2[0], t4[0], t2[0]],
            [t2[0], t4[0], t3[0]],
            [t2[0], t4[0], t4[0]],


            [t2[0], t1[0], t1[0]],
            [t2[0], t1[0], t2[0]],
            [t2[0], t1[0], t3[0]],
            [t2[0], t1[0], t4[0]]
        ]

        t8 = [
            [t3[0], t3[0], t1[0]],
            [t3[0], t3[0], t2[0]],
            [t3[0], t3[0], t3[0]],
            [t3[0], t3[0], t4[0]],

            [t3[0], t4[0], t1[0]],
            [t3[0], t4[0], t2[0]],
            [t3[0], t4[0], t3[0]],
            [t3[0], t4[0], t4[0]],


            [t3[0], t1[0], t1[0]],
            [t3[0], t1[0], t2[0]],
            [t3[0], t1[0], t3[0]],
            [t3[0], t1[0], t4[0]],


            [t3[0], t2[0], t1[0]],
            [t3[0], t2[0], t2[0]],
            [t3[0], t2[0], t3[0]],
            [t3[0], t2[0], t4[0]]
        ]

        t9 = [
            [t4[0], t4[0], t1[0]],
            [t4[0], t4[0], t2[0]],
            [t4[0], t4[0], t3[0]],
            [t4[0], t4[0], t4[0]],

            [t4[0], t1[0], t1[0]],
            [t4[0], t1[0], t2[0]],
            [t4[0], t1[0], t3[0]],
            [t4[0], t1[0], t4[0]],


            [t4[0], t2[0], t1[0]],
            [t4[0], t2[0], t2[0]],
            [t4[0], t2[0], t3[0]],
            [t4[0], t2[0], t4[0]],


            [t4[0], t3[0], t1[0]],
            [t4[0], t3[0], t2[0]],
            [t4[0], t3[0], t3[0]],
            [t4[0], t3[0], t4[0]]
        ]

        traces =[t1, t2, t3, t4, t0] + t5 + t6 + t7 + t8 + t9
        # traces = t6 + t7 + t8 + t9
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    # Experiment variables
    cnode1 = PropConditionNode('a')
    cnode2 = PropConditionNode('b')
    parll2 = Sequence('Seq')
    untila = UntilA(cnode1)
    untilb = UntilB(cnode2)
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until = Until(anddec2)
    expriments(traces, until, [until], '(a U b)', args)


def composite1_until_until(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 3
        trace1 = [
            {'a': True, 'b': False, 'c': False, 'd': True},
            {'a': True, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace2 = [
            {'a': False, 'b': False, 'c': False, 'd': True},
            {'a': True, 'b': False, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': False, 'd': True}
        ]

        trace3 = [
            {'a': False, 'b': False, 'c': False, 'd': True},
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': False, 'c': True, 'd': True}
        ]

        trace4 = [
            {'a': True, 'b': False, 'c': False, 'd': True},
            {'a': True, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace5 = [
            {'a': False, 'b': True, 'c': False, 'd': True},
            {'a': False, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': False, 'c': True, 'd': True}
        ]
        traces = [trace1, trace2, trace3, trace4, trace5]
        # traces = [trace5]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    # Experiment variables
    cnode1 = PropConditionNode('a')
    cnode2 = PropConditionNode('b')

    parll2 = Sequence('Seq')
    untila = UntilA(cnode1, name='A')
    untilb = UntilB(cnode2, name='B')
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until1 = Until(anddec2, 'U1')


    cnode3 = PropConditionNode('c')
    untilb1 = UntilB(cnode3, 'C')
    untila1 = UntilA(until1, 'AUB')
    parll3 = Sequence('Seq1')
    parll3.add_children([untilb1, untila1])
    anddec3 = And(parll3)
    until2 = Until(anddec3, 'U')

    expriments(traces, until2, [until2], '((a U b) U c)', args)



def composite1_until_next_and(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 3
        trace1 = [
            {'a': True, 'b': False, 'c': False, 'd': True},
            {'a': True, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace2 = [
            {'a': False, 'b': False, 'c': False, 'd': True},
            {'a': True, 'b': False, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': False, 'd': True}
        ]

        trace3 = [
            {'a': False, 'b': False, 'c': False, 'd': True},
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': False, 'c': True, 'd': True}
        ]

        trace4 = [
            {'a': True, 'b': False, 'c': False, 'd': True},
            {'a': True, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace5 = [
            {'a': False, 'b': True, 'c': False, 'd': True},
            {'a': False, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': False, 'c': True, 'd': True}
        ]
        traces = [trace1, trace2, trace3, trace4, trace5]
        # traces = [trace5]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    # Experiment variables
    cnode1 = PropConditionNode('a')
    cnode2 = PropConditionNode('b')
    cnode3 = PropConditionNode('c')
    cnode4 = PropConditionNode('d')

    nextd = Next(cnode3)
    parll = Parallel('And1')
    parll.add_children([nextd, cnode4])
    aand = And(parll)

    parll1 = Parallel('And2')
    parll1.add_children([cnode1, cnode2])
    band = And(parll1)


    parll2 = Sequence('Seq')
    untila = UntilA(aand, name='A')
    untilb = UntilB(band, name='B')
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until1 = Until(anddec2, 'U1')

    expriments(traces, until1, [until1], '(((X c) & d) U (a & b))', args)



def composite1_until_next_and1(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 3
        trace1 = [
            {'a': True, 'b': False, 'c': False, 'd': True},
            {'a': True, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace2 = [
            {'a': False, 'b': False, 'c': False, 'd': True},
            {'a': True, 'b': False, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': False, 'd': True}
        ]

        trace3 = [
            {'a': False, 'b': False, 'c': False, 'd': True},
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': False, 'c': True, 'd': True}
        ]

        trace4 = [
            {'a': True, 'b': False, 'c': False, 'd': True},
            {'a': True, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': True}
        ]

        trace5 = [
            {'a': False, 'b': True, 'c': False, 'd': True},
            {'a': False, 'b': True, 'c': False, 'd': False},
            {'a': True, 'b': False, 'c': True, 'd': True}
        ]
        traces = [trace1, trace2, trace3, trace4, trace5]
        # traces = [trace5]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    # Experiment variables
    cnode1 = PropConditionNode('a')
    cnode2 = PropConditionNode('b')
    cnode3 = PropConditionNode('c')
    cnode4 = PropConditionNode('d')

    nextd = Next(cnode3)
    parll = Parallel('And1')
    parll.add_children([nextd, cnode4])
    aand = And(parll)

    nexta = Next(cnode1)
    parll1 = Parallel('And2')
    parll1.add_children([nexta, cnode2])
    band = And(parll1)


    parll2 = Sequence('Seq')
    untila = UntilA(aand, name='A')
    untilb = UntilB(band, name='B')
    parll2.add_children([untilb, untila])
    anddec2 = And(parll2)
    until1 = Until(anddec2, 'U1')

    expriments(traces, until1, [until1], '(((X c) & d) U ((X a) & b))', args)


def main(args):
    if args.test == 'P':
        proposition2condition(args)
    elif args.test == '~':
        negation2decorator(args)
    elif args.test == '&':
        and2sequence(args)
    elif args.test == 'X':
        next2decorator(args)
    elif args.test == 'U':
        until2subtree(args)
    elif args.test == 'G':
        globally2decorator(args)
    elif args.test == 'F':
        finally2decorator(args)
    elif args.test == 'C1_X_&':
        composite1_next_and(args)
    elif args.test == 'C2_X_&':
        composite2_next_and(args)
    elif args.test == 'C3_X_X':
        composite3_next_next(args)
    elif args.test == 'C4_X_&':
        composite4_next_and(args)
    elif args.test == 'C1_G_&':
        composite1_globally_and(args)
    elif args.test == 'C2_G_&_X':
        composite2_globally_and_next(args)
    elif args.test == 'C1_F_&':
        composite1_finally_and(args)
    elif args.test == 'C2_F_&_X':
        composite2_finally_and_next(args)
    elif args.test == 'C1_U_U':
        composite1_until_until(args)
    elif args.test == 'C1_U_X_&':
        composite1_until_next_and(args)
    elif args.test == 'C1_U_X_&_X':
        composite1_until_next_and1(args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test', type=str, choices = [
            'P', '~', '&', 'X', 'G', 'F', 'U',
            'C1_X_&', 'C2_X_&', 'C3_X_X', 'C4_X_&',
            'C1_G_&', 'C2_G_&_X', 'C1_F_&', 'C2_F_&_X', 'C1_U_U', 'C1_U_X_&', 'C1_U_X_&_X'
            ])
    parser.add_argument('--trace', type=str, choices = ['fixed', 'random'], default='fixed')
    parser.add_argument('--max_trace_len', type=int, default=3)
    parser.add_argument('--no_trace_2_test', type=int, default=16)

    ## Usages

    # Atomic proposition
    ## python ltl2bt.py --test 'P'

    # Negation
    ## python ltl2bt.py --test '~'

    # And
    ## python ltl2bt.py --test '&'

    # Next
    ## python ltl2bt.py --test 'X'

    # Globally
    ## python ltl2bt.py --test 'G'

    # Finally
    ## python ltl2bt.py --test 'F'

    # Until sub-tree
    # to run until expeirments with pre-defined traces:
    ## python ltl2bt.py --test 'U'

    # to run until expeirments with random length trace:
    ## python ltl2bt.py --test 'U' --trace 'random' --max_trace_len 0

    # to run until expeirments with 10 length trace:
    ## python ltl2bt.py --test 'U' --trace 'random' --max_trace_len 10

    # to run until expeirments with 10 length trace and 50 different traces:
    ## python ltl2bt.py --test 'U' --trace 'random' --max_trace_len 10 --no_trace_2_test 50

    ### These aruguments (random, max_trace_len, no_trace_2_test) is equally
    ### valid for (Next, Globally, Finally) operator as well
    args = parser.parse_args()


    main(args)