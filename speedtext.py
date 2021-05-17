from ltl2btrevised import (
    Globally, Finally, PropConditionNode, getrandomtrace)

from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour

from flloat.parser.ltlf import LTLfParser
import numpy as np
import time


def expriments(traces, btroot, cnodes, formula, i=0, verbos=True):
    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine.
    for trace in traces:
        i = 0
        execute_bt(btroot, trace, cnodes, i)
        execute_ltlf(formula, trace, i)


# This supplies trace at time i for the nodes
def setup_node(nodes, trace, k):
    for node in nodes:
        node.setup(0, trace, k)


# Method executes a BT passed as an argument
def execute_bt(subtree, trace, nodes, i=0):
    # Args: bt -> BT to tick
    #       trace -> a trace of lenght m
    #       nodes -> Execution nodes of BT that takes trace as input

    # reset with the i value provided so it is consistent
    t1 = time.perf_counter()
    bt = BehaviourTree(subtree)
    # bt.root.reset(i)
    # setup_node(nodes, trace[k])
    setup_node(nodes, trace, i)
    bt.tick()
    t2 = time.perf_counter()
    print('BT', round((t2-t1), 4))
    return bt.root.status


# Execute both BT and Ltlf with same traces for comparision
def execute_ltlf(formula, trace, i=0, verbos=True):
    # Creating a LTLf parser object
    t1 = time.perf_counter()
    parser = LTLfParser()
    # Parsed formula
    parsed_formula = parser(formula)
    ltlf_status = parsed_formula.truth(trace, i)
    t2 = time.perf_counter()
    print('LTLf', round((t2-t1), 4))
    return ltlf_status


# Experiment 10 for simple globally operator
def composite1_globally(verbos=True):
    for i in range(2, 10000, 1000):
        traces = getrandomtrace(n=1, maxtracelen=i)
        cnode1 = PropConditionNode('c')
        cnode2 = PropConditionNode('d')
        final = Finally(cnode1)
        globallyd = Globally(final)
        print('trace length', i)
        expriments(traces, globallyd, [globallyd], '(G (F d))')


composite1_globally()