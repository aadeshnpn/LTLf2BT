from flloat.ltlf import LTLfAlways, LTLfAnd, LTLfEventually, LTLfNext, LTLfNot, LTLfOr, LTLfUntil
from flloat.parser.ltlf import LTLfParser, LTLfAtomic

from py_trees.composites import Sequence, Selector, Parallel
from py_trees.decorators import Decorator, Inverter
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees import common, blackboard
from py_trees.decorators import Retry
import py_trees
import copy
import argparse
import numpy as np
import time
from mpire import WorkerPool
import pandas as pd
from ppatask import (
    create_action_GBT, ActionNode)


# Just a simple decorator node that implements Finally mission operator
class Finally(Decorator):
    """Decorator node for the Finally operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Finally LTLf operator.
    """
    def __init__(self, child, name='Finally', task_max=4):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Finally, self).__init__(name=name, child=child)
        self.idx = 0
        self.memory = common.Status.SUCCESS
        self.task_max = task_max
        self.child = [copy.copy(child)]* (task_max +2)

    def reset(self, i=0):
        self.memory = common.Status.SUCCESS

    def setup(self, timeout, i=0):
        self.decorated.setup(0, self.idx)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Globally operator status
        """
        #  Repeat until logic for decorator
        return_value = self.decorated.status
        # print(return_value)
        if return_value == common.Status.RUNNING:
            return common.Status.RUNNING
        elif return_value == common.Status.FAILURE:
            self.idx += 1
            # print(self.idx, self.task_max, return_value)
            self.decorated = self.child[self.idx]
            childrens = list(self.decorated.iterate())
            action_node = [child for child in childrens if isinstance(child, ActionNode)]
            action_node[0].reset()
            if self.idx > self.task_max:
                return common.Status.FAILURE
            return common.Status.RUNNING
        return self.memory


def parse_ltlf(formula, mappings, task_max=4):
    # Just proposition
    if isinstance(formula, LTLfAtomic):
        # map to Dummy PPATaskNode
        return mappings[formula.s]
    # Unary or binary operator
    else:
        if (
            isinstance(formula, LTLfEventually) or isinstance(formula, LTLfAlways)
                or isinstance(formula, LTLfNext) or isinstance(formula, LTLfNot) ):
            if isinstance(formula, LTLfEventually):
                return Finally(parse_ltlf(formula.f, mappings), task_max=task_max)
                # return Retry(name='Retry', child=parse_ltlf(formula.f, mappings), num_failures=task_max)

        elif (isinstance(formula, LTLfAnd) or isinstance(formula, LTLfOr) or isinstance(formula, LTLfUntil)):
            if isinstance(formula, LTLfAnd):
                leftformual, rightformula = formula.formulas
                parll = Parallel(
                    'And',
                    policy=common.ParallelPolicy.SuccessOnAll(synchronise=False))
                leftnode = parse_ltlf(leftformual, mappings, task_max=task_max)
                rightnode = parse_ltlf(rightformula, mappings, task_max=task_max)
                parll.add_children([leftnode, rightnode])
                return parll
            elif isinstance(formula, LTLfOr):
                leftformual, rightformula = formula.formulas
                ornode = Selector('Or', memory=False)
                # ornode = Parallel('Or', policy=common.ParallelPolicy.SuccessOnOne())
                leftnode = parse_ltlf(leftformual, mappings, task_max=task_max)
                rightnode = parse_ltlf(rightformula, mappings, task_max=task_max)
                ornode.add_children([leftnode, rightnode])
                # ordecorator = Or(ornode)
                return ornode

            elif isinstance(formula, LTLfUntil):
                leftformual, rightformula = formula.formulas
                leftnode = parse_ltlf(leftformual, mappings, task_max=task_max)
                rightnode = parse_ltlf(rightformula, mappings, task_max=task_max)
                useq = Sequence('UntilSeq', memory=False)
                # untila = Until(leftnode, name='Until')
                # untilb = Reset(rightnode, name='Reset', tmax=task_max)
                useq.add_children([leftnode, rightnode])
                return useq


class Environment():
    def __init__(self, seed=None) -> None:
        self.state_one =[
            {'poc1': False, 'prc1': False, 'gc1': False, 'tc1': False},
            {'poc1': False, 'prc1': False, 'gc1': True,  'tc1': False},
            {'poc1': False, 'prc1': False, 'gc1': False, 'tc1': True},
            {'poc1': False, 'prc1': False, 'gc1': True,  'tc1': True},
            {'poc1': True,  'prc1': False, 'gc1': False, 'tc1': False},
            {'poc1': True,  'prc1': False, 'gc1': True,  'tc1': False},
            {'poc1': True,  'prc1': False, 'gc1': False, 'tc1': True},
            {'poc1': True,  'prc1': False, 'gc1': True,  'tc1': True},
            {'poc1': False, 'prc1': True,  'gc1': False, 'tc1': False},
            {'poc1': False, 'prc1': True,  'gc1': True,  'tc1': False},
            {'poc1': False, 'prc1': True,  'gc1': False, 'tc1': True},
            {'poc1': False, 'prc1': True,  'gc1': True,  'tc1': True},
            {'poc1': True,  'prc1': True,  'gc1': False, 'tc1': False},
            {'poc1': True,  'prc1': True,  'gc1': True,  'tc1': False},
            {'poc1': True,  'prc1': True,  'gc1': False, 'tc1': True},
            {'poc1': True,  'prc1': True,  'gc1': True,  'tc1': True},
        ]
        self.state_two =[
            {'poc2': False, 'prc2': False, 'gc2': False, 'tc2': False},
            {'poc2': False, 'prc2': False, 'gc2': True,  'tc2': False},
            {'poc2': False, 'prc2': False, 'gc2': False, 'tc2': True},
            {'poc2': False, 'prc2': False, 'gc2': True,  'tc2': True},
            {'poc2': True,  'prc2': False, 'gc2': False, 'tc2': False},
            {'poc2': True,  'prc2': False, 'gc2': True,  'tc2': False},
            {'poc2': True,  'prc2': False, 'gc2': False, 'tc2': True},
            {'poc2': True,  'prc2': False, 'gc2': True,  'tc2': True},
            {'poc2': False, 'prc2': True,  'gc2': False, 'tc2': False},
            {'poc2': False, 'prc2': True,  'gc2': True,  'tc2': False},
            {'poc2': False, 'prc2': True,  'gc2': False, 'tc2': True},
            {'poc2': False, 'prc2': True,  'gc2': True,  'tc2': True},
            {'poc2': True,  'prc2': True,  'gc2': False, 'tc2': False},
            {'poc2': True,  'prc2': True,  'gc2': True,  'tc2': False},
            {'poc2': True,  'prc2': True,  'gc2': False, 'tc2': True},
            {'poc2': True,  'prc2': True,  'gc2': True,  'tc2': True},
        ]
        if seed is not None:
            seed = time.time_ns() % 39916801
            self.random = np.random.RandomState(seed)
        else:
            self.random = np.random.RandomState()
        # self.curr_state = {
        #     **self.random.choice(self.state_one),
        #     **self.random.choice(self.state_two)
        #     }
        self.curr_state = {
            'poc1': False,  'prc1': True,  'gc1': True,  'tc1': True,
            'poc2': False,  'prc2': True,  'gc2': True,  'tc2': True
            }

    def step(self):
        self.curr_state = {
            **self.random.choice(self.state_one),
            **self.random.choice(self.state_two)
            }

    def reset(self):
        self.curr_state = {
            'poc1': False,  'prc1': True,  'gc1': True,  'tc1': True,
            'poc2': False,  'prc2': True,  'gc2': True,  'tc2': True
            }

class Environment1():
    def __init__(self, seed=None) -> None:
        self.state_one =[
            {'poc1': False, 'prc1': False, 'gc1': False, 'tc1': False},
            {'poc1': False, 'prc1': False, 'gc1': True,  'tc1': False},
            {'poc1': False, 'prc1': False, 'gc1': False, 'tc1': True},
            {'poc1': False, 'prc1': False, 'gc1': True,  'tc1': True},
            {'poc1': True,  'prc1': False, 'gc1': False, 'tc1': False},
            {'poc1': True,  'prc1': False, 'gc1': True,  'tc1': False},
            {'poc1': True,  'prc1': False, 'gc1': False, 'tc1': True},
            {'poc1': True,  'prc1': False, 'gc1': True,  'tc1': True},
            {'poc1': False, 'prc1': True,  'gc1': False, 'tc1': False},
            {'poc1': False, 'prc1': True,  'gc1': True,  'tc1': False},
            {'poc1': False, 'prc1': True,  'gc1': False, 'tc1': True},
            {'poc1': False, 'prc1': True,  'gc1': True,  'tc1': True},
            {'poc1': True,  'prc1': True,  'gc1': False, 'tc1': False},
            {'poc1': True,  'prc1': True,  'gc1': True,  'tc1': False},
            {'poc1': True,  'prc1': True,  'gc1': False, 'tc1': True},
            {'poc1': True,  'prc1': True,  'gc1': True,  'tc1': True},
        ]
        if seed is not None:
            seed = time.time_ns() % 39916801
            self.random = np.random.RandomState(seed)
        else:
            self.random = np.random.RandomState()
        self.curr_state = {
            'poc1': False,  'prc1': True,  'gc1': True,  'tc1': True
            }

    def step(self):
        self.curr_state = self.random.choice(self.state_one)

    def reset(self):
        self.curr_state = {
            'poc1': False,  'prc1': True,  'gc1': True,  'tc1': True
            }


class Environment2():
    def __init__(self, seed=None) -> None:
        self.state_two =[
            {'poc2': False, 'prc2': False, 'gc2': False, 'tc2': False},
            {'poc2': False, 'prc2': False, 'gc2': True,  'tc2': False},
            {'poc2': False, 'prc2': False, 'gc2': False, 'tc2': True},
            {'poc2': False, 'prc2': False, 'gc2': True,  'tc2': True},
            {'poc2': True,  'prc2': False, 'gc2': False, 'tc2': False},
            {'poc2': True,  'prc2': False, 'gc2': True,  'tc2': False},
            {'poc2': True,  'prc2': False, 'gc2': False, 'tc2': True},
            {'poc2': True,  'prc2': False, 'gc2': True,  'tc2': True},
            {'poc2': False, 'prc2': True,  'gc2': False, 'tc2': False},
            {'poc2': False, 'prc2': True,  'gc2': True,  'tc2': False},
            {'poc2': False, 'prc2': True,  'gc2': False, 'tc2': True},
            {'poc2': False, 'prc2': True,  'gc2': True,  'tc2': True},
            {'poc2': True,  'prc2': True,  'gc2': False, 'tc2': False},
            {'poc2': True,  'prc2': True,  'gc2': True,  'tc2': False},
            {'poc2': True,  'prc2': True,  'gc2': False, 'tc2': True},
            {'poc2': True,  'prc2': True,  'gc2': True,  'tc2': True},
        ]
        if seed is not None:
            seed = time.time_ns() % 39916801
            self.random = np.random.RandomState(seed)
        else:
            self.random = np.random.RandomState()
        self.curr_state = {
            'poc2': False,  'prc2': True,  'gc2': True,  'tc2': True
            }

    def step(self):
        self.curr_state = self.random.choice(self.state_two)


    def reset(self):
        self.curr_state = {
            'poc2': False,  'prc2': True,  'gc2': True,  'tc2': True
            }


class FinallySuccessEnvironment1:
    def __init__(self) -> None:
        self.trace = [
            {'poc': False, 'prc': True,   'gc': True,  'tc': True},
            {'poc': False, 'prc': False,  'gc': True,  'tc': True},
            {'poc': True,  'prc': False,  'gc': True,  'tc': False}
            ]
        self.curr_state = self.trace[0]
        self.index = 1

    def step(self):
        self.curr_state = self.trace[self.index]
        self.index += 1


class FinallySuccessEnvironment2:
    def __init__(self) -> None:
        self.trace = [
            {'poc': False, 'prc': True,   'gc': True,  'tc': True},
            {'poc': False, 'prc': False,  'gc': True,  'tc': True},
            {'poc': False, 'prc': False,  'gc': True,  'tc': True},
            {'poc': False, 'prc': True,   'gc': True,  'tc': True},
            {'poc': False, 'prc': False,  'gc': True,  'tc': True},
            {'poc': True,  'prc': False,  'gc': True,  'tc': False}
            ]
        self.curr_state = self.trace[0]
        self.index = 1

    def step(self):
        self.curr_state = self.trace[self.index]
        self.index += 1


class FinallyFailureEnvironment2:
    def __init__(self) -> None:
        self.trace = [
            {'poc1': False, 'prc1': True,   'gc1': True,  'tc1': True},
            {'poc1': False, 'prc1': False,  'gc1': True,  'tc1': True},
            {'poc1': False, 'prc1': False,  'gc1': True,  'tc1': True},
            {'poc1': False, 'prc1': True,   'gc1': True,  'tc1': True},
            {'poc1': False, 'prc1': False,  'gc1': True,  'tc1': True},
            {'poc1': False, 'prc1': False,  'gc1': True,  'tc1': True},
            {'poc1': False, 'prc1': False,  'gc1': True,  'tc1': False},
            {'poc1': True,  'prc1': False,  'gc1': True,  'tc1': False}
            ]
        self.curr_state = self.trace[0]
        self.index = 1

    def step(self):
        self.curr_state = self.trace[self.index]
        self.index += 1

    def reset(self):
        self.curr_state = self.trace[0]
        self.index = 1


class UntilSuccessEnvironment1:
    def __init__(self) -> None:
        self.trace = [
            {
                'poc1': False, 'prc1': True,   'gc1': True,  'tc1': True,
                'poc2': False, 'prc2': True,   'gc2': True,  'tc2': True
            },
            {
                'poc1': False, 'prc1': False,  'gc1': True,  'tc1': True,
                'poc2': False, 'prc2': False,  'gc2': True,  'tc2': True
                },
            {
                'poc1': True,  'prc1': False,  'gc1': True,  'tc1': False,
                'poc2': False,  'prc2': True,  'gc2': True,  'tc2': True
                },
            {
                'poc1': True, 'prc1': False,   'gc1': True,  'tc1': False,
                'poc2': False, 'prc2': True,   'gc2': True,  'tc2': True
            },
            {
                'poc1': True, 'prc1': False,  'gc1': True,  'tc1': False,
                'poc2': False, 'prc2': False,  'gc2': True,  'tc2': True
                },
            {
                'poc1': True,  'prc1': False,  'gc1': True,  'tc1': False,
                'poc2': True,  'prc2': False,  'gc2': True,  'tc2': True
                }
            ]
        self.curr_state = self.trace[0]
        self.index = 1

    def step(self):
        self.curr_state = self.trace[self.index]
        self.index += 1


class ANDSuccessEnvironment1:
    def __init__(self) -> None:
        self.trace = [
            {
                'poc1': True, 'prc1': False,   'gc1': True,  'tc1': False,
                'poc2': True, 'prc2': False,   'gc2': True,  'tc2': False
            }
            ]
        self.curr_state = self.trace[0]
        self.index = 1

    def step(self):
        self.curr_state = self.trace[self.index]
        self.index += 1


class ORSuccessEnvironment1:
    def __init__(self) -> None:
        self.trace = [
            {
                'poc1': False, 'prc1': False,   'gc1': False,  'tc1': False,
                'poc2': True, 'prc2': False,   'gc2': True,  'tc2': False
            }
            ]
        self.curr_state = self.trace[0]
        self.index = 1

    def step(self):
        self.curr_state = self.trace[self.index]
        self.index += 1


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


def run_experiment_until(k):
    ppatask1 = "G(gc1) & (poc1 | (prc1 & (tc1 U ((poc1) & G(gc1)))))"
    ppatask2 = "G(gc2) & (poc2 | (prc2 & (tc2 U ((poc2) & G(gc2)))))"
    mission = '((k) U (l))'
    parser = LTLfParser()
    mission_formual = parser(mission)
    long_mission = '(' +ppatask1 + ') U (' + ppatask2 + ')'
    parser = LTLfParser()
    long_formula = parser(long_mission)
    # print(long_formula)
    # env = UntilSuccessEnvironment1()
    env = Environment()
    bboard = blackboard.Client(name='gbt')
    bboard.register_key(key='trace', access=common.Access.WRITE)
    bboard.trace = [env.curr_state]
    action_node1 = ActionNode('poc1', env=env, task_max=3)
    ppatask1_bt = create_action_GBT('prc1', 'poc1', 'tc1', 'gc1', action_node1)
    action_node2 = ActionNode('poc2', env=env, task_max=3)
    ppatask2_bt = create_action_GBT('prc2', 'poc2', 'tc2', 'gc2', action_node2)
    mappings = {'k': ppatask1_bt, 'l': ppatask2_bt}
    gbt = parse_ltlf(mission_formual, mappings, task_max=3)
    # print(ppatask_bt, gbt)
    gbt = BehaviourTree(gbt)
    # print(py_trees.display.ascii_tree(gbt.root))
    # py_trees.logging.level = py_trees.logging.Level.DEBUG

    while True:
        gbt.tick()
        # print(env.curr_state)
        if (gbt.root.status == common.Status.SUCCESS or gbt.root.status == common.Status.FAILURE):
            break
    ltlf_status = long_formula.truth(bboard.trace, 0)
    bt_status = gbt.root.status
    trace = bboard.trace
    if ltlf_status is True and bt_status == common.Status.SUCCESS:
        # print("trace: {}', 'BT status: {}', 'LTLf status: {}".format(
        #     trace, bt_status, ltlf_status))
        pass
    elif ltlf_status is False and bt_status == common.Status.FAILURE:
        # print("trace: {}', 'BT status: {}', 'LTLf status: {}".format(
        #     trace, bt_status, ltlf_status))
        pass
    else:
        print("{} trace: {}, BT status: {}, LTLf status: {} {}".format(
            bcolors.WARNING, trace, bt_status, ltlf_status, bcolors.ENDC))
        print(gbt.root.status, ltlf_status)

    return (len(trace), ltlf_status, bt_status==common.Status.SUCCESS)


def run_experiment_finally(k):
    ppatask_formula = "G(gc1) & (poc1 | (prc1 & (tc1 U ((poc1) & G(gc1)))))"
    parser = LTLfParser()
    formula = parser(ppatask_formula)

    mission = '(F(k))'
    parser = LTLfParser()
    mission_formual = parser(mission)
    long_mission = 'F(' + ppatask_formula + ')'
    parser = LTLfParser()
    long_formula = parser(long_mission)
    # print(long_formula)
    # env = FinallyFailureEnvironment2()
    env = Environment()
    bboard = blackboard.Client(name='gbt')
    bboard.register_key(key='trace', access=common.Access.WRITE)
    bboard.trace = [env.curr_state]
    action_node = ActionNode('poc1', env=env, task_max=3)
    # ppatask_bt = create_PPATask_GBT('b', 'a', 'd', 'c', action_node)
    ppatask_bt = create_action_GBT('prc1', 'poc1', 'tc1', 'gc1', action_node)
    # print(dir(ppatask_bt))
    mappings = {'k': ppatask_bt}
    gbt = parse_ltlf(mission_formual, mappings, task_max=3)
    # print(ppatask_bt, gbt)
    gbt = BehaviourTree(gbt)
    # print(py_trees.display.ascii_tree(gbt.root))
    # py_trees.logging.level = py_trees.logging.Level.DEBUG

    while True:
        gbt.tick()
        # print(env.curr_state)
        if (gbt.root.status == common.Status.SUCCESS or gbt.root.status == common.Status.FAILURE):
            break
    ltlf_status = long_formula.truth(bboard.trace, 0)
    bt_status = gbt.root.status
    trace = bboard.trace
    if ltlf_status is True and bt_status == common.Status.SUCCESS:
        # print("trace: {}', 'BT status: {}', 'LTLf status: {}".format(
        #     trace, bt_status, ltlf_status))
        pass
    elif ltlf_status is False and bt_status == common.Status.FAILURE:
        # print("trace: {}', 'BT status: {}', 'LTLf status: {}".format(
        #     trace, bt_status, ltlf_status))
        pass
    else:
        print("{} trace: {}, BT status: {}, LTLf status: {} {}".format(
            bcolors.WARNING, trace, bt_status, ltlf_status, bcolors.ENDC))
        print(gbt.root.status, ltlf_status)

    return (len(trace), ltlf_status, bt_status==common.Status.SUCCESS)


def run_experiment_and(k):
    ppatask1 = "G(gc1) & (poc1 | (prc1 & (tc1 U ((poc1) & G(gc1)))))"
    ppatask2 = "G(gc2) & (poc2 | (prc2 & (tc2 U ((poc2) & G(gc2)))))"
    mission = '((k) & (l))'
    parser = LTLfParser()
    mission_formual = parser(mission)
    long_mission = '(' +ppatask1 + ') & (' + ppatask2 + ')'
    parser = LTLfParser()
    long_formula = parser(long_mission)
    # print(long_formula)
    env1 = Environment1()
    action_node1 = ActionNode('poc1', env=env1, task_max=3)
    bboard1 = blackboard.Client(name='Action'+'poc1', namespace='poc1')
    bboard1.register_key(key='trace', access=common.Access.WRITE)
    bboard1.trace = [env1.curr_state]
    ppatask1_bt = create_action_GBT('prc1', 'poc1', 'tc1', 'gc1', action_node1)

    env2 = Environment2()
    action_node2 = ActionNode('poc2', env=env2, task_max=3)
    bboard2 = blackboard.Client(name='Action'+'poc2', namespace='poc2')
    bboard2.register_key(key='trace', access=common.Access.WRITE)
    bboard2.trace = [env2.curr_state]
    ppatask2_bt = create_action_GBT('prc2', 'poc2', 'tc2', 'gc2', action_node2)
    mappings = {'k': ppatask1_bt, 'l': ppatask2_bt}
    gbt = parse_ltlf(mission_formual, mappings, task_max=3)
    # print(ppatask_bt, gbt)
    gbt = BehaviourTree(gbt)
    # print(py_trees.display.ascii_tree(gbt.root))
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    while True:
        gbt.tick()
        # print(env.curr_state)
        if (gbt.root.status == common.Status.SUCCESS or gbt.root.status == common.Status.FAILURE):
            break
    traces = combine_traces(bboard1.trace, bboard2.trace)
    ltlf_status = long_formula.truth(traces, 0)
    bt_status = gbt.root.status
    if ltlf_status is True and bt_status == common.Status.SUCCESS:
        # print("trace: {}', 'BT status: {}', 'LTLf status: {}".format(
        #     trace, bt_status, ltlf_status))
        pass
    elif ltlf_status is False and bt_status == common.Status.FAILURE:
        # print("trace: {}', 'BT status: {}', 'LTLf status: {}".format(
        #     trace, bt_status, ltlf_status))
        pass
    else:
        print("{} trace: {}, BT status: {}, LTLf status: {} {}".format(
            bcolors.WARNING, traces, bt_status, ltlf_status, bcolors.ENDC))
        print(gbt.root.status, ltlf_status)

    return (len(traces), ltlf_status, bt_status==common.Status.SUCCESS)

def run_experiment_or(k):
    ppatask1 = "G(gc1) & (poc1 | (prc1 & (tc1 U ((poc1) & G(gc1)))))"
    ppatask2 = "G(gc2) & (poc2 | (prc2 & (tc2 U ((poc2) & G(gc2)))))"
    mission = '((k) | (l))'
    parser = LTLfParser()
    mission_formual = parser(mission)
    long_mission = '(' +ppatask1 + ') | (' + ppatask2 + ')'
    parser = LTLfParser()
    long_formula = parser(long_mission)
    # print(long_formula)
    env1 = Environment1()
    action_node1 = ActionNode('poc1', env=env1, task_max=3)
    bboard1 = blackboard.Client(name='Action'+'poc1', namespace='poc1')
    bboard1.register_key(key='trace', access=common.Access.WRITE)
    bboard1.trace = [env1.curr_state]
    ppatask1_bt = create_action_GBT('prc1', 'poc1', 'tc1', 'gc1', action_node1)

    env2 = Environment2()
    action_node2 = ActionNode('poc2', env=env2, task_max=3)
    bboard2 = blackboard.Client(name='Action'+'poc2', namespace='poc2')
    bboard2.register_key(key='trace', access=common.Access.WRITE)
    bboard2.trace = [env2.curr_state]
    ppatask2_bt = create_action_GBT('prc2', 'poc2', 'tc2', 'gc2', action_node2)
    mappings = {'k': ppatask1_bt, 'l': ppatask2_bt}
    gbt = parse_ltlf(mission_formual, mappings, task_max=3)
    # print(ppatask_bt, gbt)
    gbt = BehaviourTree(gbt)
    # print(py_trees.display.ascii_tree(gbt.root))
    # py_trees.logging.level = py_trees.logging.Level.DEBUG

    while True:
        gbt.tick()
        # print(env.curr_state)
        if (gbt.root.status == common.Status.SUCCESS or gbt.root.status == common.Status.FAILURE):
            break
    traces = combine_traces(bboard1.trace, bboard2.trace)
    # print(traces, long_formula)
    ltlf_status = long_formula.truth(traces, 0)
    bt_status = gbt.root.status
    if ltlf_status is True and bt_status == common.Status.SUCCESS:
        # print("trace: {}', 'BT status: {}', 'LTLf status: {}".format(
        #     trace, bt_status, ltlf_status))
        pass
    elif ltlf_status is False and bt_status == common.Status.FAILURE:
        # print("trace: {}', 'BT status: {}', 'LTLf status: {}".format(
        #     trace, bt_status, ltlf_status))
        pass
    else:
        print("{} trace: {}, BT status: {}, LTLf status: {} {}".format(
            bcolors.WARNING, traces, bt_status, ltlf_status, bcolors.ENDC))
        print(gbt.root.status, ltlf_status)

    return (len(traces), ltlf_status, bt_status==common.Status.SUCCESS)


def combine_traces(traces1, traces2):
    # print(len(traces1), len(traces2))
    # print(traces1, traces2)
    if len(traces1) >= len(traces2):
        for i in range(len(traces1)):
            if i >= len(traces2):
                traces1[i].update(traces2[-1])
            else:
                traces1[i].update(traces2[i])

        return traces1
    else:
        for i in range(len(traces2)):
            if i >= len(traces1):
                traces2[i].update(traces1[-1])
            else:
                traces2[i].update(traces1[i])
        return traces2


def combine_traces_alt(traces1, traces2):
    return traces1 + traces2


def main():
    # run_experiment_finally()
    # run_experiment_until()
    # run_experiment_and()
    # run_experiment_or()
    with WorkerPool(n_jobs=8) as pool:
        results = pool.map(run_experiment_and, range(1024*16), progress_bar=True)
    pd_data = pd.DataFrame(data=np.array(results))
    # Where BT and LTf return success
    data_subset = pd_data.loc[(pd_data[1]==1) & (pd_data[1]==1)][0].to_numpy()
    # np.save('/home/aadeshnpn/Desktop/finally_trace_data.npy', pd_data.to_numpy())
    print( 'Successful Traces',
        'Mean, Media, Q1, Q3, Max',
        np.mean(data_subset),
        np.quantile(data_subset, q=0.5),
        np.quantile(data_subset, q=0.25),
        np.quantile(data_subset, q=0.75),
        np.max(data_subset)
        )
    percent = ((len(data_subset)*1.0)/ ( len(results) * 1.0)) * 100
    print('Successful Traces %', percent)

    # Where BT and LTf disagree
    data_subset = pd_data.loc[(pd_data[1]==0) & (pd_data[1]==0)][0].to_numpy()
    print( 'Fail Traces',
        'Mean, Media, Q1, Q3, Max',
        np.mean(data_subset),
        np.quantile(data_subset, q=0.5),
        np.quantile(data_subset, q=0.25),
        np.quantile(data_subset, q=0.75),
        np.max(data_subset)
        )
    percent = ((len(data_subset)*1.0)/ ( len(results) * 1.0)) * 100
    print('Failed Traces %', percent)


if __name__ == '__main__':
    main()
    # run_experiment_finally(9)
    # run_experiment_or(9)
    # run_experiment_and(9)