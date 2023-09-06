"""Nodes for Generative Behavior Tree."""
from flloat.ltlf import LTLfAlways, LTLfAnd, LTLfEventually, LTLfNext, LTLfNot, LTLfOr, LTLfUntil
from flloat.parser.ltlf import LTLfParser, LTLfAtomic

from py_trees.composites import Sequence, Selector, Parallel
from py_trees.decorators import Decorator, Inverter
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees import common, blackboard
import py_trees
import copy
import argparse
import numpy as np
from flloat.parser.ltlf import LTLfParser


class ConditionNode(Behaviour):
    """Condition node for the atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the condition node for the atomic LTLf propositions.
    """

    def __init__(self, name, env=None):
        """Init method for the condition node."""
        super(ConditionNode, self).__init__(name)
        self.proposition_symbol = name
        self.blackboard = blackboard.Client(name='gbt')
        self.blackboard.register_key(key='trace', access=common.Access.WRITE)
        # common.Name.AUTO_GENERATED
        self.env = env

    # def setup(self, timeout, value=False):
    def setup(self, **kwargs):
        """Have defined the setup method.

        This method defines the other objects required for the
        condition node.
        Args:
        timeout: property from Behavior super class. Not required
        index: index of the trace. Not required
        """
        pass

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def reset(self, **kwargs):
        pass

    def update(self):
        """
        Main function that is called when BT ticks.
        """
        try:
            if self.env.curr_state[self.proposition_symbol]:
                # if self.blackboard.trace[-1][self.proposition_symbol]:
                return_value = common.Status.SUCCESS
            else:
                return_value = common.Status.FAILURE
        except IndexError:
            return_value = common.Status.FAILURE

        return return_value


class ActionNode(Behaviour):
    """Action node for the planning atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the action node for the planning LTLf propositions.
    """

    def __init__(self, name, env, planner=None, task_max=3):
        """Init method for the action node."""
        super(ActionNode, self).__init__('Action'+name)
        self.action_symbol = name
        self.blackboard = blackboard.Client(name='gbt')
        self.blackboard.register_key(key='trace', access=common.Access.WRITE)
        self.env = env
        self.planner = planner
        self.index = 0
        self.task_max = task_max

    def setup(self, **kwargs):
        """Have defined the setup method.

        This method defines the other objects required for the
        condition node.
        Args:
        timeout: property from Behavior super class. Not required
        symbol: Name of the proposition symbol
        value: A dict object with key as the proposition symbol and
               boolean value as values. Supplied by trace.
        """
        pass

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def reset(self, **kwargs):
        self.index = 0

    def update(self):
        """
        Main function that is called when BT ticks.
        """
        # Plan action and take that action in the environment.
        # print('action node before',self.index, self.env.curr_state)
        self.env.step()
        self.index += 1
        # print('action node after',self.index, self.env.curr_state)
        self.blackboard.trace.append(self.env.curr_state)
        curr_symbol_truth_value = self.env.curr_state[self.action_symbol]
        if curr_symbol_truth_value and self.index <= self.task_max:
            return common.Status.SUCCESS
        elif curr_symbol_truth_value == False and self.index < self.task_max:
            return common.Status.RUNNING
        else:
            return common.Status.FAILURE


def create_PPATask_GBT(precond, postcond, taskcnstr, gblcnstr, action_node):
    ppatask = Parallel(
        'pi_ppatask',
        policy=py_trees.common.ParallelPolicy.SuccessOnAll())
    post_blk = Selector('lambda_postblk', memory=False)
    task_seq = Parallel(
        'pi_task',
        policy=py_trees.common.ParallelPolicy.SuccessOnAll())
    until_seq = Sequence('sigma_until', memory=False)
    action_seq = Sequence(
        'sigma_action',
        memory=False)
        # policy=py_trees.common.ParallelPolicy.SuccessOnAll())
    precond_node = ConditionNode(precond,  env=action_node.env)
    postcond_node = ConditionNode(postcond, env=action_node.env)
    taskcnstr_node = ConditionNode(taskcnstr, env=action_node.env)
    gblcnstr_node_first = ConditionNode(gblcnstr, env=action_node.env)
    gblcnstr_node_second = ConditionNode(gblcnstr, env=action_node.env)
    action_seq.add_children([action_node, gblcnstr_node_second])
    until_seq.add_children([taskcnstr_node, action_seq])
    task_seq.add_children([precond_node, until_seq])
    post_blk.add_children([postcond_node, task_seq])
    ppatask.add_children([gblcnstr_node_first, post_blk])
    return ppatask


def create_action_GBT(precond, postcond, taskcnstr, gblcnstr, action_node):
    # By default parallel policy will not tick previous child that
    # was successful last time
    # So modify the synchronise flag
    ppatask = Parallel(
        'pi_ppatask',
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(
            synchronise=False))
    post_blk = Selector('lambda_postblk', memory=False)
    task_seq = Parallel(
        'pi_task',
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(
            synchronise=False))
    until_seq = Parallel(
        'sigma_until',
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(
            synchronise=False))
    action_seq = Parallel(
        'sigma_action',
        # memory=False,
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(
            synchronise=False))
    precond_node = ConditionNode(precond,  env=action_node.env)
    postcond_node = ConditionNode(postcond, env=action_node.env)
    taskcnstr_node = ConditionNode(taskcnstr, env=action_node.env)
    gblcnstr_node_first = ConditionNode(gblcnstr, env=action_node.env)
    gblcnstr_node_second = ConditionNode(gblcnstr, env=action_node.env)
    action_seq.add_children([action_node, gblcnstr_node_second])
    until_seq.add_children([taskcnstr_node, action_seq])
    task_seq.add_children([precond_node, until_seq])
    post_blk.add_children([postcond_node, task_seq])
    ppatask.add_children([gblcnstr_node_first, post_blk])
    return ppatask


class Environment:
    def __init__(self) -> None:
        self.trace = [
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
        self.curr_state = np.random.choice(self.trace)

    def step(self):
        self.curr_state = np.random.choice(self.trace)


class SuccessEnvironment1:
    def __init__(self) -> None:
        self.trace = [
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': False, 'b': False, 'c': True, 'd': True},
            {'a': True, 'b': False, 'c': True, 'd': True},
        ]
        self.curr_state = self.trace[0]
        self.index = 1

    def step(self):
        self.curr_state = self.trace[self.index]
        self.index += 1


class SuccessEnvironment2:
    def __init__(self) -> None:
        self.trace = [
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': True},
        ]
        self.curr_state = self.trace[0]
        self.index = 1

    def step(self):
        self.curr_state = self.trace[self.index]
        self.index += 1


class SuccessEnvironment3:
    def __init__(self) -> None:
        self.trace = [
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': False},
        ]
        self.curr_state = self.trace[0]
        self.index = 1

    def step(self):
        self.curr_state = self.trace[self.index]
        self.index += 1


class FailureEnvironment1:
    def __init__(self) -> None:
        self.trace = [
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': False, 'b': True, 'c': False, 'd': True},
            {'a': True, 'b': True, 'c': True, 'd': False},
        ]
        self.curr_state = self.trace[0]
        self.index = 1

    def step(self):
        self.curr_state = self.trace[self.index]
        self.index += 1


class FailureEnvironment2:
    def __init__(self) -> None:
        self.trace = [
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': False, 'b': True, 'c': True, 'd': False},
            {'a': True, 'b': True, 'c': True, 'd': False},
        ]
        self.curr_state = self.trace[0]
        self.index = 1

    def step(self):
        self.curr_state = self.trace[self.index]
        self.index += 1


class FailureEnvironment3:
    def __init__(self) -> None:
        self.trace = [
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': False, 'b': True, 'c': True, 'd': True},
            {'a': False, 'b': True, 'c': True, 'd': False},
            {'a': False, 'b': False, 'c': False, 'd': False},
            {'a': False, 'b': False, 'c': False, 'd': False},
            {'a': True, 'b': False, 'c': False, 'd': True}]
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


def main():
    ## a,b,c,d -> PoC, PrC, GC, TC
    for i in range(100):
        ppatask_formula = "G(c) & (a | (b & (d U ((a) & G(c)))))"
        # ppatask_formula = "(a) & G(c)"
        parser = LTLfParser()
        formula = parser(ppatask_formula)
        env = Environment()
        # env = FailureEnvironment2()
        bboard = blackboard.Client(name='gbt')
        bboard.register_key(key='trace', access=common.Access.WRITE)
        bboard.trace = [env.curr_state]
        action_node = ActionNode('a', env=env, task_max=5)
        # ppatask_bt = create_PPATask_GBT('b', 'a', 'd', 'c', action_node)
        ppatask_bt = create_action_GBT('b', 'a', 'd', 'c', action_node)
        bt = BehaviourTree(ppatask_bt)
        # print(py_trees.display.ascii_tree(bt.root))
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # print(bt.root.status)
        while True:
            bt.tick()
            # print(env.curr_state)
            if (bt.root.status == common.Status.SUCCESS or bt.root.status == common.Status.FAILURE):
                break
        ltlf_status = formula.truth(bboard.trace, 0)
        bt_status = bt.root.status
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
            print(bt.root.status, ltlf_status)


if __name__ == "__main__":
    main()
