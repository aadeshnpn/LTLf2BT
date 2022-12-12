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


class ConditionNode(Behaviour):
    """Condition node for the atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the condition node for the atomic LTLf propositions.
    """

    def __init__(self, name):
        """Init method for the condition node."""
        super(ConditionNode, self).__init__(name)
        self.proposition_symbol = name
        self.blackboard = blackboard.Client(name='gbt')
        self.blackboard.register_key(key='trace', access=common.Access.WRITE)
        # common.Name.AUTO_GENERATED

    # def setup(self, timeout, value=False):
    def setup(self, timeout, index=0):
        """Have defined the setup method.

        This method defines the other objects required for the
        condition node.
        Args:
        timeout: property from Behavior super class. Not required
        index: index of the trace. Not required
        """
        self.index = index

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
        #     self.name, self.proposition_symbol, self.blackboard.trace[-1],
        #     self.blackboard.trace[-1][self.proposition_symbol])
        try:
            if self.blackboard.trace[-1][self.proposition_symbol]:
                return_value = common.Status.SUCCESS
            else:
                return_value = common.Status.FAILURE
        except IndexError:
            return_value = common.Status.FAILURE

        return return_value


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
        self.memory = common.Status.SUCCESS

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
        if return_value == common.Status.FAILURE:
            self.memory = common.Status.FAILURE
        # rint('globally', self.memory)
        return self.memory


class PreCond(Decorator):
    """Decorator node for the Precondition decorator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Precondition decorator logic.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(PreCond, self).__init__(name=name, child=child)
        self.idx = 0
        self.memory = common.Status.SUCCESS

    def reset(self, i=0):
        self.memory = common.Status.SUCCESS
        self.idx = 0


    def setup(self, timeout, i=0):
        self.decorated.setup(0, self.idx)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Precondition logic
        """
        #  Repeat until logic for decorator
        return_value = self.decorated.status
        # print(self.decorated.name, return_value, self.idx, self.memory)
        if (self.idx ==0 and return_value == common.Status.SUCCESS):
            self.memory = common.Status.SUCCESS
        elif (self.idx ==0 and return_value == common.Status.FAILURE):
            self.memory = common.Status.FAILURE
        self.idx += 1
        return self.memory


class TaskCnstr(Decorator):
    """Decorator node for the Task Constraint.

    Inherits the Decorator class from py_trees. This
    behavior implements the Task constraint decorator logic.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(TaskCnstr, self).__init__(name=name, child=child)
        self.idx = 0
        self.memory = common.Status.FAILURE

    def reset(self, i=0):
        self.memory = common.Status.FAILURE


    def setup(self, timeout, i=0):
        self.decorated.setup(0, self.idx)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Precondition logic
        """
        #  Repeat until logic for decorator
        child_status = self.decorated.status
        # print('taskcnstr', child_status, self.idx)
        if self.idx ==0:
            return_value =  common.Status.SUCCESS
        elif (self.idx > 0 and self.memory == common.Status.FAILURE):
            return_value = common.Status.FAILURE
        elif (self.idx > 0 and self.memory == common.Status.SUCCESS):
            return_value = common.Status.SUCCESS

        if self.idx > 0 and self.memory == common.Status.FAILURE:
            pass
        else:
            self.memory = child_status

        self.idx += 1
        # print('taskcnstr', child_status, self.idx, return_value)
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
        # Plan action and take that action in the environment.
        # print('action node',self.index, self.blackboard.trace[-1])
        self.env.step()
        self.index += 1
        self.blackboard.trace.append(self.env.curr_state)
        curr_symbol_truth_value = self.blackboard.trace[-1][self.action_symbol]
        # print('action node',self.name, self.index, self.task_max, self.blackboard.trace[-1])
        if  curr_symbol_truth_value and self.index <= self.task_max:
            return common.Status.SUCCESS
        elif curr_symbol_truth_value == False and self.index < self.task_max:
            # print('Inside running')
            return common.Status.RUNNING
        else:
            return common.Status.FAILURE


def create_PPATask_GBT(precond, postcond, taskcnstr, gblcnstr, action_node):
    seletector_ppatask = Selector('lambda_ppatask', memory=False)
    post_blk = Sequence('sigma_postblk', memory=False)
    pre_blk = Sequence('sigma_preblk', memory=False)
    task_seq = Sequence('sigma_task', memory=False)
    until_seq = Sequence('sigma_until', memory=False)
    action_seq = Parallel('sigma_action')
    precond_node  = ConditionNode(precond)
    postcond_node  = ConditionNode(postcond)
    taskcnstr_node  = ConditionNode(taskcnstr)
    gblcnstr_node  = ConditionNode(gblcnstr)
    gblcnstr_decorator_1 = Globally(gblcnstr_node)
    gblcnstr_decorator_2 = copy.copy(gblcnstr_decorator_1)
    gblcnstr_decorator_3 = copy.copy(gblcnstr_decorator_1)
    precond_decorator = PreCond(precond_node)
    taskcnstr_decorator = TaskCnstr(taskcnstr_node)
    action_seq.add_children([action_node, gblcnstr_decorator_3])
    until_seq.add_children([taskcnstr_decorator, action_seq])
    pre_blk.add_children([gblcnstr_decorator_2, precond_decorator])
    task_seq.add_children([pre_blk, until_seq])
    post_blk.add_children([gblcnstr_decorator_1, postcond_node])
    seletector_ppatask.add_children([post_blk, task_seq])
    return seletector_ppatask


# Just a simple decorator node that implements Finally mission operator
class Finally(Decorator):
    """Decorator node for the Finally operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Finally LTLf operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED, task_max=4):
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
        self.idx += 1
        if return_value == common.Status.RUNNING:
            return common.Status.RUNNING
        elif return_value == common.Status.FAILURE:
            # Reset all child decorator nodes and return running
            def reset(children, i):
                for child in children:
                    try:
                        child.reset(i)
                    except AttributeError:
                        reset(child.children, i)
            reset(self.children, 0)
            if self.idx > self.task_max:
                return common.Status.FAILURE
            return common.Status.RUNNING
        return self.memory


# Just a simple decorator node that implements Until mission operator
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
        self.memory = common.Status.SUCCESS

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
        if self.decorated.status == common.Status.RUNNING:
            return self.decorated.status
        else:
            self.idx += 1
        if self.idx ==0:
            self.memory = return_value
            return common.Status.SUCCESS
        elif (self.idx >0 and self.memory == common.Status.SUCCESS):
            pass
        elif (self.idx >0 and self.memory == common.Status.FAILURE):
            return common.Status.FAILURE
        if return_value == common.Status.FAILURE:
            self.memory = common.Status.FAILURE
        self.idx += 1
        return self.memory


# Just a simple decorator node that implements Reset Decorator
class Reset(Decorator):
    """Decorator node for the Until operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Reset Decorator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED, tmax=20):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Reset, self).__init__(name=name, child=child)
        self.idx = 0
        self.memory = common.Status.SUCCESS
        self.tmax = tmax

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
        self.idx += 1
        if return_value == common.Status.SUCCESS and self.idx <= self.tmax:
            return return_value
        elif self.idx > self.tmax:
            return common.Status.FAILURE
        else:
            return common.Status.RUNNING


class PPATaskNode(Behaviour):
    """Substitute node for PPATask.

    Inherits the Behaviors class from py_trees.
    """

    def __init__(self, name):
        """Init method for the condition node."""
        super(PPATaskNode, self).__init__(name)
        self.name = name
        self.id = 0


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
        return common.Status.SUCCESS


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

        elif (isinstance(formula, LTLfAnd) or isinstance(formula, LTLfOr) or isinstance(formula, LTLfUntil)):
            if isinstance(formula, LTLfAnd):
                leftformual, rightformula = formula.formulas
                parll = Parallel('And')
                leftnode = parse_ltlf(leftformual, mappings)
                rightnode = parse_ltlf(rightformula, mappings)
                parll.add_children([leftnode, rightnode])
                return parll
            elif isinstance(formula, LTLfOr):
                leftformual, rightformula = formula.formulas
                ornode = Selector('Or', memory=False)
                leftnode = parse_ltlf(leftformual, mappings)
                rightnode = parse_ltlf(rightformula, mappings)
                ornode.add_children([leftnode, rightnode])
                # ordecorator = Or(ornode)
                return ornode

            elif isinstance(formula, LTLfUntil):
                leftformual, rightformula = formula.formulas
                leftnode = parse_ltlf(leftformual, mappings)
                rightnode = parse_ltlf(rightformula, mappings)
                useq = Sequence('UntilSeq', memory=False)
                untila = Until(leftnode, name='Until')
                untilb = Reset(rightnode, name='Reset', tmax=4)
                useq.add_children([untila, untilb])
                return useq


# def replace_dummynodes_with_PPATaskBT(gbt, mapping):
#     # list all the nodes.
#     allnodes = list(gbt.root.iterate())
#     alldummy_nodes = list(filter(
#         lambda x: isinstance(x, PPATaskNode), allnodes)
#         )
#     for node in alldummy_nodes:
#         print(node)
#         if node.name in mapping:
#             parent= node.parent
#             print(dir(parent))
#             parent.remove
#             parent.replace_child(node, mapping[node.name])
#     return gbt
