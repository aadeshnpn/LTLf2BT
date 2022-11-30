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


class PropConditionNode(Behaviour):
    """Condition node for the atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the condition node for the atomic LTLf propositions.
    """

    def __init__(self, name):
        """Init method for the condition node."""
        super(PropConditionNode, self).__init__(name)
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
        #     self.name, self.index, self.proposition_symbol,
        #     self.trace[self.index][self.proposition_symbol])
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
        super(PreCond, self).__init__(name=name, child=child)
        self.idx = 0
        self.memory = common.Status.SUCCESS

    def reset(self, i=0):
        self.memory = common.Status.SUCCESS


    def setup(self, timeout, i=0):
        self.decorated.setup(0, self.idx)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Precondition logic
        """
        #  Repeat until logic for decorator
        return_value = self.decorated.status
        if self.idx ==0:
            self.memory = common.Status.SUCCESS
        elif (self.idx > 0 and self.memory == common.Status.FAILURE):
            self.memory = common.Status.FAILURE

        self.idx += 1

        return self.memory
