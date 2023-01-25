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
            if self.env.get_states()[self.proposition_symbol]:
            # if self.blackboard.trace[-1][self.proposition_symbol]:
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
        self.memory = common.Status.SUCCESS

    def reset(self, **kwargs):
        self.memory = common.Status.SUCCESS

    def setup(self, **kwargs):
        pass

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
        self.memory = common.Status.SUCCESS
        self.first = True

    def reset(self, **kwargs):
        self.memory = common.Status.SUCCESS
        self.first = True

    def setup(self, **kwargs):
        pass

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Precondition logic
        """
        #  Repeat until logic for decorator
        return_value = self.decorated.status
        # print(self.decorated.name, return_value, self.idx, self.memory)
        if (self.first and return_value == common.Status.SUCCESS):
            self.memory = common.Status.SUCCESS
        elif (self.first and return_value == common.Status.FAILURE):
            self.memory = common.Status.FAILURE
        self.first = False
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
        self.previous = common.Status.SUCCESS

    def reset(self, **kwargs):
        self.previous = common.Status.SUCCESS

    def setup(self, **kwargs):
        pass

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Precondition logic
        """
        #  Repeat until logic for decorator
        child_status = self.decorated.status

        return_value = self.previous
        self.previous = child_status
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
        # print('action node',self.index, self.blackboard.trace[-1])
        self.env.step()
        self.index += 1
        self.blackboard.trace.append(self.env.curr_state)
        curr_symbol_truth_value = self.env.curr_state[self.action_symbol]
        # print('action node',self.name, self.index, self.task_max, self.blackboard.trace[-1])
        if  curr_symbol_truth_value and self.index <= self.task_max:
            return common.Status.SUCCESS
        elif curr_symbol_truth_value == False and self.index < self.task_max:
            # print('Inside running')
            return common.Status.RUNNING
        else:
            return common.Status.FAILURE


class LearnerRootNode(Decorator):
    """Learner Decorator node for the aggreating trace results.

    Inherits the Decorator class from py_trees. This
    decorator implements the learning of policy when
    the trace is failure or success.
    """

    def __init__(
            self, child, name=common.Name.AUTO_GENERATED,
            policy=None, discount=0.9, env=None):
        """Init method for the action node."""
        super(LearnerRootNode, self).__init__(name=name, child=child)
        self.action_symbol = name
        self.blackboard = blackboard.Client(name='gbt')
        self.blackboard.register_key(key='trace', access=common.Access.READ)
        self.tkey = 'trace'+name
        self.blackboard.register_key(key=self.tkey, access=common.Access.WRITE)
        self.index = 0
        self.gtable = policy
        self.psi = discount
        self.env = env

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
        # If return success,
        #  ->update the policy for all the state in the trace with
        #   positive rewards.
        # If return failure,
        # -> update the policy for all the state in the trace with
        #   negative rewards.
        child_status = self.decorated.status
        if child_status == common.Status.RUNNING:
            # print(child_status, self.blackboard.trace)
            pass
        else:
            if len(self.blackboard.get(self.tkey))>0:
                # print('from propagation step', self.action_symbol, child_status)
                self.blackboard.set(self.tkey,
                    self.blackboard.get(self.tkey) + [self.blackboard.trace[-1]]
                )
                tracea = []
                traces = [state['state'] for state in self.blackboard.get(self.tkey)]
                # traces = [state['state'] for state in self.blackboard.trace]
                for state in self.blackboard.get(self.tkey):
                    if state.get('action', None) is not None:
                        tracea.append(state['action'])
                tracea = tracea[::-1]
                traces = traces[::-1]
                # psi = 0.9
                # print([state['state'] for state in self.blackboard.trace])
                # print(traces)
                # print(tracea)
                # previous_action = None
                # previous_state = None
                j = 1
                for i in range(0, len(traces)-1, 1):
                    a = tracea[i]
                    ss = traces[i+1]
                    # print(ss, a, self.gtable[ss])
                    # if previous_state == ss and previous_action == a:
                    #          pass
                    # else:
                    prob = self.gtable[ss][a]
                    Psi = pow(self.psi, j)
                    j += 1
                    if child_status == common.Status.FAILURE:
                        # temp_state = self.blackboard.trace[i+1]
                        # # print(child_status, temp_state)
                        # if temp_state['t'] is False or temp_state['g'] is False or temp_state['p'] is False:
                        #     new_prob = prob - (Psi * prob)
                        # else:
                        #     new_prob = prob
                        new_prob = prob -  (Psi * prob)
                    elif child_status == common.Status.SUCCESS:
                        new_prob = prob + (Psi * prob)

                    self.gtable[ss][a] = new_prob
                    probs = np.array(list(self.gtable[ss].values()))
                    probs = probs / probs.sum()
                    self.gtable[ss] = dict(zip(self.gtable[ss].keys(), probs))
                    # previous_state = ss
                    # previous_action = a
                # print('after propagation', self.action_symbol, self.gtable, child_status)
                self.blackboard.set(self.tkey, [])
                # self.env.reset()
        return child_status


def create_PPATask_GBT(precond, postcond, taskcnstr, gblcnstr, action_node):
    seletector_ppatask = Selector('lambda_ppatask', memory=False)
    post_blk = Sequence('sigma_postblk', memory=False)
    pre_blk = Sequence('sigma_preblk', memory=False)
    task_seq = Sequence('sigma_task', memory=False)
    until_seq = Sequence('sigma_until', memory=False)
    action_seq = Parallel('sigma_action')
    precond_node  = ConditionNode(precond,  env=action_node.env)
    postcond_node  = ConditionNode(postcond, env=action_node.env)
    taskcnstr_node  = ConditionNode(taskcnstr, env=action_node.env)
    gblcnstr_node  = ConditionNode(gblcnstr, env=action_node.env)
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


def create_PPATask_GBT_learn(
        precond, postcond, taskcnstr, gblcnstr, action_node):
    ppatask = create_PPATask_GBT(
        precond, postcond, taskcnstr, gblcnstr, action_node)
    learner = LearnerRootNode(
        ppatask, name=postcond, policy=action_node.gtable,
        discount=action_node.discount, env=action_node.env)
    return learner


################Mission Operators######################################
# Just a simple decorator node that implements Finally mission operator
class Finally(Decorator):
    """Decorator node for the Finally operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Finally LTLf operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED, task_max=2):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Finally, self).__init__(name=name, child=child)
        self.memory = common.Status.SUCCESS
        self.task_max = task_max
        self.idx = 0

    def reset(self, **kwargs):
        self.memory = common.Status.SUCCESS
        def reset(children, **kwargs):
            for child in children:
                try:
                    child.reset(**kwargs)
                except AttributeError:
                    reset(child.children, **kwargs)
        reset(self.children, **kwargs)

    def setup(self, **kwargs):
        pass

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
            self.reset()
            if self.idx > self.task_max:
                return common.Status.FAILURE
            else:
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
        self.first = common.Status.SUCCESS

    def reset(self, **kwargs):
        self.first = common.Status.SUCCESS
        def reset(children, **kwargs):
            for child in children:
                try:
                    child.reset(**kwargs)
                except AttributeError:
                    reset(child.children, **kwargs)
        reset(self.children, **kwargs)

    def setup(self, **kwargs):
        pass

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Until operator status
        """
        #  Repeat until logic for decorator
        if self.decorated.status == common.Status.RUNNING:
            return self.decorated.status
        else:
            return_value = self.first
            self.first = self.decorated.status
            # print('Until', return_value, self.first)
            return return_value


# Just a simple decorator node that implements Reset Decorator
class Reset(Decorator):
    """Decorator node for the Until operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Reset Decorator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED, tmax=2):
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

    def reset(self, **kwargs):
        self.memory = common.Status.SUCCESS
        # self.idx = 0
        def reset(children, **kwargs):
            for child in children:
                try:
                    child.reset(**kwargs)
                except AttributeError:
                    reset(child.children, **kwargs)
        reset(self.children, **kwargs)

    def setup(self):
        self.decorated.setup()

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Globally operator status
        """
        #  Repeat until logic for decorator
        return_value = self.decorated.status
        self.idx += 1
        print('From reset', self.idx, self.tmax, return_value)
        if return_value == common.Status.RUNNING:
            return return_value
        elif return_value == common.Status.SUCCESS and self.idx <= self.tmax:
            return return_value
        elif self.idx > self.tmax:
            return common.Status.FAILURE
        elif return_value == common.Status.FAILURE and self.idx <=self.tmax:
            self.reset()
            return common.Status.RUNNING
        else:
            return common.Status.FAILURE


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
                leftnode = parse_ltlf(leftformual, mappings, task_max=task_max)
                rightnode = parse_ltlf(rightformula, mappings, task_max=task_max)
                parll.add_children([leftnode, rightnode])
                return parll
            elif isinstance(formula, LTLfOr):
                leftformual, rightformula = formula.formulas
                ornode = Selector('Or', memory=False)
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
                untila = Until(leftnode, name='Until')
                untilb = Reset(rightnode, name='Reset', tmax=task_max)
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