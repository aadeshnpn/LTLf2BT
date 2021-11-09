"""Algorithm to create GBT given a goal specification."""
from flloat.ltlf import LTLfAlways, LTLfAnd, LTLfEventually, LTLfNext, LTLfNot, LTLfOr, LTLfUntil
from flloat.parser.ltlf import LTLfParser, LTLfAtomic

from py_trees.composites import Sequence, Selector, Parallel
from py_trees.decorators import Decorator, Inverter
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees import common, blackboard
import py_trees
from py_trees import common
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


class PropConditionNodeEnv(Behaviour):
    """Condition node for the atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the condition node for the atomic LTLf propositions.
    """

    def __init__(self, name, env):
        """Init method for the condition node."""
        super(PropConditionNodeEnv, self).__init__(name)
        self.proposition_symbol = name
        self.env = env
        # self.blackboard = blackboard.Client(name='gbt')
        # self.blackboard.register_key(key='trace', access=common.Access.WRITE)

    # def setup(self, timeout, value=False):
    def setup(self, timeout, trace=[], i=0):
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
        try:
            if self.env.states[self.proposition_symbol]:
                return_value = common.Status.SUCCESS
            else:
                return_value = common.Status.FAILURE
        except IndexError:
            return_value = common.Status.FAILURE

        return return_value


# Just a simple decorator node that implements Negation
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


# Just a simple decorator node that implements And LTLf operator
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

        def setupreq(child, trace, i):
            try:
                # print('setupreq', child)
                child.setup(0, trace, i)
            except:
                for c in child.children:
                    setupreq(c, trace, i)
        for c1 in childs:
            setupreq(c1, trace, i)
        # for child in childs:
        #     try:
        #         child.setup(0, trace, i)
        #     except TypeError:
        #         for c in child.children:
        #             c.setup(0, trace, i)

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


# Just a simple decorator node that implements Or LTLf operator
class Or(Decorator):
    """Decorator node for the Or of an atomic proposition.

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
        super(Or, self).__init__(name=name, child=child)

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

        def setupreq(child, trace, i):
            try:
                # print('setupreq', child)
                child.setup(0, trace, i)
            except:
                for c in child.children:
                    setupreq(c, trace, i)
        for c1 in childs:
            setupreq(c1, trace, i)
        # for child in childs:
        #     try:
        #         child.setup(0, trace, i)
        #     except TypeError:
        #         for c in child.children:
        #             c.setup(0, trace, i)

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


# Just a simple decorator node that implements Next LTLf operator
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

        # return self.next_status
        return self.decorated.status


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
                    return_value = list(self.decorated.tick())[-1].status
                except AttributeError:
                    [c.reset(k) for c in self.decorated.children]
                    [c.setup(0, self.trace, k) for c in self.decorated.children]
                    # print(self.decorated, self.decorated.status)
                    # print(list(self.decorated.tick())[-1])
                    return_value = list(self.decorated.tick())[-1].status
                if return_value == common.Status.FAILURE:
                    break
        else:
            return_value = common.Status.FAILURE
        return return_value


# Just a simple decorator node that implements Delta_A operator for planner
class DeltaA(Decorator):
    """Decorator node for the Delta A operator.

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
        super(DeltaA, self).__init__(name=name, child=child)
        self.idx = 0
        self.blackboard = blackboard.Client(name='gbt')
        self.blackboard.register_key(key='trace', access=common.Access.WRITE)

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
        ## Add to dashboard is the state satisfies the post condition
        if return_value == common.Status.SUCCESS:
            self.blackboard.trace.append(self.env.states)
        return return_value


class ActionNode(Behaviour):
    """Action node for the planning atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the action node for the planning LTLf propositions.
    """

    def __init__(self, name, env, planner=None):
        """Init method for the action node."""
        super(ActionNode, self).__init__('Action'+name)
        self.action_symbol = name
        # self.blackboard = blackboard.Client(name='gbt')
        # self.blackboard.register_key(key='trace', access=common.Access.WRITE)
        self.env = env
        self.planner = planner

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
        pass


def create_planner_subtree(pname, env, planner):
    postcond_left = PropConditionNodeEnv(pname, env)
    deltaA_left = DeltaA(postcond_left, 'DeltaAL')

    postcond_right = PropConditionNodeEnv(pname, env)
    deltaA_right = DeltaA(postcond_right, 'DeltaAR')

    actionplanner = ActionNode(pname, env, planner=planner)

    parallelnode = Parallel('PlanParallel')
    parallelnode.add_children([actionplanner, deltaA_right])

    planroot = Selector('PRoot')
    planroot.add_children([deltaA_left, parallelnode])

    return planroot


def create_recognizer(formulas, debug=False, bt=False):
    parser = LTLfParser()
    formula = parser(formulas)
    if bt:
        bt = BehaviourTree(parse_ltlf(formula))
        if debug:
            print(py_trees.display.ascii_tree(bt.root))
    else:
        bt = parse_ltlf(formula)
    return bt


def create_ppatask(postcond, precond, taskcnstr, gcnstr):
    # PostCond | (PreCond & X (TaskBulk U PostCond))
    postbulk = gcnstr + ' & '+  postcond
    prebulk = gcnstr + ' & '+  precond
    taskbulk = gcnstr + ' & '+  taskcnstr
    ppatask = '('+ postbulk + ') | ((' + prebulk + ') & X((' + taskbulk + ') U (' +postbulk + '))' +  ')'
    print(ppatask)
    parser = LTLfParser()
    ppaformula = parser(ppatask)
    create_recognizer(ppaformula)


def create_generator():
    pass


def create_planner(postcond='C', env=None, plan_algo=None):
    planroot = create_planner_subtree(postcond, env, plan_algo)
    print(py_trees.display.ascii_tree(planroot))


def parse_ltlf(formula):
    # Just proposition
    if isinstance(formula, LTLfAtomic):
        # map to BT conditio node
        return PropConditionNode(formula.s)
    # Unary or binary operator
    else:
        if (
            isinstance(formula, LTLfEventually) or isinstance(formula, LTLfAlways)
                or isinstance(formula, LTLfNext) or isinstance(formula, LTLfNot) ):
            if isinstance(formula, LTLfEventually):
                return Finally(parse_ltlf(formula.f))
            elif isinstance(formula, LTLfAlways):
                return Globally(parse_ltlf(formula.f))
            elif isinstance(formula, LTLfNext):
                return Next(parse_ltlf(formula.f))
            elif isinstance(formula, LTLfNot):
                return Negation(parse_ltlf(formula.f))

        elif (isinstance(formula, LTLfAnd) or isinstance(formula, LTLfOr) or isinstance(formula, LTLfUntil)):
            if isinstance(formula, LTLfAnd):
                leftformual, rightformula = formula.formulas
                parll = Parallel('And')
                leftnode = parse_ltlf(leftformual)
                rightnode = parse_ltlf(rightformula)
                parll.add_children([leftnode, rightnode])
                anddecorator = And(parll)
                return anddecorator
            elif isinstance(formula, LTLfOr):
                leftformual, rightformula = formula.formulas
                ornode = Selector('Or')
                leftnode = parse_ltlf(leftformual)
                rightnode = parse_ltlf(rightformula)
                ornode.add_children([leftnode, rightnode])
                ordecorator = Or(ornode)
                return ordecorator

            elif isinstance(formula, LTLfUntil):
                leftformual, rightformula = formula.formulas
                leftnode = parse_ltlf(leftformual)
                rightnode = parse_ltlf(rightformula)
                useq = Sequence('UntilSeq')
                untila = UntilA(leftnode, name='<j')
                untilb = UntilB(rightnode, name='=j')
                useq.add_children([untilb, untila])
                anddec2 = And(useq, name='UntilAnd')
                untildecorator = Until(anddec2, name='Until')
                return untildecorator


# def main():
#     # formulas = [
#     #     '(a)', '(!a)', 'F(a)', 'G(a)', 'X(a)',
#     #     '(a | b)', '(a & b)', '(a U b)']
#     # for formula_string in formulas:
#     #     parser = LTLfParser()
#     #     formula = parser(formula_string)
#     #     create_recognizer(formula)
#     # ppa1 = '(!t & s) | ( (!t & !c) & X((!t) U (!t & s)))'
#     # ppa2 = '(!t & c) | ( (!t & s) & X((!t & o) U (!t & c)))'
#     # ppa = '('+ ppa1 + ') U (' + ppa2 + ')'
#     # complexformulas = [ppa1, ppa2, ppa]

#     # for formula_string in complexformulas:
#     #     parser = LTLfParser()
#     #     formula = parser(formula_string)
#     #     create_recognizer(formula)

#     # create_ppatask('c', 's', 'o', '!t')


# main()
# create_planner()