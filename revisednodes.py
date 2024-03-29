"""Algorithm to create GBT given a goal specification."""

from flloat.ltlf import (
    LTLfAlways, LTLfAnd, LTLfEventually, LTLfNext,
    LTLfNot, LTLfOr, LTLfUntil)
from flloat.parser.ltlf import LTLfParser, LTLfAtomic

from py_trees.composites import Sequence, Selector, Parallel
from py_trees.decorators import Decorator
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees import common, blackboard
import py_trees
import copy
import numpy as np


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
    def setup(self, timeout):
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
        except KeyError:
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
        self.memory = False

    def reset(self, i=0):
        self.memory = False
        def reset(children, i):
            for child in children:
                try:
                    child.reset(i)
                except AttributeError:
                    reset(child.children, i)
        reset(self.children, i)

    def setup(self, timeout):
        pass

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Finally operator status
        """
        #  Repeat until logic for decorator
        return_value = self.decorated.status
        if self.memory == common.Status.SUCCESS:
            return self.memory

        if return_value == common.Status.SUCCESS:
            self.memory = return_value

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
        # self.blackboard.trace.append(self.env.states)
        # curr_symbol_truth_value = self.blackboard.trace[-1][self.action_symbol]
        curr_symbol_truth_value = self.env.states[self.action_symbol]
        # print('action node',self.name, self.index, self.task_max, self.blackboard.trace[-1])
        if curr_symbol_truth_value and self.index <= self.task_max:
            return common.Status.SUCCESS
        elif (
                (curr_symbol_truth_value is False) and (
                    self.index < self.task_max)):
            return common.Status.RUNNING
        else:
            return common.Status.FAILURE


class ActionNodeBoard(Behaviour):
    """Action node for the planning atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the action node for the planning LTLf propositions.
    """

    def __init__(
            self, name, env, planner=None, task_max=3,
            props=['a', 'b', 'c', 'd']):
        """Init method for the action node."""
        super(ActionNodeBoard, self).__init__('Action'+name)
        self.action_symbol = name
        self.blackboard = blackboard.Client(name=name)
        self.blackboard.register_key(
            key='trace'+name, access=common.Access.WRITE)
        self.env = env
        self.planner = planner
        self.index = 0
        self.task_max = task_max
        self.props = props
        self.trace_name = 'trace'+name

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

        try:
            self.blackboard.get(self.trace_name)
        except KeyError:
            self.blackboard.set(
                self.trace_name, [{p: self.env.states[p] for p in self.props}])
        self.env.step()
        self.index += 1
        temp_trace = self.blackboard.get(
                self.trace_name)+[{p: self.env.states[p] for p in self.props}]
        # print('action node',self.blackboard.get(self.trace_name), [{p: self.env.states[p] for p in self.props}])
        self.blackboard.set(
            self.trace_name,
            temp_trace
            )
        # print('action node', self.name, 'trace appended', temp_trace)
        # curr_symbol_truth_value = self.blackboard.trace[-1][self.action_symbol]
        curr_symbol_truth_value = self.env.states[self.action_symbol]
        # print('action node',self.name, self.index, self.task_max, self.blackboard.trace[-1])
        if curr_symbol_truth_value and self.index <= self.task_max:
            return common.Status.SUCCESS
        elif (
                (curr_symbol_truth_value is False) and (
                    self.index < self.task_max)):
            return common.Status.RUNNING
        else:
            return common.Status.FAILURE


# Just a simple decorator node that implements Finally mission operator
class FinallyM(Decorator):
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
        super(FinallyM, self).__init__(name=name, child=child)
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
                return FinallyM(
                    parse_ltlf(formula.f, mappings), task_max=task_max)

        elif (
            isinstance(formula, LTLfAnd) or isinstance(
                formula, LTLfOr) or isinstance(formula, LTLfUntil)):
            if isinstance(formula, LTLfAnd):
                leftformual, rightformula = formula.formulas
                parll = Parallel('And')
                leftnode = parse_ltlf(leftformual, mappings, task_max=task_max)
                rightnode = parse_ltlf(
                    rightformula, mappings, task_max=task_max)
                parll.add_children([leftnode, rightnode])
                return parll
            elif isinstance(formula, LTLfOr):
                leftformual, rightformula = formula.formulas
                ornode = Selector('Or', memory=False)
                leftnode = parse_ltlf(
                    leftformual, mappings, task_max=task_max)
                rightnode = parse_ltlf(
                    rightformula, mappings, task_max=task_max)
                ornode.add_children([leftnode, rightnode])
                # ordecorator = Or(ornode)
                return ornode

            elif isinstance(formula, LTLfUntil):
                leftformual, rightformula = formula.formulas
                leftnode = parse_ltlf(leftformual, mappings, task_max=task_max)
                rightnode = parse_ltlf(
                    rightformula, mappings, task_max=task_max)
                useq = Sequence('UntilSeq', memory=False)
                useq.add_children([leftnode, rightnode])
                return useq


def create_PPATask_GBT(
        precond, postcond, taskcnstr, gblcnstr, action_node):
    seletector_ppatask = Selector('lambda_ppatask', memory=False)
    post_blk = Sequence('sigma_postblk')
    pre_blk = Sequence('sigma_preblk')
    task_seq = Sequence('sigma_task', memory=False)
    until_seq = Sequence('sigma_until', memory=False)
    action_seq = Sequence('sigma_action')
    precond_node = PropConditionNodeEnv(precond, action_node.env)
    precond_decorator = Finally(precond_node, name='F')
    postcond_node = PropConditionNodeEnv(postcond, action_node.env)
    taskcnstr_node = PropConditionNodeEnv(taskcnstr, action_node.env)
    gblcnstr_node1 = PropConditionNodeEnv(gblcnstr, action_node.env)
    gblcnstr_node2 = copy.copy(gblcnstr_node1)
    gblcnstr_node3 = copy.copy(gblcnstr_node1)
    action_seq.add_children([action_node, gblcnstr_node3])
    until_seq.add_children([taskcnstr_node, action_seq])
    pre_blk.add_children([gblcnstr_node2, precond_decorator])
    task_seq.add_children([pre_blk, until_seq])
    post_blk.add_children([gblcnstr_node1, postcond_node])
    seletector_ppatask.add_children([post_blk, task_seq])
    return seletector_ppatask


class Env:
    def __init__(
        self, proposition_symbols=[
            'a', 'b', 'c', 'd', 'w', 'v', 'x', 'y']):
        self.proposition_symbols = proposition_symbols
        self.states = {
            p: np.random.choice(
                [True, False]) for p in proposition_symbols}

    def step(self):
        self.states = {
            p: np.random.choice(
                [True, False]) for p in self.proposition_symbols}


def test_PPATASK():
    formula = '(G (d) & b) | ((G(d) & a) & (c U (b & G(d))))'
    parser = LTLfParser()
    task_formula = parser(formula)

    for t in range(10000):
        env = Env(['a', 'b', 'c', 'd'])
        action_node = ActionNode('b', env, task_max=3)
        bboard = blackboard.Client(name='gbt' + str(t))
        bboard.register_key(key='trace', access=common.Access.WRITE)
        try:
            print('try', bboard.get('trace'))
        except KeyError:
            # print('Need to hit this twice')
            bboard.trace = []
        bboard.trace.append(env.states)
        ppataskbt = create_PPATask_GBT('a', 'b', 'c', 'd', action_node)
        ppataskbt = BehaviourTree(ppataskbt)
        # print(py_trees.display.ascii_tree(ppataskbt.root))
        # add debug statement
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        for i in range(5):
            ppataskbt.tick()
            bboard.trace.append(env.states)
            if ppataskbt.root.status in [
                    common.Status.SUCCESS, common.Status.FAILURE]:
                break
        # print(t, bboard.trace)
        bt_status = True if ppataskbt.root.status == common.Status.SUCCESS else False
        ltlf_parse_status = task_formula.truth(bboard.trace, 0)
        # print(t, ltlf_parse_status, ppataskbt.root.status, bt_status)
        # print("\n")
        assert ltlf_parse_status == bt_status
        bboard.unset('trace')
    print(t)


def stack_trace(trace_one, trace_two):
    trace = []
    if len(trace_one) >= len(trace_two):
        for k in range(len(trace_two)):
            trace.append(
                {**trace_one[k], **trace_two[k]})
        # trace += trace_one[len(trace_two):]
    else:
        for k in range(len(trace_one)):
            trace.append(
                {**trace_one[k], **trace_two[k]})
        # trace += trace_one[len(trace_one):]

    return trace


def test_mission_until():
    formula_one = '(G (d) & b) | ((G(d) & a) & (c U (b & G(d))))'
    formula_two = '(G (y) & v) | ((G(y) & w) & (x U (v & G(y))))'
    # mission = '(F (b)) U (F (v))'
    mission = 'b U v'
    parser = LTLfParser()
    mission_formula = parser(mission)
    mission_ppa = '(' + formula_one + ') U (' + formula_two + ')'
    mparser = LTLfParser()
    mission_ppa_formual = mparser(mission_ppa)
    # print(mission_ppa_formual)
    parser_task_one = LTLfParser()
    parser_task_two = LTLfParser()
    task_one_forumal = parser_task_one(formula_one)
    task_two_forumal = parser_task_two(formula_two)
    for t in range(10000):
        # env_one = Env(['a', 'b', 'c', 'd'])
        env = Env()
        trace = [env.states]
        action_node_one = ActionNodeBoard(
            'b', env, task_max=3, props=['a', 'b', 'c', 'd'])
        action_node_two = ActionNodeBoard(
            'v', env, task_max=3, props=['w', 'v', 'x', 'y'])
        ppatask_one = create_PPATask_GBT('a', 'b', 'c', 'd', action_node_one)
        ppatask_two = create_PPATask_GBT('w', 'v', 'x', 'y', action_node_two)
        mappings = {'b': ppatask_one, 'v': ppatask_two}
        gbt = parse_ltlf(mission_formula, mappings)
        gbt = BehaviourTree(gbt)
        blackboard_one = blackboard.Client(name='b')
        blackboard_one.register_key(
            key='trace'+'b', access=common.Access.WRITE)
        blackboard_two = blackboard.Client(name='v')
        blackboard_two.register_key(
            key='trace'+'v', access=common.Access.WRITE)
        # bboard = blackboard.Client(name='gbt' + str(t))
        # bboard.register_key(key='trace', access=common.Access.WRITE)
        # print(blackboard_one, blackboard_two)
        blackboard_one.set('traceb', [
            {p: env.states[p] for p in ['a', 'b', 'c', 'd']}])

        for i in range(3):
            gbt.tick()
            trace.append(env.states)
        #     # Trace Stack logic
        #     if (
        #         (action_node_one.status != common.Status.INVALID) and (
        #             one_idx == action_node_one.index)):
        #         try:
        #             bboard.trace[one_idx].update(
        #                 {p: env.states[p] for p in ['a', 'b', 'c', 'd']})
        #         except IndexError:
        #             bboard.trace.append(dict())
        #             bboard.trace[one_idx] = {
        #                 p: env.states[p] for p in ['a', 'b', 'c', 'd']}
        #         one_idx += 1
        #     if (
        #         (action_node_two.status != common.Status.INVALID) and (
        #             two_idx == action_node_two.index-1)):
        #         try:
        #             bboard.trace.append(dict())
        #             bboard.trace[two_idx].update(
        #                 {p: env.states[p] for p in ['w', 'v', 'x', 'y']})
        #         except IndexError:
        #             bboard.trace[two_idx] = {
        #                 p: env.states[p] for p in ['w', 'v', 'x', 'y']}
        #         two_idx += 1
            if gbt.root.status in [
                    common.Status.SUCCESS, common.Status.FAILURE]:
                break
        # print(t, bboard.trace)
        if (ppatask_two.status is not common.Status.INVALID):
            try:
                blackboard_two.get('tracev')
            except KeyError:
                blackboard_two.set('tracev', [
                    {p: env.states[p] for p in ['w', 'v', 'x', 'y']}])
        try:
            main_trace = stack_trace(
                blackboard_one.get('traceb'), blackboard_two.get('tracev'))
        except KeyError:
            main_trace = blackboard_one.get('traceb')
        # print(t, main_trace)
        # print(t, trace)
        # print(t, 'one', blackboard_one.get('traceb'))
        try:
            # print(t, 'two', blackboard_two.get('tracev'))
            blackboard_two.get('tracev')
        except KeyError:
            pass
        bt_status = True if gbt.root.status == common.Status.SUCCESS else False
        ltlf_parse_status = mission_ppa_formual.truth(main_trace, 0)
        # print(t, ltlf_parse_status, gbt.root.status, bt_status)
        # print("\n")
        # print(
        #     t, task_one_forumal.truth(main_trace, 0),
        #     task_two_forumal.truth(main_trace, 0))
        if len(main_trace) > 1:
            assert ltlf_parse_status == bt_status
        blackboard_one.unset('traceb')
        blackboard_two.unset('tracev')
    print(t)


def test_mission_finally_until():
    formula_one = '(G (d) & b) | ((G(d) & a) & (c U (b & G(d))))'
    formula_two = '(G (y) & v) | ((G(y) & w) & (x U (v & G(y))))'
    # mission = '(F (b)) U (F (v))'
    mission = 'b U (F(v))'
    parser = LTLfParser()
    mission_formula = parser(mission)
    mission_ppa = '(' + formula_one + ') U (F(' + formula_two + '))'
    mparser = LTLfParser()
    mission_ppa_formual = mparser(mission_ppa)
    # print(mission_ppa_formual)
    parser_task_one = LTLfParser()
    parser_task_two = LTLfParser()
    task_one_forumal = parser_task_one(formula_one)
    task_two_forumal = parser_task_two(formula_two)
    for t in range(10000):
        # env_one = Env(['a', 'b', 'c', 'd'])
        env = Env()
        trace = [env.states]
        action_node_one = ActionNodeBoard(
            'b', env, task_max=5, props=['a', 'b', 'c', 'd'])
        action_node_two = ActionNodeBoard(
            'v', env, task_max=5, props=['w', 'v', 'x', 'y'])
        ppatask_one = create_PPATask_GBT('a', 'b', 'c', 'd', action_node_one)
        ppatask_two = create_PPATask_GBT('w', 'v', 'x', 'y', action_node_two)
        mappings = {'b': ppatask_one, 'v': ppatask_two}
        gbt = parse_ltlf(mission_formula, mappings)
        gbt = BehaviourTree(gbt)
        blackboard_one = blackboard.Client(name='b')
        blackboard_one.register_key(
            key='trace'+'b', access=common.Access.WRITE)
        blackboard_two = blackboard.Client(name='v')
        blackboard_two.register_key(
            key='trace'+'v', access=common.Access.WRITE)
        # bboard = blackboard.Client(name='gbt' + str(t))
        # bboard.register_key(key='trace', access=common.Access.WRITE)
        # print(blackboard_one, blackboard_two)
        blackboard_one.set('traceb', [
            {p: env.states[p] for p in ['a', 'b', 'c', 'd']}])

        for i in range(10):
            gbt.tick()
            trace.append(env.states)
            if gbt.root.status in [
                    common.Status.SUCCESS, common.Status.FAILURE]:
                break
        # print(t, bboard.trace)
        if (ppatask_two.status is not common.Status.INVALID):
            try:
                blackboard_two.get('tracev')
            except KeyError:
                blackboard_two.set('tracev', [
                    {p: env.states[p] for p in ['w', 'v', 'x', 'y']}])
        try:
            main_trace = stack_trace(
                blackboard_one.get('traceb'), blackboard_two.get('tracev'))
        except KeyError:
            main_trace = blackboard_one.get('traceb')
        # print(t, main_trace)
        # print(t, trace)
        # print(t, 'one', blackboard_one.get('traceb'))
        try:
            # print(t, 'two', blackboard_two.get('tracev'))
            blackboard_two.get('tracev')
        except KeyError:
            pass
        bt_status = True if gbt.root.status == common.Status.SUCCESS else False
        ltlf_parse_status = mission_ppa_formual.truth(main_trace, 0)
        # print(t, ltlf_parse_status, gbt.root.status, bt_status)
        # print("\n")
        # print(
        #     t, task_one_forumal.truth(main_trace, 0),
        #     task_two_forumal.truth(main_trace, 0))
        if len(main_trace) > 1:
            assert ltlf_parse_status == bt_status
        blackboard_one.unset('traceb')
        blackboard_two.unset('tracev')
    print(t)


def random_test():
    mission = 'b U v'
    parser = LTLfParser()
    mission_formula = parser(mission)
    trace = [{'b': False}]
    print(mission_formula.truth(trace, 0))


def main():
    # test_PPATASK()
    # test_mission_until()
    # random_test()
    test_mission_finally_until()


if __name__ == '__main__':
    main()