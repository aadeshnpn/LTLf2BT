from gbtnodes import (
    create_PPATask_GBT, ActionNode, parse_ltlf)

from py_trees import common, blackboard
from py_trees.trees import BehaviourTree
import py_trees
from flloat.parser.ltlf import LTLfParser


class Env:
    def __init__(self, trace):
        self.trace = trace
        self.curr_state = trace[0]
        self.idx = 0

    def step(self):
        self.idx += 1
        self.curr_state = self.trace[self.idx]

    def reset(self):
        self.idx = 0


def get_trace_postcond_true():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': True, 'c': True, 'd': True}
    ]
    return trace

def get_trace_postcond_false():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': False, 'c': True, 'd': True}
    ]
    return trace


def get_trace_precond_0_false():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': True, 'c': True, 'd': True}
    ]
    return trace


def get_trace_precond_true():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': True, 'c': True, 'd': True}
    ]
    return trace


def get_trace_glob_false():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': False},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': True, 'c': True, 'd': True}
    ]
    return trace


def get_trace_task_0_false():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': True, 'b': False, 'c': False, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': True, 'c': True, 'd': True}
    ]
    return trace


def get_trace_task_01_false():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': True, 'b': False, 'c': False, 'd': True},
        {'a': False, 'b': False, 'c': False, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': True, 'c': True, 'd': True}
    ]
    return trace


def get_trace_task_0_true():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': True, 'b': True, 'c': False, 'd': True},
        {'a': False, 'b': False, 'c': False, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': True, 'c': True, 'd': True}
    ]
    return trace


def get_trace_task_3_true():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': True, 'c': False, 'd': True}
    ]
    return trace


def get_trace_action_3_false():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True}
    ]
    return trace


def get_trace_action_3_glob_false():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': False, 'c': True, 'd': True},
        {'a': False, 'b': True, 'c': True, 'd': False}
    ]
    return trace


def get_trace_both_postcond_true():
    trace = [
        {'p': True, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h':False},

        {'p': False, 'c': True, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h':True}
    ]
    return trace


def get_trace_home_postcond_false():
    trace = [
        {'p': True, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h':False},

        {'p': False, 'c': True, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h':False}
    ]
    return trace


def get_trace_cheese_postcond_false():
    trace = [
        {'p': True, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h':False},

        {'p': False, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h':False}
    ]
    return trace


def get_trace_cheese_precondition_true():
    trace = [
        {'p': True, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': False, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h':False},

        {'p': False, 'c': True, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h': False},
        {'p': False, 'c': True, 'a': True, 't': True, 'h': True}
    ]
    return trace


def setup_environment(trace_func):
    env = Env(trace_func())
    bboard = blackboard.Client(name='gbt')
    bboard.register_key(key='trace', access=common.Access.WRITE)
    bboard.trace = [env.curr_state]
    # action_node = ActionNode('b', env)
    return env


def ppatask():
    trace_functions = {
        get_trace_postcond_true: common.Status.SUCCESS,
        get_trace_postcond_false: common.Status.FAILURE,
        get_trace_precond_0_false: common.Status.FAILURE,
        get_trace_precond_true: common.Status.SUCCESS,
        get_trace_glob_false: common.Status.FAILURE,
        get_trace_task_0_false: common.Status.FAILURE,
        get_trace_task_01_false: common.Status.FAILURE,
        get_trace_task_0_true: common.Status.SUCCESS,
        get_trace_task_3_true: common.Status.SUCCESS,
        get_trace_action_3_false: common.Status.FAILURE,
        get_trace_action_3_glob_false: common.Status.FAILURE,
        }
    real_ltl_truth_value = [True, False, False, True, False, False, False, True, True, False, False]
    formula = '(G (d) & b) | ((G(d) & a) & (c U (b & G(d))))'
    parser = LTLfParser()
    task_formula = parser(formula)
    j = 0
    for trace_func,status in trace_functions.items():
        env = setup_environment(trace_func)
        action_node = ActionNode('b', env, task_max=3)
        ppataskbt = create_PPATask_GBT('a', 'b', 'c', 'd', action_node)
        ppataskbt = BehaviourTree(ppataskbt)
        # print(py_trees.display.ascii_tree(ppataskbt.root))
        # add debug statement
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        for i in range(3):
            # print(i, env.trace[i], ppataskbt.root.status)
            ppataskbt.tick()
        print(trace_func.__name__, ppataskbt.root.status, status)
        assert ppataskbt.root.status ==status, "Failed"
        assert task_formula.truth(
            trace_func(), 0) == real_ltl_truth_value[j], "Failed Real LTL Parser"
        j += 1


def mission():
    # mission = '(F c) U (F h)'
    mission = '(F (c)) U (F (h))'
    chess_task = '(G (t) & c) | ((G(t) & p) & (a U (c & G(t))))'
    parser = LTLfParser()
    cheese_formula = parser(chess_task)
    home_task = '(G (t) & h) | ((G(t) & c) & (a U (h & G(t))))'
    # home_task_finally = 'F (G (t) & h) | ((G(t) & c) & (a U (h & G(t))))'
    home_formula = parser(home_task)
    mission_in_ppatask = '(F(' +chess_task + ')) U (F(' + home_task + '))'
    parser = LTLfParser()
    mission_formula = parser(mission)
    mparser = LTLfParser()
    mission_formula_ppa = mparser(mission_in_ppatask)
    # print(mission_formula_ppa)

    trace_functions = {
        get_trace_both_postcond_true: common.Status.SUCCESS,
        get_trace_home_postcond_false: common.Status.FAILURE,
        get_trace_cheese_postcond_false: common.Status.FAILURE,
        get_trace_cheese_precondition_true: common.Status.SUCCESS
        }

    real_ltl_truth_value =[True, False, False, True]

    j = 0
    for trace_func, status in trace_functions.items():
        env = setup_environment(trace_func)
        action_nodec = ActionNode('c', env, None, task_max=4)
        action_nodeh = ActionNode('h', env, None, task_max=4)
        ppataskc = create_PPATask_GBT('p','c','a','t', action_nodec)
        ppataskh = create_PPATask_GBT('c','h','a','t', action_nodeh)
        mappings = {'c':ppataskc, 'h':ppataskh}
        gbt = parse_ltlf(mission_formula, mappings)
        gbt = BehaviourTree(gbt)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # print(py_trees.display.ascii_tree(gbt.root))
        for i in range(8):
            gbt.tick()
            # print(trace_func.__name__, gbt.root.status, status)
            if gbt.root.status == common.Status.SUCCESS:
                break
        print(trace_func.__name__, gbt.root.status, status)
        assert gbt.root.status ==status, "Failed"
        # print(trace_func())
        assert mission_formula_ppa.truth(trace_func(), 0) == real_ltl_truth_value[j], "Failed"
        # assert cheese_formula.truth(trace_func(), 0) == real_ltl_truth_value[j], "Failed"
        # assert home_formula.truth(trace_func(), 3) == real_ltl_truth_value[j], "Failed"
        j += 1


def main():
    # ppatask()
    mission()


if __name__ == "__main__":
    main()