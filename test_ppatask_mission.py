from gbtnodes import (
    create_PPATask_GBT, ActionNode, parse_ltlf,
    replace_dummynodes_with_PPATaskBT)

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


    for trace_func,status in trace_functions.items():
        env = setup_environment(trace_func)
        action_node = ActionNode('b', env)
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


def mission():
    mission = '(F c) & (F h)'
    parser = LTLfParser()
    mission_formula = parser(mission)
    print(mission_formula)
    actionnodec = ActionNode('c', None, None)
    actionnodeh = ActionNode('h', None, None)
    mappings = {'c':actionnodec, 'h':actionnodeh}

    gbt = parse_ltlf(mission_formula, mappings)
    gbt = BehaviourTree(gbt)
    print(py_trees.display.ascii_tree(gbt.root))
    # print(dir(gbt.root))
    print(py_trees.display.ascii_tree(gbt.root))


def main():
    # ppatask()
    mission()


if __name__ == "__main__":
    main()