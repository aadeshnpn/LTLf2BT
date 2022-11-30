from gbtnodes import create_PPATask_GBT,ActionNode
from py_trees.trees import BehaviourTree
import py_trees


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


def get_trace_1():
    # pre-condition: a
    # post-condition: b
    # task-constaint: c
    # global-constaraint: d
    trace = [
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': False, 'c': True, 'd': True},
        {'a': True, 'b': True, 'c': True, 'd': True}
    ]
    return trace


def main():
    env = Env(get_trace_1())
    action_node = ActionNode('b', env)
    ppataskbt = create_PPATask_GBT('a', 'b', 'c', 'd', action_node)
    ppataskbt = BehaviourTree(ppataskbt)
    print(py_trees.display.ascii_tree(ppataskbt.root))


if __name__ == "__main__":
    main()