from gbtnodes import create_PPATask_GBT,ActionNode
from py_trees import common, blackboard
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
    bboard = blackboard.Client(name='gbt')
    bboard.register_key(key='trace', access=common.Access.WRITE)
    bboard.trace = [env.curr_state]
    action_node = ActionNode('b', env)
    ppataskbt = create_PPATask_GBT('a', 'b', 'c', 'd', action_node)
    ppataskbt = BehaviourTree(ppataskbt)
    print(py_trees.display.ascii_tree(ppataskbt.root))
    for i in range(3):
        ppataskbt.tick()
        print(i, ppataskbt.root.status)


if __name__ == "__main__":
    main()