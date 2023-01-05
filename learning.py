from mdp import (
    GridMDP, create_policy,
    policy_iteration, policy_test,
    policy_test_step, random_policy)
from gbtnodes import create_PPATask_GBT_learn

from py_trees import common, blackboard
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
import py_trees
import numpy as np
import pickle


class MDPActionNode(Behaviour):
    """Action node for the planning atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the action node for the planning LTLf propositions.
    """

    def __init__(
            self, name, env, planner=None, max_task=20,
            actions=[0, 1, 2, 3], seed=None):
        """Init method for the action node."""
        super(MDPActionNode, self).__init__('Action'+name)
        self.action_symbol = name
        self.blackboard = blackboard.Client(name='gbt')
        self.blackboard.register_key(key='trace', access=common.Access.WRITE)
        self.env = env
        self.gtable = planner
        self.index = 0
        self.task_max = max_task
        self.curr_loc = env.curr_loc
        self.actionsidx = actions
        if seed is None:
            self.nprandom = np.random.RandomState()   # pylint: disable=E1101
        else:
            self.nprandom = np.random.RandomState(    # pylint: disable=E1101
                seed)

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

    def create_gtable_indv(self, state):
        p = np.ones(len(self.actionsidx), dtype=np.float64)
        p = p / (len(self.actionsidx) * 1.0)

        self.gtable[state] = dict(
                        zip(self.actionsidx, p))

    def get_action(self, state):
        return self.nprandom.choice(
            self.actionsidx,
            p=list(self.gtable[state].values())
            )

    def env_action_dict(self, action):
        action_dict = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1)
        }
        return action_dict[action]

    def update(self):
        """
        Main function that is called when BT ticks.
        """
        # Plan action and take that action in the environment.
        # print('action node',self.index, self.blackboard.trace[-1])
        # while True:
        #     curr_loc, curr_reward, done, _ = mdp.step(pi[curr_loc])
        #     print(_)
        #     trace.append(curr_loc)
        #     if done:
        #         break
        # return trace
        # print('Planner', self.planner)
        state = self.curr_loc
        if self.gtable.get(state, None) is None:
            self.create_gtable_indv(state)
        action = self.get_action(state)

        curr_loc, curr_reward, done, state = self.env.step(
            self.env_action_dict(action))
        self.index += 1
        self.blackboard.trace[-1]['action'] = action
        # print('action node',self.index, self.task_max, self.blackboard.trace[-1])
        self.blackboard.trace.append(state)
        self.curr_loc = curr_loc
        curr_symbol_truth_value = self.blackboard.trace[-1][self.action_symbol]
        if curr_symbol_truth_value and self.index <= self.task_max:
            return common.Status.SUCCESS
        elif curr_symbol_truth_value == False and self.index < self.task_max:
            return common.Status.RUNNING
        else:
            return common.Status.FAILURE


def init_mdp(
        sloc=(3,0), reward = [-0.04, 2, -2],
        uncertainty=(0.9,0.05,0.05)):
    """Initialized a simple MDP world."""
    grid = np.ones((4, 4)) * reward[0]

    # # Obstacles
    # grid[3][0] = None
    # grid[2][2] = None
    # grid[1][1] = None

    # Cheese and trap
    grid[0][3] = None   # Cheese
    grid[1][3] = None   # Trap

    grid = np.where(np.isnan(grid), None, grid)
    grid = grid.tolist()

    # Terminal and obstacles defination
    grid[0][3] = reward[1]
    grid[1][3] = reward[2]

    mdp = GridMDP(
        grid, terminals=[(3,3), (3,2)], startloc=sloc,
        uncertainty=uncertainty
        )

    return mdp


def run_experiment(reward, uncertainty, runs, maxtrace=30):
    env = init_mdp(reward=reward, uncertainty=uncertainty)
    # policy = policy_iteration(env)
    # policy = random_policy()
    policy = dict()
    # env.display_in_grid(policy)
    results = []
    for k in range(runs):
        # results.append(policy_test(policy, env))
        # print(policy_test_step(policy, env))
        env.restart()
        bboard = blackboard.Client(name='gbt')
        bboard.register_key(key='trace', access=common.Access.WRITE)
        bboard.trace = [env.get_states()]
        # print(k, policy)
        policynode = MDPActionNode('c', env, policy, maxtrace)
        ppataskbt = create_PPATask_GBT_learn('p', 'c', 't', 'g', policynode)
        ppataskbt = BehaviourTree(ppataskbt)
        # print(py_trees.display.ascii_tree(ppataskbt.root))
        # add debug statement
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        for i in range(maxtrace):
            ppataskbt.tick()
            # print(bboard.trace, ppataskbt.root.status)
            if (
                (ppataskbt.root.status == common.Status.SUCCESS) or
                    (ppataskbt.root.status == common.Status.FAILURE)):
                break
        results.append([bboard.trace, ppataskbt.root.status])
    print(k, [state['state'] for state in bboard.trace])

    return results, policy


def experiments_parameters():
    discounts = [
        (-0.04, 2, -2), (-0.1, 2, -2),
        (-0.5, 2, -2), (-1, 2, -2),
        (-1.5, 2, -2), (-0.04, 5, -2),
        (-0.04, 10, -2), (-0.04, 1, -2),
        (-0.04, 0.5, -2), (-0.04, 0.1, -2),
        (-0.04, 2, -5), (-0.04, 2, -10),
        (-0.04, 2, -1), (-0.04, 2, -0.5),
        (-0.04, 2, -0.1), (-0.04, 5, -5),
        ]
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        (0.7, 0.15, 0.15), (0.6, 0.2, 0.2),
        (0.5, 0.25, 0.25), (0.4, 0.3, 0.3),
        ]
    discounts = [(-0.04, 2, -2)]
    uncertainties = [(0.95, 0.025, 0.025)]
    runs = 50
    results = dict()
    for discount in discounts:
        results[discount] = dict()
        for uc in uncertainties:
            results[discount][uc] = dict()
            res, policy = run_experiment(discount, uc, runs)
            results[discount][uc]['result'] = res
            results[discount][uc]['policy'] = policy

    # with open('/tmp/learning_30.pickle', 'wb') as file:
    #     pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('/tmp/learning_30.pickle', 'rb') as file:
    #     data = pickle.load(file)
    # print('Experiment Done')


def main():
    experiments_parameters()


if __name__ =='__main__':
    main()