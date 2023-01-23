from inspect import trace
from mdp import (
    GridMDP, create_policy,
    policy_iteration, policy_test,
    random_policy)
from gbtnodes import create_PPATask_GBT_learn, create_PPATask_GBT

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
            actions=[0, 1, 2, 3], discount=0.9, seed=None):
        """Init method for the action node."""
        super(MDPActionNode, self).__init__('Action'+name)
        self.action_symbol = name
        self.blackboard = blackboard.Client(name='gbt')
        self.blackboard.register_key(key='trace', access=common.Access.WRITE)
        self.tkey = 'trace' +name
        self.blackboard.register_key(key=self.tkey, access=common.Access.WRITE)
        self.blackboard.set(self.tkey, [])
        self.env = env
        self.gtable = planner
        self.index = 0
        self.task_max = max_task
        self.curr_loc = env.curr_loc
        self.actionsidx = actions
        self.discount = discount
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
        # print('different notation', self.tkey, state, self.blackboard.trace[-1])
        self.blackboard.set(
            self.tkey, self.blackboard.get(self.tkey) + [self.blackboard.trace[-1]])
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
        uncertainty=(0.9,0.05,0.05), random=True):
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

    if random:
        while True:
            randloc = tuple(np.random.randint(0, 4, 2).tolist())
            if randloc not in [(3,3), [3,2]]:
                break
        sloc = randloc if random else sloc

    mdp = GridMDP(
        grid, terminals=[(3,3), (3,2)], startloc=sloc,
        uncertainty=uncertainty
        )

    return mdp


def run_experiment(
        reward, uncertainty, runs=10,
        maxtrace=30, propsteps=30, discount=0.9, random=True):
    env = init_mdp(
        reward=reward, uncertainty=uncertainty, random=random)
    # policy = policy_iteration(env)
    # policy = random_policy()
    # env.display_in_grid(policy)
    policies = []
    results = []
    for l in range(runs):
        result = []
        policy = dict()
        for k in range(propsteps):
            # results.append(policy_test(policy, env))
            # print(policy_test_step(policy, env))
            env.restart(random=random)
            bboard = blackboard.Client(name='gbt')
            bboard.register_key(key='trace', access=common.Access.WRITE)
            bboard.trace = [env.get_states()]
            # print(k, policy)
            policynode = MDPActionNode(
                'c', env, policy, maxtrace, discount=discount)
            ppataskbt = create_PPATask_GBT_learn('p', 'c', 't', 'g', policynode)
            ppataskbt = BehaviourTree(ppataskbt)
            # print(py_trees.display.ascii_tree(ppataskbt.root))
            # add debug statement
            # py_trees.logging.level = py_trees.logging.Level.DEBUG
            for i in range(maxtrace):
                ppataskbt.tick()
                # print(bboard.trace[-1], ppataskbt.root.status)
                if (
                    (ppataskbt.root.status == common.Status.SUCCESS) or
                        (ppataskbt.root.status == common.Status.FAILURE)):
                    break
            result.append([bboard.trace, ppataskbt.root.status])
        results.append(result)
        policies.append(policy)
    # print(len(result), len(results))
    # print(l, k, [state['state'] for state in bboard.trace])
    return results, policies


def policy_test_step(pi, mdp, max_trace=29):
    curr_loc = mdp.curr_loc
    trace = [curr_loc]
    idx = 1
    result = False
    while True:
        try:
            curr_loc, curr_reward, done, _ = mdp.step(pi[curr_loc])
        except KeyError:
            result = False
            break
        # print(_)
        trace.append(curr_loc)
        if done and curr_loc == (3,3):
            result = True
            break
        elif done and curr_loc == (3,2):
            result = False
            break
        else:
            pass
        if idx >= max_trace:
            result = False
            break
        idx += 1
    return result, trace


def run_experiment_given_policy(
        policy, uncertainty, runs=10,
        maxtrace=30, random=True):
    env = init_mdp(
        reward=[0.2, 2, 2], uncertainty=uncertainty, random=random)
    # policy = policy_iteration(env)
    # policy = random_policy()
    # env.display_in_grid(policy)
    action_dict = {
        0: (1, 0),
        1: (0, 1),
        2: (-1, 0),
        3: (0, -1)
    }
    policy = {state:action_dict[np.argmax(actions)] for state,actions in policy.items()}
    results = []
    for l in range(runs):
        env.restart(random=random)
        result, trace = policy_test_step(policy, env)
        print(result, trace)
        results.append([result, trace])

    return results


def experiments_parameters():
    rewards = [
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
        # (0.7, 0.15, 0.15), (0.6, 0.2, 0.2),
        # (0.5, 0.25, 0.25), (0.4, 0.3, 0.3),
        ]
    discounts = [
        0.99, 0.95, 0.9, 0.85, 0.8, 0.7,
        0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    tracelens = [10, 15, 20, 25, 30, 40, 50]
    propsteps = [10, 15, 20, 25, 30, 40, 50]

    # uncertainties = [(0.95, 0.025, 0.025)]
    # discounts = [0.9]
    # rewards = [(-0.04, 2, -2)]
    # tracelens = [30]
    # propsteps = [50]
    random = False
    runs = 50
    results = dict()
    j = 0
    for discount in discounts:
        results[discount] = dict()
        for uc in uncertainties:
            results[discount][uc] = dict()
            for tlen in tracelens:
                results[discount][uc][tlen] = dict()
                for pstep in propsteps:
                    results[discount][uc][tlen][pstep] = dict()
                    res, policy = run_experiment(
                        rewards[j], uc, runs, maxtrace=tlen,
                        propsteps=pstep, discount=discount, random=random)
                    results[discount][uc][tlen][pstep]['result'] = res
                    results[discount][uc][tlen][pstep]['policy'] = policy
        j += 1

    with open('/tmp/learning_30_all.pickle', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('/tmp/learning_30_all.pickle', 'rb') as file:
        data = pickle.load(file)
    print('Experiment Done', len(data))


def run_experiment_with_random_loc():
    runs = 50
    tlen = 15
    pstep = 25
    discount = 0.7
    rewards = [(-0.04, 2, -2)]
    uncertainties = [(0.95, 0.025, 0.025)]
    res, policy = run_experiment(
        rewards[0], uncertainties[0], runs, maxtrace=tlen,
        propsteps=pstep, discount=discount,random=False)

    results = []
    for p in policy:
        results.append(
            run_experiment_given_policy(p, uncertainties[0], 30, True))

    with open('/tmp/resilence_test_randomstart.pickle', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('/tmp/resilence_test_randomstart.pickle', 'rb') as file:
        data = pickle.load(file)

    print('Experiment Done', len(data))


def main():
    # experiments_parameters()
    run_experiment_with_random_loc()


if __name__ =='__main__':
    main()