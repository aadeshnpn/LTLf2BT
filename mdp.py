"""Markov Decision Process class."""

import numpy as np
import operator
import collections
import random
import copy


# MDP
def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))


orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
turns = LEFT, RIGHT = (+1, -1)


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    return turn_heading(heading, LEFT)


argmax = max


def isnumber(x):
    """Is x a number?"""
    return hasattr(x, '__int__')


def issequence(x):
    """Is x a sequence?"""
    return isinstance(x, collections.abc.Sequence)


def print_table(table, header=None, sep='   ', numfmt='{}'):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '{:.2f}'.
    (If you want different formats in different columns,
    don't use print_table.) sep is the separator between columns."""
    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]

    if header:
        table.insert(0, header)

    table = [[numfmt.format(x) if isnumber(x) else x for x in row]
             for row in table]

    sizes = list(
        map(lambda seq: max(map(len, seq)),
            list(zip(*[map(str, row) for row in table]))))

    for row in table:
        print(sep.join(getattr(
            str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))


class MDP:

    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text. Instead of P(s' | s, a) being a probability number for each
    state/state/action triplet, we instead have T(s, a) return a
    list of (p, s') pairs. We also keep track of the possible states,
    terminal states, and actions for each state. [page 646]"""

    def __init__(
        self, init, actlist, terminals,
            transitions=None, reward=None, states=None, gamma=0.99):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        # collect states from transitions table if not passed.
        self.states = states or self.get_states_from_transitions(transitions)

        self.init = init

        if isinstance(actlist, list):
            # if actlist is a list, all states have the same actions
            self.actlist = actlist

        elif isinstance(actlist, dict):
            # if actlist is a dict, different actions for each state
            self.actlist = actlist

        self.terminals = terminals
        self.transitions = transitions or {}
        if not self.transitions:
            print("Warning: Transition table is empty.")

        self.gamma = gamma

        self.reward = reward or {s: 0 for s in self.states}

        # self.check_consistency()

    def R(self, state):
        """Return a numeric reward for this state."""

        return self.reward[state]

    def T(self, state, action):
        """Transition model. From a state and an action, return a list
        of (probability, result-state) pairs."""

        if not self.transitions:
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    def actions(self, state):
        """Return a list of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""

        if state in self.terminals:
            return [None]
        else:
            return self.actlist

    def get_states_from_transitions(self, transitions):
        if isinstance(transitions, dict):
            s1 = set(transitions.keys())
            s2 = set(tr[1] for actions in transitions.values()
                     for effects in actions.values()
                     for tr in effects)
            return s1.union(s2)
        else:
            print('Could not retrieve states from transitions')
            return None

    def check_consistency(self):

        # check that all states in transitions are valid
        assert set(self.states) == self.get_states_from_transitions(
            self.transitions)

        # check that init is a valid state
        assert self.init in self.states

        # check reward for each state
        assert set(self.reward.keys()) == set(self.states)

        # check that all terminals are valid states
        assert all(t in self.states for t in self.terminals)

        # check that probability distributions for all actions sum to 1
        for s1, actions in self.transitions.items():
            for a in actions.keys():
                s = 0
                for o in actions[a]:
                    s += o[0]
                assert abs(s - 1) < 0.001


class GridMDP(MDP):

    """A two-dimensional grid MDP. All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state). Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(
            self, grid, terminals, init=(0, 0),
            gamma=.9, startloc=(3, 0), seed=None,
            uncertainty=(0.8,0.1,0.1)):
        grid.reverse()     # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        if seed is None:
            self.nprandom = np.random.RandomState()  # pylint: disable=E1101
        else:
            self.nprandom = np.random.RandomState(   # pylint: disable=E1101
                seed)
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        self.uncertainty = uncertainty
        actlist = orientations
        transitions = {}
        # Goal related
        # self.goalspec = goalspec
        self.startloc = startloc
        self.curr_loc = self.startloc
        self.cheese_memory = False

        if len(terminals) >= 2:
            self.cheese = terminals[0]
            self.trap = terminals[1]
        else:
            self.cheese = None
            self.trap = None

        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a)
        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, transitions=transitions,
                     reward=reward, states=states, gamma=gamma)

        self.action_dict = {
            'S': (1, 0),
            'E': (0, 1),
            'N': (-1, 0),
            'W': (0, -1)
        }
        self.env_action_dict = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1)
        }
        self.state_dict = dict()
        self.curr_reward = self.reward[self.startloc]

    def check_near_object(self, state, name='cheese'):
        states = [vector_add(
            state, action) for action in self.action_dict.values()]
        if name == 'cheese':
            if self.cheese in states:
                return True
            else:
                return False
        elif name == 'trap':
            if self.trap in states:
                return True
            else:
                return False
        else:
            return False

    # fp=0.95, lp=0.025, rp=0.025
    def calculate_T(self, state, action, fp=0.8, lp=0.1, rp=0.1, bp=0.0):
        fp = self.uncertainty[0]
        lp = self.uncertainty[1]
        rp = self.uncertainty[2]
        if action:
            return [(fp, self.go(state, action)),
                    (rp, self.go(state, turn_right(action))),
                    (lp, self.go(state, turn_left(action)))]
        else:
            return [(0.0, state)]

    def get_states(self):
        # (3,3), (3,2)
        state = {
            'c':False, 'g':True,
            'p': True, 't':True, 'h':False,
            'state':self.curr_loc}
        if self.curr_loc == (3,3):
            state['c'] = True
            self.cheese_memory = True
        elif self.curr_loc == (3,2):
            state['g'] = False
        elif self.curr_loc == (3,0):
            state['h'] = True
        # Remember once cheese is picked
        if self.cheese_memory:
            state['c'] = True
        return state

    def step(self, action):
        # print(self.curr_loc)
        done = False
        if self.curr_loc in self.terminals:
            done = True
            self.curr_reward = self.reward[self.curr_loc]
            return self.curr_loc, self.curr_reward, done, self.get_states()
        p, s1 = zip(*self.T(self.curr_loc, action))
        p, s1 = list(p), list(s1)
        indx = list(range(len(s1)))
        # print(action, p, s1, self.curr_loc)
        indx = self.nprandom.choice(indx, p=p)
        next_state = s1[indx]
        self.curr_loc = next_state
        if self.curr_loc in self.terminals:
            done = True
        self.curr_reward = self.reward[next_state]
        return self.curr_loc, self.curr_reward, done, self.get_states()

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""

        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""

        return list(reversed([[mapping.get((x, y), 'W')
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def restart(self, random=False):
        if random:
            while True:
                randloc = tuple(np.random.randint(0, 4, 2).tolist())
                if randloc not in [(3,3), [3,2]]:
                    break
            self.curr_loc = randloc
            self.curr_reward = self.reward[randloc]
        else:
            self.curr_loc = self.startloc
            self.curr_reward = self.reward[self.startloc]
        self.cheese_memory = False

    def to_arrows(self, policy):
        # EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        # Right, Up, Left, Down
        chars = {
            (1, 0): '>', (0, 1): '^',
            (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})

    def display_in_grid(self, policy):
        array = self.to_arrows(policy)
        print('\n'.join([''.join(['{:4}'.format(item) for item in row])
            for row in array]))


    def qlearning(self, epoch):
        # qtable = np.zeros((len(self.states), 4))
        qtable = dict()
        for state in self.states:
            qtable[state] = dict(zip(orientations, [0, 0, 0, 0]))
        alpha = 0.1
        gamma = 0.7
        epsilon = 0.1
        # slookup = dict(zip(self.states, range(len(self.states))))
        for e in range(epoch):
            reward = 0
            state = self.startloc
            while True:
                if random.uniform(0, 1) < epsilon:
                    action = np.random.choice([0, 1, 2, 3])
                    action = orientations[action]
                else:
                    action = dictmax(qtable[state], s='key')
                p, s1 = zip(*self.T(state, action))
                p, s1 = list(p), list(s1)
                s1 = s1[np.argmax(p)]
                next_state = s1
                reward = self.R(next_state)
                if action is None:
                    break
                old_value = qtable[state][action]
                next_max = dictmax(qtable[next_state], s='val')
                new_value = (1-alpha) * old_value + alpha * (
                    reward + gamma * next_max)
                qtable[state][action] = new_value
                state = next_state
                if state in self.terminals:
                    break
        return qtable

    def format_state(self, state):
        state_val = str(state[0]) + str(state[1])
        self.state_dict[state_val] = state
        return state_val

    def format_action(self, action):
        action_dict = {
            (1, 0): 'S',
            (0, 1): 'E',
            (-1, 0): 'N',
            (0, -1): 'W'
        }
        return action_dict[action]


class GridMDPModfy(MDP):

    def __init__(
            self, grid, terminals, init=(0, 0),
            gamma=.9, startloc=(3, 0), seed=None):

        grid.reverse()     # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        if seed is None:
            self.nprandom = np.random.RandomState()  # pylint: disable=E1101
        else:
            self.nprandom = np.random.RandomState(   # pylint: disable=E1101
                seed)
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        actlist = orientations
        transitions = {}
        # Goal related
        # self.goalspec = goalspec
        self.startloc = startloc
        self.curr_loc = self.startloc

        if len(terminals) >= 2:
            self.cheese = terminals[0]
            self.trap = terminals[1]
        else:
            self.cheese = None
            self.trap = None

        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a)
        # print(reward)
        self.backup_reward = reward.copy()
        self.terminals = terminals
        MDP.__init__(self, init, actlist=actlist,
                     terminals=[terminals[1]], transitions=transitions,
                     reward=reward, states=states, gamma=gamma)
        self.reward_swap = False
        self.action_dict = {
            'S': (1, 0),
            'E': (0, 1),
            'N': (-1, 0),
            'W': (0, -1)
        }
        self.env_action_dict = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1)
        }
        self.state_dict = dict()
        self.curr_reward = self.reward[self.startloc]
        self.state_keyslist = ['s'+str(i)+str(j) for i in range(self.rows) for j in range(self.cols)]
        self.default_props = dict(zip(self.state_keyslist, [False] * len(self.state_keyslist)))

    def generate_default_props(self):
        props = copy.copy(self.default_props)
        props[self.get_state_keys(self.curr_loc)] = True

    def generate_props_loc(self, loc):
        props = copy.copy(self.default_props)
        props[self.get_state_keys(loc)] = True
        return {'s33': props['s33'], 's32': props['s32']}
        # return props

    def get_state_keys(self, loc):
        return  's' + str(loc[0])+str(loc[1])

    def R(self, state):
        """Return a numeric reward for this state."""
        if (state == self.cheese and self.reward_swap is False) :
            r = self.reward[state]
            self.reward[state] = self.reward[self.startloc]
            self.reward[self.startloc] = r
            self.reward_swap = True
            return r
        else:
            return self.reward[state]

    def calculate_T(self, state, action, fp=0.8, lp=0.1, rp=0.1, bp=0.0):
        if action:
            return [(fp, self.go(state, action)),
                    (rp, self.go(state, turn_right(action))),
                    (lp, self.go(state, turn_left(action)))]
        else:
            return [(0.0, state)]

    def step(self, action):
        # print(self.curr_loc)
        done = False
        if self.curr_loc in self.terminals:
            done = True
            self.curr_reward = self.R(self.curr_loc)
            return self.curr_loc, self.curr_reward, done, None
        p, s1 = zip(*self.T(self.curr_loc, action))
        p, s1 = list(p), list(s1)
        indx = list(range(len(s1)))
        indx = self.nprandom.choice(indx, p=p)
        next_state = s1[indx]
        self.curr_loc = next_state
        if self.startloc in self.terminals:
            done = True
        self.curr_reward = self.R(next_state)
        return self.curr_loc, self.curr_reward, done, None

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""

        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""

        return list(reversed([[mapping.get((x, y), 'W')
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def restart(self):
        self.curr_loc = self.startloc
        self.reward = self.backup_reward.copy()
        self.reward_swap = False
        # self.cheese = self.terminals[0]
        self.curr_reward = self.reward[self.startloc]

    def to_arrows(self, policy):
        # EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        # Right, Up, Left, Down
        chars = {
            (1, 0): '>', (0, 1): '^',
            (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})


class GridMDPModfySeq(MDP):

    def __init__(
            self, grid, terminals, init=(0, 0),
            gamma=.9, startloc=(3, 0), seed=None):

        grid.reverse()     # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        if seed is None:
            self.nprandom = np.random.RandomState()  # pylint: disable=E1101
        else:
            self.nprandom = np.random.RandomState(   # pylint: disable=E1101
                seed)
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        actlist = orientations
        transitions = {}
        # Goal related
        # self.goalspec = goalspec
        self.startloc = startloc
        self.curr_loc = self.startloc

        if len(terminals) >= 2:
            self.cheese = terminals[0]
            self.trap = terminals[1]
        else:
            self.cheese = None
            self.trap = None

        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a)
        # print(reward)
        self.backup_reward = reward.copy()
        self.terminals = terminals
        MDP.__init__(self, init, actlist=actlist,
                     terminals=[terminals[1]], transitions=transitions,
                     reward=reward, states=states, gamma=gamma)
        self.reward_swap = False
        self.action_dict = {
            'S': (1, 0),
            'E': (0, 1),
            'N': (-1, 0),
            'W': (0, -1)
        }
        self.env_action_dict = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1)
        }
        self.state_dict = dict()
        self.curr_reward = self.reward[self.startloc]
        self.state_keyslist = ['s'+str(i)+str(j) for i in range(self.rows) for j in range(self.cols)]
        home = 's'+ str(self.startloc[0]) + str(self.startloc[1])
        self.state_map = {'c': 's33', 't': 's32', 'h': home}
        self.default_props = dict(zip(self.state_keyslist, [False] * len(self.state_keyslist)))
        self.found_cheese = False
        # self.found_trap = False
        # self.found_home = False

    def update_props(self, props):
        if not self.found_cheese and props[self.state_map['c']]:
            self.found_cheese = True
        if self.found_cheese:
            props[self.state_map['c']] = True

        # if not self.found_trap and props[self.state_map['t']]:
        #     self.found_trap = True
        # if self.found_trap:
        #     props[self.state_map['t']] = True

        # if not self.found_home and props[self.state_map['h']]:
        #     self.found_home = True
        # if self.found_home:
        #     props[self.state_map['h']] = True

        return props

    def generate_default_props(self):
        props = copy.copy(self.default_props)
        props[self.get_state_keys(self.curr_loc)] = True
        props = self.update_props(props)
        return dict(zip(
                self.state_map.keys(),
                [props[v] for v in self.state_map.values()])
                )

    def generate_props_loc(self, loc):
        props = copy.copy(self.default_props)
        props[self.get_state_keys(loc)] = True
        props = self.update_props(props)
        return dict(zip(
                self.state_map.keys(),
                [props[v] for v in self.state_map.values()])
                )
        # return {'s33': props['s33'], 's32': props['s32']}
        # return props

    def get_state_keys(self, loc):
        return  's' + str(loc[0])+str(loc[1])

    def R(self, state):
        """Return a numeric reward for this state."""
        if (state == self.cheese and self.reward_swap is False) :
            r = self.reward[state]
            self.reward[state] = self.reward[self.startloc]
            self.reward[self.startloc] = r
            self.reward_swap = True
            return r
        else:
            return self.reward[state]

    def calculate_T(self, state, action, fp=0.8, lp=0.1, rp=0.1, bp=0.0):
        if action:
            return [(fp, self.go(state, action)),
                    (rp, self.go(state, turn_right(action))),
                    (lp, self.go(state, turn_left(action)))]
        else:
            return [(0.0, state)]

    def step(self, action):
        # print(self.curr_loc)
        done = False
        if self.curr_loc in self.terminals:
            done = True
            self.curr_reward = self.R(self.curr_loc)
            return self.curr_loc, self.curr_reward, done, None
        p, s1 = zip(*self.T(self.curr_loc, action))
        p, s1 = list(p), list(s1)
        indx = list(range(len(s1)))
        indx = self.nprandom.choice(indx, p=p)
        next_state = s1[indx]
        self.curr_loc = next_state
        if self.startloc in self.terminals:
            done = True
        self.curr_reward = self.R(next_state)
        return self.curr_loc, self.curr_reward, done, None

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""

        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""

        return list(reversed([[mapping.get((x, y), 'W')
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def restart(self):
        self.curr_loc = self.startloc
        self.reward = self.backup_reward.copy()
        self.reward_swap = False
        # self.cheese = self.terminals[0]
        self.curr_reward = self.reward[self.startloc]

    def to_arrows(self, policy):
        # EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        # Right, Up, Left, Down
        chars = {
            (1, 0): '>', (0, 1): '^',
            (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})


class GridMDPFire(MDP):

    def __init__(
            self, grid, terminals, init=(0, 0),
            gamma=.9, startloc=(3, 0), seed=None):

        grid.reverse()     # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        if seed is None:
            self.nprandom = np.random.RandomState()  # pylint: disable=E1101
        else:
            self.nprandom = np.random.RandomState(   # pylint: disable=E1101
                seed)
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        actlist = orientations
        transitions = {}
        # Goal related
        # self.goalspec = goalspec
        self.startloc = startloc
        self.curr_loc = self.startloc

        if len(terminals) >= 2:
            self.ext = terminals[0]
            self.fire = terminals[1]
        else:
            self.ext = None
            self.fire = None

        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a)
        # print(reward)
        self.backup_reward = reward.copy()
        self.terminals = terminals
        MDP.__init__(self, init, actlist=actlist,
                     terminals=[terminals[1]], transitions=transitions,
                     reward=reward, states=states, gamma=gamma)
        self.reward_swap = False
        self.action_dict = {
            'S': (1, 0),
            'E': (0, 1),
            'N': (-1, 0),
            'W': (0, -1)
        }
        self.env_action_dict = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1)
        }
        self.state_dict = dict()
        self.curr_reward = self.reward[self.startloc]
        self.state_keyslist = ['s'+str(i)+str(j) for i in range(self.rows) for j in range(self.cols)]
        home = 's'+ str(self.startloc[0]) + str(self.startloc[1])
        self.state_map = {'e': 's33', 'f': 's31'}
        self.default_props = dict(zip(self.state_keyslist, [False] * len(self.state_keyslist)))
        self.carrying_ext = False
        # self.fire_ext = False

    def update_props(self, props):
        if not self.carrying_ext and props[self.state_map['e']]:
            self.carrying_ext = True
            props[self.state_map['e']] = True
        if self.carrying_ext:
            props[self.state_map['e']] = True

        # if not self.found_trap and props[self.state_map['t']]:
        #     self.found_trap = True
        # if self.found_trap:
        #     props[self.state_map['t']] = True

        # if not self.found_home and props[self.state_map['h']]:
        #     self.found_home = True
        # if self.found_home:
        #     props[self.state_map['h']] = True

        return props

    def generate_default_props(self):
        props = copy.copy(self.default_props)
        props[self.get_state_keys(self.curr_loc)] = True
        props = self.update_props(props)
        return dict(zip(
                self.state_map.keys(),
                [props[v] for v in self.state_map.values()])
                )

    def generate_props_loc(self, loc):
        props = copy.copy(self.default_props)
        props[self.get_state_keys(loc)] = True
        props = self.update_props(props)
        return dict(zip(
                self.state_map.keys(),
                [props[v] for v in self.state_map.values()])
                )
        # return {'s33': props['s33'], 's32': props['s32']}
        # return props

    def get_state_keys(self, loc):
        return  's' + str(loc[0])+str(loc[1])

    def R(self, state):
        """Return a numeric reward for this state."""
        if (state == self.cheese and self.reward_swap is False) :
            r = self.reward[state]
            self.reward[state] = self.reward[self.startloc]
            self.reward[self.startloc] = r
            self.reward_swap = True
            return r
        else:
            return self.reward[state]

    def calculate_T(self, state, action, fp=0.8, lp=0.1, rp=0.1, bp=0.0):
        if action:
            return [(fp, self.go(state, action)),
                    (rp, self.go(state, turn_right(action))),
                    (lp, self.go(state, turn_left(action)))]
        else:
            return [(0.0, state)]

    def step(self, action):
        # print(self.curr_loc)
        done = False
        if self.curr_loc in self.terminals:
            done = True
            self.curr_reward = self.R(self.curr_loc)
            return self.curr_loc, self.curr_reward, done, None
        p, s1 = zip(*self.T(self.curr_loc, action))
        p, s1 = list(p), list(s1)
        indx = list(range(len(s1)))
        indx = self.nprandom.choice(indx, p=p)
        next_state = s1[indx]
        self.curr_loc = next_state
        if self.startloc in self.terminals:
            done = True
        self.curr_reward = self.R(next_state)
        return self.curr_loc, self.curr_reward, done, None

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""

        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""

        return list(reversed([[mapping.get((x, y), 'W')
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def restart(self):
        self.curr_loc = self.startloc
        self.reward = self.backup_reward.copy()
        self.reward_swap = False
        # self.cheese = self.terminals[0]
        self.curr_reward = self.reward[self.startloc]

    def to_arrows(self, policy):
        # EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        # Right, Up, Left, Down
        chars = {
            (1, 0): '>', (0, 1): '^',
            (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})


class GridMDPCheeseBeans(MDP):

    def __init__(
            self, grid, terminals, init=(0, 0),
            gamma=.9, startloc=(3, 0), seed=None):

        grid.reverse()     # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        if seed is None:
            self.nprandom = np.random.RandomState()  # pylint: disable=E1101
        else:
            self.nprandom = np.random.RandomState(   # pylint: disable=E1101
                seed)
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        actlist = orientations
        transitions = {}
        # Goal related
        # self.goalspec = goalspec
        self.startloc = startloc
        self.curr_loc = self.startloc

        if len(terminals) >= 2:
            self.cheese = terminals[0]
            self.beans = terminals[1]
        else:
            self.cheese = None
            self.beans = None

        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a)
        # print(reward)
        self.backup_reward = reward.copy()
        self.terminals = terminals
        MDP.__init__(self, init, actlist=actlist,
                     terminals=[terminals[1]], transitions=transitions,
                     reward=reward, states=states, gamma=gamma)
        self.reward_swap = False
        self.action_dict = {
            'S': (1, 0),
            'E': (0, 1),
            'N': (-1, 0),
            'W': (0, -1)
        }
        self.env_action_dict = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1)
        }
        self.state_dict = dict()
        self.curr_reward = self.reward[self.startloc]
        self.state_keyslist = ['s'+str(i)+str(j) for i in range(self.rows) for j in range(self.cols)]
        # home = 's'+ str(self.startloc[0]) + str(self.startloc[1])
        home = 's02'
        self.state_map = {'c': 's33', 'b': 's30', 'h': home, 'ct':'s23', 'bt':'s20'}
        self.default_props = dict(zip(self.state_keyslist, [False] * len(self.state_keyslist)))
        self.carrying_cheese = False
        self.carrying_beans = False

    def update_props(self, props):
        if not self.carrying_cheese and props[self.state_map['c']]:
            self.carrying_cheese = True
            props[self.state_map['c']] = True
        elif self.carrying_cheese:
            props[self.state_map['c']] = True
        if not self.carrying_beans and props[self.state_map['b']]:
            self.carrying_beans = True
            props[self.state_map['b']] = True
        elif self.carrying_beans:
            props[self.state_map['b']] = True

        if self.reward[(2,3)] ==-2 and self.curr_loc==(2,3):
            props[self.state_map['ct']] = True
        else:
            props[self.state_map['ct']] = False
        if self.reward[(2,0)] ==-2 and self.curr_loc==(2,0):
            props[self.state_map['bt']] = True
        else:
            props[self.state_map['bt']] = False
        return props

    def generate_default_props(self):
        props = copy.copy(self.default_props)
        props[self.get_state_keys(self.curr_loc)] = True
        props = self.update_props(props)
        return dict(zip(
                self.state_map.keys(),
                [props[v] for v in self.state_map.values()])
                )

    def generate_props_loc(self, loc):
        props = copy.copy(self.default_props)
        props[self.get_state_keys(loc)] = True
        props = self.update_props(props)
        return dict(zip(
                self.state_map.keys(),
                [props[v] for v in self.state_map.values()])
                )

    def get_state_keys(self, loc):
        return  's' + str(loc[0])+str(loc[1])

    def R(self, state):
        """Return a numeric reward for this state."""
        if (state == self.cheese and self.reward_swap is False) :
            r = self.reward[state]
            self.reward[state] = self.reward[self.startloc]
            self.reward[self.startloc] = r
            self.reward_swap = True
            return r
        else:
            return self.reward[state]

    def calculate_T(self, state, action, fp=0.95, lp=0.025, rp=0.025, bp=0.0):
        if action:
            return [(fp, self.go(state, action)),
                    (rp, self.go(state, turn_right(action))),
                    (lp, self.go(state, turn_left(action)))]
        else:
            return [(0.0, state)]

    def step(self, action):
        # print(self.curr_loc)
        done = False
        if self.curr_loc in self.terminals:
            done = True
            self.curr_reward = self.R(self.curr_loc)
            return self.curr_loc, self.curr_reward, done, None
        p, s1 = zip(*self.T(self.curr_loc, action))
        p, s1 = list(p), list(s1)
        indx = list(range(len(s1)))
        indx = self.nprandom.choice(indx, p=p)
        next_state = s1[indx]
        self.curr_loc = next_state
        if self.startloc in self.terminals:
            done = True
        self.curr_reward = self.R(next_state)
        return self.curr_loc, self.curr_reward, done, None

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""

        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""

        return list(reversed([[mapping.get((x, y), 'W')
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def restart(self):
        self.curr_loc = self.startloc
        self.reward = self.backup_reward.copy()
        self.reward_swap = False
        # self.cheese = self.terminals[0]
        self.curr_reward = self.reward[self.startloc]

    def to_arrows(self, policy):
        # EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        # Right, Up, Left, Down
        chars = {
            (1, 0): '>', (0, 1): '^',
            (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})


"""Helpful classes for the MDP world."""


def dictmax(d, s='key'):
    maxval = -999999999
    maxkey = None
    for key, value in d.items():
        if value > maxval:
            maxval = value
            maxkey = key
    if s == 'key':
        return maxkey
    elif s == 'val':
        return maxval


def create_policy(qtable):
    for s in qtable.keys():
        qtable[s] = dictmax(qtable[s], s='key')

    return qtable


def value_to_policy(mdp, U):
    """Solve an MDP by policy iteration"""
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
    for s in mdp.states:
        a = argmax(
            mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
        pi[s] = a
    return pi


def value_iteration(mdp, epsilon=0.001):
    """Solving an MDP by value iteration."""
    U1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return U


def expected_utility(a, s, U, mdp):
    """The expected utility of doing a in state s,
    according to the MDP and U."""
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])


def policy_evaluation(pi, U, mdp, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            temp = sum([p * U[s1] for (p, s1) in T(s, pi[s])])
            U[s] = R(s) + gamma * temp
    return U


def policy_test(pi, mdp, k=34):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    _, T, _ = mdp.R, mdp.T, mdp.gamma
    # s = (0, 0)
    s = mdp.startloc
    trace = [s]
    for i in range(k):
        p, s1 = zip(*T(s, pi[s]))
        p, s1 = list(p), list(s1)
        s1 = s1[np.argmax(p)]
        s = s1
        trace.append(s)
        if s == (3, 3):
            return i+1, True, trace
        if s in mdp.terminals:
            return 0, False, trace
    return 0, False, trace


def policy_test_step(pi, mdp, k=34):
    curr_loc = mdp.startloc
    trace = [curr_loc]
    while True:
        curr_loc, curr_reward, done, _ = mdp.step(pi[curr_loc])
        print(_)
        trace.append(curr_loc)
        if done:
            break
    return trace


def policy_iteration(mdp):
    """Solve an MDP by policy iteration"""
    U = {s: 0 for s in mdp.states}
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(
                mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi

        # home = 's'+ str(self.startloc[0]) + str(self.startloc[1])
        # return {
        #     's33': props['s33'], 's32': props['s32'],
        #     home: props[home]}


def random_policy(mdp):
    """Solve an MDP by policy iteration"""
    U = {s: 0 for s in mdp.states}
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
    return pi