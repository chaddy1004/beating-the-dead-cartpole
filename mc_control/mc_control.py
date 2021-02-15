import gym

import numpy as np

import itertools


class Agent:

    def __init__(self):
        n_bins = 5
        # reference from https://github.com/openai/gym/blob/8a721ace460cbaf8c3e6c03c12d06c616fd6e1e8/gym/envs/classic_control/cartpole.py#L51
        self._state_bins = [
            # Cart Position
            self._discretize(lb=-2.4, ub=2.4, n_bins=n_bins),
            # Cart Velocity
            self._discretize(lb=-10.0, ub=10.0, n_bins=n_bins),
            # Pole Angle
            self._discretize(lb=-0.5, ub=0.5, n_bins=n_bins),
            # Pole Angular Velocity
            self._discretize(lb=-2.0, ub=2.0, n_bins=n_bins)
        ]

        self.n_states = n_bins * len(self._state_bins)
        self.n_actions = 2
        self.q_table = np.ones((self.n_states, self.n_actions)) * 0.5

        self.state_to_idx = {}
        self.init_states()

    def init_states(self):
        possible_states = list(
            itertools.product(list(self._state_bins[0]),
                              list(self._state_bins[1]),
                              list(self._state_bins[2]),
                              list(self._state_bins[3]))
        )

        for i, state in enumerate(possible_states):
            self.state_to_idx[state] = i

    def get_discrete_state(self, cts_state):
        cp = cts_state[0]
        cv = cts_state[1]
        pa = cts_state[2]
        pav = cts_state[3]

        cp_dis = self._find_nearest(self._state_bins[0], cp)
        cv_dis = self._find_nearest(self._state_bins[1], cv)
        pa_dis = self._find_nearest(self._state_bins[2], pa)
        pav_dis = self._find_nearest(self._state_bins[3], pav)

        state = (cp_dis, cv_dis, pa_dis, pav_dis)
        return self.state_to_idx[state]

    @staticmethod
    def _discretize(lb, ub, n_bins):
        return np.linspace(lb, ub, n_bins)

    @staticmethod
    def _find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
