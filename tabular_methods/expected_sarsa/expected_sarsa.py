import gym
import numpy as np

import itertools

import random
from collections import namedtuple

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

random.seed(a=19971124)
np.random.seed(seed=19940513)
sns.set_theme(style="darkgrid")


class Agent:

    def __init__(self, gamma, n_bins, n_iterations, env):
        self.n_iterations = n_iterations
        # reference from https://github.com/openai/gym/blob/8a721ace460cbaf8c3e6c03c12d06c616fd6e1e8/gym/envs/classic_control/cartpole.py#L51
        self._state_bins = [
            # Cart Position
            self._discretize(lb=-2.4, ub=2.4, n_bins=n_bins),
            # Cart Velocity
            self._discretize(lb=-30.0, ub=30.0, n_bins=n_bins),
            # Pole Angle
            self._discretize(lb=-0.5, ub=0.5, n_bins=n_bins),
            # Pole Angular Velocity
            self._discretize(lb=-2.0, ub=2.0, n_bins=n_bins)
        ]

        self.n_states = n_bins ** len(self._state_bins)
        self.n_actions = 2
        self.q_table = np.ones((self.n_states, self.n_actions)) * 0.5
        self.c = np.zeros((self.n_states, self.n_actions))
        self.gamma = gamma

        self.state_to_idx = {}
        self.init_states()
        self.env = env

        self.epsilon = 0.1
        self.lr = 0.01
        self.test_scores = []

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

    def policy(self, state):
        decision = np.random.rand()
        if decision < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            q = self.q_table[state]
            action = np.argmax(q)
        return action

    def sarsa(self):
        for ep in range(self.n_iterations):
            s_curr = self.env.reset()
            a_curr = self.policy(self.get_discrete_state(s_curr))
            done = False
            score = 0
            while not done:
                # env.render()
                s_next, r, done, _ = self.env.step(a_curr)
                r = r if not done or score >= 199 else - 100

                score += r

                if done:
                    score = score if score == 200 else score + 100
                    # print("St################Goal Reached###################", score)
                s_curr_discrete = self.get_discrete_state(s_curr)
                s_next_discrete = self.get_discrete_state(s_next)
                a_next = self.policy(s_next_discrete)

                q_curr = self.q_table[s_curr_discrete, a_curr]
                # finding the expected value of q by taking expectaion over all possible actions
                # compared to sarsa, where just the next action was taken to find the q value
                expected_q_next = 0
                for i in range(self.n_actions):
                    if i == np.argmax(self.q_table[s_next_discrete]):
                        p = 1 - self.epsilon + (self.epsilon / self.n_actions)
                    else:
                        p = self.epsilon / self.n_actions
                    expected_q_next += p * self.q_table[s_next_discrete, i]

                done = int(done)
                self.q_table[s_curr_discrete, a_curr] = q_curr + self.lr * (
                            r + (1 - done) * (self.gamma * expected_q_next) - q_curr)

                s_curr = s_next
                a_curr = a_next
            score = score if score == 200 else score + 100
            self.test_scores.append(score)
            if (ep + 1) % 1 == 0:
                print(f"Iteration {ep}: score = {score}")

    def generate_plot(self):
        x_axis = [i for i in range(len(self.test_scores))]
        d = {"iteration": x_axis, "scores": self.test_scores}
        experiment = pd.DataFrame(data=d)
        plot = sns.lineplot(data=experiment, x="iteration", y="scores")
        plot.set_title("Score of Off Policy MC Control on Discretized Cartpole")
        plot.figure.savefig("Score_vs_Iteration.png")


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    scores = []
    n_iterations = 5000
    n_trials = 5
    for i in range(n_trials):
        agent = Agent(gamma=0.9, n_bins=20, n_iterations=n_iterations, env=env)
        agent.sarsa()
        scores += agent.test_scores

    x_axis = [i for i in range(n_iterations)] * n_trials
    trials = []
    for i in range(n_trials):
        trials += [i for _ in range(n_iterations)]

    d = {"Iteration": x_axis, "Trial": trials}
    experiments = pd.DataFrame(data=d)

    experiments.insert(2, f"Score", scores, False)
    plot = sns.lineplot(data=experiments, x="Iteration", y="Score")
    plot.set_title("Score of Expected SARSA on Discretized Cartpole")
    plot.figure.savefig(f"Score_vs_Iteration_{n_iterations}_{n_trials}.png")
