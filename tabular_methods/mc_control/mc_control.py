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

    def generate_episodes(self, iteration):
        done = False
        episode = []

        Sample = namedtuple('Sample', ['s', 'a', 'b_a_s', 'r', 's_next', 'done'])
        s_curr = env.reset()
        score = 0
        while not done:
            # action = np.random.choice(n_actions, 1, p=policy)
            if iteration < self.n_iterations // 8:
                epsilon = 0.1
            elif self.n_iterations // 8 < iteration < self.n_iterations // 4:
                epsilon = 0.05
            else:
                epsilon = 0.001
            random_sampled = random.uniform(0, 1)
            if random_sampled < epsilon:
                action_prob = epsilon / 2
                prob_value = random.uniform(0, 1)
                if prob_value < 0.5:
                    action = 0
                else:
                    action = 1
            else:
                # greedy action selection
                action_prob = (1 - epsilon)
                action = np.argmax(self.q_table[self.get_discrete_state(s_curr)])
            s_next, r, done, _ = env.step(int(action))

            # reward of -100 (punishment) if the pole falls before reaching the maximum score
            r = r if not done or r >= 199 else - 100

            sample = Sample(s=s_curr, a=int(action), b_a_s=action_prob, r=r, s_next=s_next, done=done)
            episode.append(sample)

            # score is reflective of how much reward the agent received
            score += r
            s_curr = s_next

        return episode

    def mc_control(self):
        for i in range(self.n_iterations):
            episode = self.generate_episodes(iteration=i)
            G_tp1 = 0  # G_{t+1}
            W = 1
            for step in reversed(episode):
                s_curr, a, b_a_s, r, s_next, done = step.s, step.a, step.b_a_s, step.r, step.s_next, step.done
                G_t = r + self.gamma * G_tp1
                s_curr_discrete = self.get_discrete_state(s_curr)
                self.c[s_curr_discrete][a] = self.c[s_curr_discrete][a] + W
                self.q_table[s_curr_discrete][a] = self.q_table[s_curr_discrete][a] + (
                        W / self.c[s_curr_discrete][a]) * (
                                                           G_t - self.q_table[s_curr_discrete][a])
                pi_st = np.argmax(self.q_table[s_curr_discrete])
                G_tp1 = G_t
                if a != pi_st:
                    break
                W = W / b_a_s

            done = False
            s_curr = env.reset()
            score = 0
            while not done:
                # if len(agent.experience_replay) == agent.replay_size:
                #     env.render()
                s_curr_discrete = self.get_discrete_state(s_curr)
                action = np.argmax(self.q_table[s_curr_discrete])
                s_next, r, done, _ = env.step(int(action))
                r = r if not done or score >= 199 else - 100
                score += r
                s_curr = s_next
            score = score if score == 200 else score + 100
            self.test_scores.append(score)
            if (i + 1) % 500 == 0:
                print(f"Iteration {i}: score = {score}")


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    scores = []
    n_iterations = 5000
    n_trials = 10
    for i in range(n_trials):
        agent = Agent(gamma=1, n_bins=10, n_iterations=n_iterations, env=env)
        agent.mc_control()
        scores += agent.test_scores

    x_axis = [i for i in range(n_iterations)] * n_trials
    trials = []
    for i in range(n_trials):
        trials += [i for _ in range(n_iterations)]

    d = {"Iteration": x_axis, "Trial": trials}
    experiments = pd.DataFrame(data=d)

    experiments.insert(2, f"Score", scores, False)
    plot = sns.lineplot(data=experiments, x="Iteration", y="Score")
    plot.set_title("Score of Off Policy MC Control on Discretized Cartpole")
    plot.figure.savefig(f"Score_vs_Iteration_{n_iterations}_{n_trials}.png")
