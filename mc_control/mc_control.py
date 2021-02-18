import gym

import numpy as np

import itertools

import random
from collections import namedtuple

random.seed(19971124)


class Agent:

    def __init__(self, gamma, n_bins, n_iterations, env):
        self.n_iterations = n_iterations
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

        self.n_states = n_bins ** len(self._state_bins)
        self.n_actions = 2
        self.q_table = np.ones((self.n_states, self.n_actions)) * 0.5
        self.c = np.zeros((self.n_states, self.n_actions))
        self.gamma = gamma

        self.state_to_idx = {}
        self.init_states()
        self.env = env

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

    # def calculate_G(self):
    #     T = len(self.rewards)
    #     discounted_rewards = [0 for _ in range(T)]  # [G0, G1, G2, G3, ... G_{T-1}]]
    #     last_index = T - 1
    #     G_tp1 = 0  # G_{t+1}
    #     for i, r in enumerate(reversed(self.rewards)):
    #         # starting from {T-1}, counting down to to 0 (Given the episode started at 0 and ended at T-1)
    #         curr_index = last_index - i
    #         G_t = r + self.gamma * G_tp1
    #         discounted_rewards[curr_index] = G_t
    #         G_tp1 = G_t  # current G is the future G for the next iteration lol
    #     # print(discounted_rewards)
    #     # print(self.rewards)
    #     return discounted_rewards

    def generate_episodes(self, iteration):
        done = False
        n_actions = self.env.action_space.n
        episode = []

        Sample = namedtuple('Sample', ['s', 'a', 'r', 's_next', 'done'])
        s_curr = env.reset()
        while not done:
            # if len(agent.experience_replay) == agent.replay_size:
            #     env.render()
            # action = np.random.choice(n_actions, 1, p=policy)
            if iteration < self.n_iterations//2:
                epsilon = 0.1
            else:
                epsilon = 0.001
            random_sampled = random.uniform(0,1)
            # print(epsilon)
            if random_sampled < epsilon:
                prob_value = random.uniform(0, 1)
                if prob_value < 0.5:
                    action = 0
                else:
                    action = 1
            else:
                action = np.argmax(self.q_table[self.get_discrete_state(s_curr)])
            s_next, r, done, _ = env.step(int(action))
            r = r if not done or r > 10001 else - 100
            sample = Sample(s=s_curr, a=int(action), r=r, s_next=s_next, done=done)
            episode.append(sample)
            s_curr = s_next
        return episode

    def mc_control(self):
        for i in range(self.n_iterations):
            prob_value = random.uniform(0, 1)
            b = [prob_value, 1 - prob_value]
            # print("b", b)
            episode = self.generate_episodes(iteration=i)
            # print("N EPISODE", len(episode))
            # last_index = len(episode) - 1
            G_tp1 = 0  # G_{t+1}
            W = 1
            for step in reversed(episode):
                s_curr, a, r, s_next, done = step.s, step.a, step.r, step.s_next, step.done
                G_t = r + self.gamma * G_tp1
                # curr_index = last_index - i
                s_curr_discrete = self.get_discrete_state(s_curr)
                self.c[s_curr_discrete][a] = self.c[s_curr_discrete][a] + W
                self.q_table[s_curr_discrete][a] = self.q_table[s_curr_discrete][a] + (W / self.c[s_curr_discrete][a]) * (
                            G_t - self.q_table[s_curr_discrete][a])
                pi_st = np.argmax(self.q_table[s_curr_discrete])
                if a != pi_st:
                    # print("asdf", a, pi_st)
                    break
                W = W / b[a]
                # print("W", W)
                G_tp1 = G_t

            done = False
            n_actions = self.env.action_space.n
            episode = []
            s_curr = env.reset()
            score = 0
            while not done:
                # if len(agent.experience_replay) == agent.replay_size:
                #     env.render()
                s_curr_discrete = self.get_discrete_state(s_curr)
                action = np.argmax(self.q_table[s_curr_discrete])
                s_next, r, done, _ = env.step(int(action))
                r = r if not done or r > 10001 else - 100
                score += r
                s_curr = s_next
            score = score if score >= 10000 else score + 100
            print(f"Iteration {i}: score = {score}")





if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    agent = Agent(gamma=0.99, n_bins=70, n_iterations=10000000, env=env)
    agent.mc_control()
