import argparse
import os
import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
import tensorflow as tf
from torch.nn import Module, Linear, ReLU, Sequential
from torch.optim import Adam


class Network(Module):
    def __init__(self, n_states, n_actions):
        super(Network, self).__init__()
        self.lin1 = Sequential(Linear(in_features=n_states, out_features=16), ReLU())
        self.lin2 = Sequential(Linear(in_features=16, out_features=24), ReLU())
        self.final_lin = Linear(in_features=24, out_features=n_actions)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        output = self.final_lin(x)
        return output


class Reinforce:
    def __init__(self, n_states, n_actions):
        self.states = [] # logging the states
        self.actions = [] # logging the ACTUAL action that was performed at t
        self.model_outputs = [] # logging PI(a|s) that was outputted at t
        self.rewards = [] # logging the rewards that go from t_i -> t_{i+1}
        self.n_actions = n_actions
        self.states = n_states
        self.lr = 0.001
        self.batch_size = 64
        self.gamma = 0.99
        self.batch_indices = np.array([i for i in range(64)])
        self.policy_network = Network(n_states=n_states, n_actions=n_actions)
        self.optim = Adam(params=self.policy_network.parameters(), lr=self.lr)

    def calculate_G(self):
        T = len(self.rewards)
        discounted_rewards = [0 for _ in range(T)] # [G0, G2, G3, ... G_{T-1}]]
        last_index = T-1
        G_tp1 = 0 # G_{t+1}
        for i, r in reversed(enumerate(reversed(self.rewards[:-1]))):
            # starting from {T-1}, counting down to to 0 (Given the episode started at 0 and ended at T-1)
            curr_index = last_index - i
            G_t = r + self.gamma*G_tp1
            discounted_rewards[curr_index] = G_t
            G_tp1 = G_t # current G is the future G for the next iteration lol
        return discounted_rewards

    def get_action(self, state, test=False):
        pass

    def train_on_batch(self, states, targets):
        self.optim.zero_grad()
        self.optim.step()
        return

    def train(self):
        pass


def main(episodes, exp_name):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    env = gym.make('CartPole-v1')
    states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.n
    agent = Reinforce(n_states=states, n_actions=n_actions)
    warmup_ep = 0
    for ep in range(episodes):
        s_curr = env.reset()
        s_curr = np.reshape(s_curr, (1, states))
        s_curr = s_curr.astype(np.float32)
        done = False
        score = 0
        agent.update_weights()  # update weight every time an episode ends
        while not done:
            # if len(agent.experience_replay) == agent.replay_size:
            #     env.render()
            s_curr_tensor = torch.from_numpy(s_curr)
            a_curr = agent.get_action(s_curr_tensor)
            s_next, r, done, _ = env.step(a_curr)
            s_next_tensor = torch.from_numpy(s_next)
            s_next = np.reshape(s_next, (1, states))
            r = r if not done or r > 499 else -100
            sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])

            sample.s_curr = s_curr_tensor
            sample.a_curr = a_curr
            sample.reward = r
            sample.s_next = s_next_tensor
            sample.done = done

            if len(agent.experience_replay) < agent.replay_size:
                agent.experience_replay.append(sample)
                s_curr = s_next
                continue
            else:
                agent.experience_replay.append(sample)
                x_batch = random.sample(agent.experience_replay, agent.batch_size)
                agent.train(x_batch)

            score += r

            s_curr = s_next

            if done:
                score = score if score == 500 else score + 100
                print(f"ep:{ep - warmup_ep}:################Goal Reached###################", score)

                with writer.as_default():
                    tf.summary.scalar("reward", r, ep)
                    tf.summary.scalar("score", score, ep)
    return agent


def env_with_render(agent):
    done = False
    env = gym.make('CartPole-v1')
    score = 0
    s_curr = env.reset()
    while True:
        if done:
            score = 0
            s_curr = env.reset()
        env.render()
        s_curr_tensor = torch.from_numpy(s_curr)
        a_curr = agent.get_action(s_curr_tensor, test=True)
        s_next, r, done, _ = env.step(a_curr)
        s_curr = s_next
        score += r
        print(score)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="PG_REINFORCE_pt", help="exp_name")
    ap.add_argument("--episodes", type=int, default=200, help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"])
    env_with_render(agent=trained_agent)
