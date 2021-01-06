import argparse
import os
import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
import tensorflow as tf
from torch.nn import Module, Linear, ReLU, Sequential, Softmax
from torch.optim import Adam


class Network(Module):
    def __init__(self, n_states, n_actions):
        super(Network, self).__init__()
        self.lin1 = Sequential(Linear(in_features=n_states, out_features=16), ReLU())
        self.lin2 = Sequential(Linear(in_features=16, out_features=24), ReLU())
        self.final_lin = Sequential(Linear(in_features=24, out_features=n_actions), Softmax(dim=1))

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        output = self.final_lin(x)
        return output


class Reinforce:
    def __init__(self, n_states, n_actions):
        self.states = []  # logging the states
        self.actions = []  # logging the ACTUAL action that was performed at t
        self.model_outputs = []  # logging PI(a|s) that was outputted at t
        self.rewards = []  # logging the rewards that go from t_i -> t_{i+1}
        self.n_actions = n_actions
        self.lr = 0.001
        self.batch_size = 64
        self.gamma = 0.99
        self.batch_indices = np.array([i for i in range(64)])
        self.policy_network = Network(n_states=n_states, n_actions=n_actions)
        self.optim = Adam(params=self.policy_network.parameters(), lr=self.lr)

    def reset(self):
        self.states = []
        self.actions = []
        self.model_outputs = []
        self.rewards = []

    def calculate_G(self):
        T = len(self.rewards)
        discounted_rewards = [0 for _ in range(T)]  # [G0, G2, G3, ... G_{T-1}]]
        last_index = T - 1
        G_tp1 = 0  # G_{t+1}
        for i, r in enumerate(reversed(self.rewards[:-1])):
            # starting from {T-1}, counting down to to 0 (Given the episode started at 0 and ended at T-1)
            curr_index = last_index - i
            G_t = r + self.gamma * G_tp1
            discounted_rewards[curr_index] = G_t
            G_tp1 = G_t  # current G is the future G for the next iteration lol
        return discounted_rewards

    def get_action(self, state):
        action_probs = self.policy_network(state.float())
        action_probs_np = action_probs.detach().cpu().numpy().squeeze()
        return int(np.random.choice(self.n_actions, 1, p=action_probs_np)), action_probs

    def get_action_probs(self):
        model_outputs_tensor = torch.cat(self.model_outputs, 0)
        index = torch.Tensor(self.actions).long()
        chosen_actions_tensor = torch.zeros_like(model_outputs_tensor).long()
        chosen_actions_tensor[torch.arange(chosen_actions_tensor.size(0)), index] = 1.
        action_probs_tensor = model_outputs_tensor * chosen_actions_tensor
        # print(model_outputs_tensor)
        # print(action_probs_tensor)
        return action_probs_tensor

    def train(self):
        action_probs = self.get_action_probs()

        self.optim.zero_grad()
        self.optim.step()
        print("training")
        self.reset()
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

        while not done:
            # if len(agent.experience_replay) == agent.replay_size:
            #     env.render()
            s_curr_tensor = torch.from_numpy(s_curr)
            a_curr, model_output = agent.get_action(s_curr_tensor)
            s_next, r, done, _ = env.step(a_curr)
            # s_next_tensor = torch.from_numpy(s_next)
            # s_next = np.reshape(s_next, (1, states))
            r = r if not done or r > 499 else -100

            agent.states.append(s_curr)
            agent.actions.append(a_curr)
            agent.rewards.append(r)
            agent.model_outputs.append(model_output)

            score += r

            s_next = np.reshape(s_next, (1, states))
            s_curr = s_next

            if done:
                score = score if score == 500 else score + 100
                print(f"ep:{ep - warmup_ep}:################Goal Reached###################", score)
                agent.train()

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
