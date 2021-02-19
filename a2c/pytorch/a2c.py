import argparse
import os
import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
import tensorflow as tf
from torch.nn import Module, Linear, ReLU, Sequential, Softmax
from torch.nn import functional as F
from torch.optim import Adam

torch.manual_seed(19971124)
np.random.seed(42)
random.seed(101)

mse_loss_function = torch.nn.MSELoss()


class Actor(Module):
    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        self.lin1 = Sequential(Linear(in_features=n_states, out_features=24), ReLU())
        # self.lin2 = Sequential(Linear(in_features=16, out_features=24), ReLU())
        self.final_lin = Sequential(Linear(in_features=24, out_features=n_actions), Softmax(dim=1))

    def forward(self, x):
        x = self.lin1(x)
        # x = self.lin2(x)
        output = self.final_lin(x)
        return output


class Critic(Module):
    def __init__(self, n_states):
        super(Critic, self).__init__()
        self.lin1 = Sequential(Linear(in_features=n_states, out_features=24), ReLU())
        self.lin2 = Sequential(Linear(in_features=24, out_features=24), ReLU())
        self.final_lin = Sequential(Linear(in_features=24, out_features=1))

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        # x = self.lin3(x)
        output = self.final_lin(x)
        return output


class A2C:
    def __init__(self, n_states, n_actions):
        self.states = []  # logging the states
        self.states_next = []  # logging the states
        self.actions = []  # logging the ACTUAL action that was performed at t
        self.model_outputs = []  # logging PI(a|s) that was outputted at t
        self.rewards = []  # logging the rewards that go from t_i -> t_{i+1}
        self.n_actions = n_actions
        self.lr = 0.001
        self.batch_size = 64
        self.gamma = 0.99
        self.actor = Actor(n_states=n_states, n_actions=n_actions)
        self.critic = Critic(n_states=n_states)
        self.optim_actor = Adam(params=self.actor.parameters(), lr=self.lr)
        self.optim_critic = Adam(params=self.critic.parameters(), lr=self.lr * 5)

    def reset(self):
        self.states = []
        self.states_next = []
        self.actions = []
        self.model_outputs = []
        self.rewards = []

    def calculate_delta(self):
        T = len(self.rewards)
        v_s = [0 for _ in range(T + 1)]  # there is one more value compare to number of states
        d_s = [0 for _ in range(T)]  # [delta_0, delta_1, delta_2, delta_3, ... delta_{T-1}]]
        last_index = T - 1
        for i, r in enumerate(reversed(self.rewards)):
            curr_index = last_index - i
            s_curr_tensor = torch.Tensor(self.states[curr_index])
            v_s[curr_index] = self.critic(s_curr_tensor)
            d_s[curr_index] = r + self.gamma * v_s[curr_index + 1] - v_s[curr_index]
        deltas_tensor = torch.Tensor(d_s).unsqueeze(dim=1)
        return deltas_tensor

    def get_action(self, state):
        action_probs = self.actor(state.float())
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
        action_probs_tensor_flattened = torch.sum(action_probs_tensor, dim=1).unsqueeze(1)
        return action_probs_tensor_flattened

    # critic -> policy evaluation (value function approx)
    def train_critic(self):
        self.optim_critic.zero_grad()
        states_tensor = torch.cat(self.states, dim=0)
        predict = self.critic(states_tensor)

        rewards_tensor = torch.Tensor(self.rewards).unsqueeze(dim=-1)
        next_states_tensor = torch.cat(self.states_next, dim=0)
        with torch.no_grad():
            target = self.gamma * self.critic(next_states_tensor) + rewards_tensor
            # since value at terminal state is zero, R_T + gamma*V(s_T) == R_T
            target[-1] = rewards_tensor[-1]
        loss = mse_loss_function(predict, target.detach())

        loss.backward()
        self.optim_critic.step()
        # self.reset()

    # actor -> policy network (improve policy network)
    def train_actor(self):
        self.optim_actor.zero_grad()
        deltas = self.calculate_delta()
        action_probs = self.get_action_probs()
        log_action_probs = torch.log(action_probs)
        # print(log_action_probs.shape, discounted_rewards.shape)
        cross_entropy = log_action_probs * deltas
        # print(log_action_probs, discounted_rewards)
        # print(cross_entropy)
        loss = -1 * torch.sum(cross_entropy)
        loss.backward()
        self.optim_actor.step()
        # self.reset()


def main(episodes, exp_name):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    env = gym.make('CartPole-v1')
    states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.n
    agent = A2C(n_states=states, n_actions=n_actions)
    warmup_ep = 0
    for ep in range(episodes):
        s_curr = np.reshape(env.reset(), (1, states))
        s_curr = s_curr.astype(np.float32)

        done = False
        score = 0

        while not done:
            # if len(agent.experience_replay) == agent.replay_size:
            #     env.render()
            s_curr_tensor = torch.from_numpy(s_curr)
            a_curr, model_output = agent.get_action(s_curr_tensor)
            s_next, r, done, _ = env.step(a_curr)
            s_next = np.reshape(s_next, (1, states)).astype(np.float32)
            s_next_tensor = torch.from_numpy(s_next)
            r = r if not done or score >= 499 else -100

            agent.states.append(s_curr_tensor)
            agent.states_next.append(s_next_tensor)
            agent.actions.append(a_curr)
            agent.rewards.append(r)
            agent.model_outputs.append(model_output)
            score += r

            s_curr = s_next

            if done:
                score = score if score == 500 else score + 100
                print(f"ep:{ep - warmup_ep}:################Goal Reached###################", score)
                agent.train_actor()
                agent.train_critic()
                agent.reset()

                with writer.as_default():
                    tf.summary.scalar("reward", r, ep)
                    tf.summary.scalar("score", score, ep)
    return agent


def env_with_render(agent):
    done = False
    env = gym.make('CartPole-v1')
    score = 0
    states = env.observation_space.shape[0]  # shape returns a tuple
    s_curr = np.reshape(env.reset(), (1, states))
    while True:
        if done:
            score = 0
            s_curr = np.reshape(env.reset(), (1, states))
        env.render()
        s_curr_tensor = torch.from_numpy(s_curr)
        a_curr, _ = agent.get_action(s_curr_tensor)
        s_next, r, done, _ = env.step(a_curr)
        s_next = np.reshape(s_next, (1, states))
        s_curr = s_next
        score += r
        print(score)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="PG_REINFORCE_pt", help="exp_name")
    ap.add_argument("--episodes", type=int, default=2000, help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"])
    env_with_render(agent=trained_agent)

gym.envs.register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.74
)
