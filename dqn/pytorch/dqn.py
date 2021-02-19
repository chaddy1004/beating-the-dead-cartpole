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

torch.manual_seed(19971124)
np.random.seed(42)
random.seed(101)

sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])

mse_loss_function = torch.nn.MSELoss()


def l2_regularization(kernel):
    return torch.norm(kernel, p=2)


class _DQN(Module):
    def __init__(self, n_states, n_actions):
        super(_DQN, self).__init__()
        self.lin1 = Sequential(Linear(in_features=n_states, out_features=16), ReLU())
        self.lin2 = Sequential(Linear(in_features=16, out_features=24), ReLU())
        self.final_lin = Linear(in_features=24, out_features=n_actions)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        output = self.final_lin(x)
        return output


class DQN:
    def __init__(self, n_states, n_actions):
        self.replay_size = 1024
        self.experience_replay = deque(maxlen=self.replay_size)
        self.epsilon = 0.1
        self.min_epsilon = 0.01
        self.n_actions = n_actions
        self.states = n_states
        self.lr = 0.001
        self.gamma = 0.99
        self.batch_size = 64
        self.batch_indices = np.array([i for i in range(64)])
        self.target_network = _DQN(n_states=n_states, n_actions=n_actions)
        self.main_network = _DQN(n_states=n_states, n_actions=n_actions)
        self.optim = Adam(params=self.main_network.parameters(), lr=self.lr)

    def get_action(self, state, test=False):
        # e-greedy decision making
        decision = np.random.rand()
        # exploration
        if not test:
            if decision < self.epsilon:
                action = np.random.choice(self.n_actions)
            # exploitation
            else:
                q = self.main_network(state.float())
                q_np = q.detach().cpu().numpy()
                action = np.argmax(q_np)
        else:
            q = self.main_network(state.float())
            q_np = q.detach().cpu().numpy()
            action = np.argmax(q_np)

        return action

    def train_on_batch(self, states, targets):
        self.optim.zero_grad()
        predict = self.main_network(states)
        loss = mse_loss_function(input=predict,
                                 target=targets)  # + l2_regularization( kernel=self.main_network.final_lin.weight)
        loss.backward()
        self.optim.step()
        return

    def train(self, x_batch):
        s_currs = torch.zeros((self.batch_size, self.states))
        a_currs = torch.zeros((self.batch_size, 1))
        r = torch.zeros((self.batch_size, 1))
        s_nexts = torch.zeros((self.batch_size, self.states))
        dones = torch.zeros((self.batch_size,))

        for batch in range(self.batch_size):
            s_currs[batch] = x_batch[batch].s_curr
            a_currs[batch] = x_batch[batch].a_curr
            r[batch] = x_batch[batch].reward
            s_nexts[batch] = x_batch[batch].s_next
            dones[batch] = x_batch[batch].done

        target = self.main_network(s_currs)
        predicts = self.target_network(s_nexts)
        max_qs = torch.max(input=predicts, dim=1).values  # find max along an axis
        max_qs = max_qs.unsqueeze(-1)
        a_indices = a_currs.long()
        target[self.batch_indices, torch.squeeze(a_indices)] = torch.squeeze(r + self.gamma * (max_qs))

        done_indices = np.argwhere(dones)
        if done_indices.shape[0] > 0:
            done_indices = torch.squeeze(dones.nonzero())
            target[done_indices, torch.squeeze(a_indices[done_indices])] = torch.squeeze(r[done_indices])

        self.train_on_batch(s_currs, target)

    def update_weights(self):
        self.target_network.load_state_dict(self.main_network.state_dict())


def main(episodes, exp_name):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    env = gym.make('CartPole-v1')
    states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.n
    agent = DQN(n_states=states, n_actions=n_actions)
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
            r = r if not done or score >= 499 else -100
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
    ap.add_argument("--exp_name", type=str, default="DQN_pt", help="exp_name")
    ap.add_argument("--episodes", type=int, default=300, help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"])
    env_with_render(agent=trained_agent)
