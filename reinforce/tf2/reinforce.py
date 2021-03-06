import argparse
import gym
import numpy as np
import os
import random
import tensorflow as tf
from collections import namedtuple, deque
from tensorflow.keras.layers import Dense, Input, Activation, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class Reinforce:
    def __init__(self, n_states, n_actions):
        self.states = []  # logging the states
        self.actions = []  # logging the ACTUAL action that was performed at t
        self.model_outputs = []  # logging PI(a|s) that was outputted at t
        self.rewards = []  # logging the rewards that go from t_i -> t_{i+1}
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.001
        self.batch_size = 64
        self.gamma = 0.99
        self.batch_indices = np.array([i for i in range(64)])
        self.policy_network = self.define_model()

    def define_model(self):
        state = Input(self.n_states)
        x = Dense(16, kernel_initializer='he_uniform')(state)
        x = Activation("relu")(x)
        x = Dense(24, kernel_initializer='he_uniform')(x)
        x = Activation("relu")(x)
        x = Dense(self.n_actions, kernel_initializer='he_uniform', kernel_regularizer="l2")(x)
        prob = Softmax()(x)
        model = Model(state, prob)
        return model

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
        action_probs = self.policy_network(state)
        action_probs_squeeze = tf.squeeze(action_probs).numpy()
        action_probs_squeeze /= action_probs_squeeze.sum()  # normalize
        return np.random.choice(self.n_actions, 1, p=action_probs_squeeze), action_probs

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

    def train(self):
        with tf.GradientTape() as tape:
            discounted_rewards = self.calculate_G()
            action_probs = self.get_action_probs()

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
            a_curr, model_output = agent.get_action(s_curr)
            s_next, r, done, _ = env.step(a_curr)
            s_next = np.reshape(s_next, (1, states))
            r = r if not done or score >= 499 else -100

            agent.states.append(s_curr)
            agent.actions.append(a_curr)
            agent.rewards.append(r)
            agent.model_outputs.append(model_output)

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
    n_states = env.observation_space.shape[0]  # shape returns a tuple
    s_curr = env.reset()
    while True:
        if done:
            score = 0
            s_curr = env.reset()
        env.render()
        s_curr = np.reshape(s_curr, (1, n_states))
        a_curr = agent.get_action(s_curr, test=True)
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
