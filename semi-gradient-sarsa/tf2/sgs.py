import argparse
import gym
import numpy as np
import os
import random
import tensorflow as tf
from collections import namedtuple, deque
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])


class QNetwork:
    def __init__(self, n_states, n_actions):
        self.epsilon = 0.1
        self.min_epsilon = 0.01
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.00001
        self.gamma = 0.999
        # self.target_network = self.define_model()
        # self.main_network = self.define_model()
        self.q_network = self.define_model()

    def define_model(self):
        state = Input(self.n_states)
        x = Dense(16, kernel_initializer='he_uniform')(state)
        x = Activation("relu")(x)
        x = Dense(24, kernel_initializer='he_uniform')(x)
        x = Activation("relu")(x)
        q = Dense(self.n_actions, kernel_initializer='he_uniform', kernel_regularizer="l2")(x)
        model = Model(state, q)
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return model

    def get_action(self, state, test=False):
        # e-greedy decision making
        if not test:
            decision = np.random.rand()
            # exploration
            if decision < self.epsilon:
                action = np.random.choice(self.n_actions)
            # exploitation
            else:
                q = self.q_network.predict(state)
                action = np.argmax(q)
        else:
            q = self.q_network.predict(state)
            action = np.argmax(q)

        return action

    def train(self, x_batch):
        s_curr, a_curr, r, s_next, a_next, done = x_batch
        target = np.zeros((1, self.n_actions))
        target[0, a_next] = r
        if not done:
            q_hat_next = self.q_network.predict(s_curr)
            target[0, a_curr] += self.gamma * q_hat_next[0, a_next]

        self.q_network.train_on_batch(s_curr, target)
    # def train(self, x_batch):
    #     s_curr, a_curr, r, s_next, a_next, done = x_batch
    #     target = np.ones((1, self.n_actions))*r
    #     if not done:
    #         q_hat_next = self.q_network.predict(s_curr)
    #         target += self.gamma * q_hat_next
    #
    #     self.q_network.train_on_batch(s_curr, target)


def main(episodes, exp_name):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    env = gym.make('CartPole-v1')
    states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.n
    agent = QNetwork(n_states=states, n_actions=n_actions)
    warmup_ep = 0
    for ep in range(episodes):
        s_curr = env.reset()
        s_curr = np.reshape(s_curr, (1, states))
        done = False
        score = 0
        while not done:
            # env.render()
            a_curr = agent.get_action(s_curr)
            s_next, r, done, _ = env.step(a_curr)
            s_next = np.reshape(s_next, (1, states))
            r = r if not done or score >= 499 else -100
            a_next = agent.get_action(s_next)
            x_batch = (s_curr, a_curr, r, s_next, a_next, done)
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
    n_states = env.observation_space.shape[0]  # shape returns a tuple
    s_curr = env.reset()
    while True:
        if done:
            score = 0
            s_curr = env.reset()
        env.render()
        s_curr = np.reshape(s_curr, (1, sn_states))
        a_curr = agent.get_action(s_curr, test=True)
        s_next, r, done, _ = env.step(a_curr)
        s_curr = s_next
        score += r
        print(score)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="DQN_tf2", help="exp_name")
    ap.add_argument("--episodes", type=int, default=500, help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"])
    env_with_render(agent=trained_agent)
