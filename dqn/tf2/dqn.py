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


class DQN:
    def __init__(self, n_states, n_actions):
        self.replay_size = 1024
        self.experience_replay = deque(maxlen=self.replay_size)
        self.epsilon = 0.1
        self.min_epsilon = 0.01
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.001
        self.gamma = 0.99
        self.batch_size = 64
        self.batch_indices = np.array([i for i in range(64)])
        # self.batch_indices = self.batch_indices[:, np.newaxis]
        self.target_network = self.define_model()
        self.main_network = self.define_model()

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
                q = self.main_network.predict(state)
                action = np.argmax(q)
        else:
            q = self.main_network.predict(state)
            action = np.argmax(q)

        return action

    def train(self, x_batch):
        s_currs = np.zeros((self.batch_size, self.n_states))
        a_currs = np.zeros((self.batch_size, 1))
        r = np.zeros((self.batch_size, 1))
        s_nexts = np.zeros((self.batch_size, self.n_states))
        dones = np.zeros((self.batch_size,))

        for batch in range(self.batch_size):
            s_currs[batch] = x_batch[batch].s_curr
            a_currs[batch] = x_batch[batch].a_curr
            r[batch] = x_batch[batch].reward
            s_nexts[batch] = x_batch[batch].s_next
            dones[batch] = x_batch[batch].done

        target = self.main_network.predict(s_currs)
        max_qs = np.amax(a=self.target_network.predict(s_nexts), axis=1)
        max_qs = max_qs[..., np.newaxis]
        a_indices = a_currs.astype(np.int)
        target[self.batch_indices, np.squeeze(a_indices)] = np.squeeze(r + self.gamma * (max_qs))

        done_indices = np.argwhere(dones)
        if done_indices.shape[0] > 0:
            done_indices = np.squeeze(np.argwhere(dones))
            target[done_indices, np.squeeze(a_indices[done_indices])] = np.squeeze(r[done_indices])

        self.main_network.train_on_batch(s_currs, target)

    def update_weights(self):
        self.target_network.set_weights(self.main_network.get_weights())


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
        done = False
        score = 0
        agent.update_weights()  # update weight every time an episode ends
        if agent.epsilon > agent.min_epsilon:
            agent.epsilon *= 0.99
        while not done:
            # env.render()
            a_curr = agent.get_action(s_curr)
            s_next, r, done, _ = env.step(a_curr)
            s_next = np.reshape(s_next, (1, states))
            r = r if not done or score >= 499 else -100
            sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])

            sample.s_curr = s_curr
            sample.a_curr = a_curr
            sample.reward = r
            sample.s_next = s_next
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
    ap.add_argument("--exp_name", type=str, default="DQN_tf2", help="exp_name")
    ap.add_argument("--episodes", type=int, default=200, help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"])
    env_with_render(agent=trained_agent)
