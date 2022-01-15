#!/usr/bin/env python

import gym
import numpy as np


class Evaluate:
    def __init__(self, env):
        self.env = env

    def random_agent(self, seeds):
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                ###
                action = self.env.action_space.sample()
                ###
                self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)


    def constant_agent(self, const, seeds=[0]):
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                ###
                action = const*np.ones(3)
                ###
                state, reward, done, _ = self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

env = gym.make(
    "reference_environment:rangl-nztc-v0"
)

evaluate = Evaluate(env)
const = 1
mean_reward_rand = evaluate.random_agent(seeds=[12])
mean_reward_const = evaluate.constant_agent(const=const)

print("mean reward of random "+str(mean_reward_rand/10000))
print('mean reward of constant {} agent '.format(const) +str(mean_reward_const/10000))
