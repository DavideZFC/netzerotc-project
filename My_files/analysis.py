import gym
import numpy as np


env = gym.make(
    # "reference_environment_direct_deployment:reference-environment-direct-deployment-v0"
    "reference_environment:rangl-nztc-v0"
)
print(env.state)
print('action space = {}, state space = {}'.format(env.action_space, env.observation_space))