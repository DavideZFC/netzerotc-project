import gym
import numpy as np


class Records:
    def __init__(self, env):
        self.env = env

    def record_random_trajectories(self, episodes):
        rewards = []
        action_lists = []
        reward_lists = []
        state_lists = []
        for ep in range(episodes):
            actions = []
            rews = []
            states = []
            self.env.reset()
            while not self.env.state.is_done():
                ###
                action = self.env.action_space.sample()
                ###
                state, rew, done, _ = self.env.step(action)

                actions.append(action)
                rews.append(rew)
                states.append(state)
            action_lists.append(np.array(actions))
            reward_lists.append(np.array(rews))
            state_lists.append(np.array(states))
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards), action_lists, reward_lists, state_lists


env = gym.make(
    # "reference_environment_direct_deployment:reference-environment-direct-deployment-v0"
    "reference_environment:rangl-nztc-v0"
)
episodes = 100
rec = Records(env)
mean_reward_rand, action_lists, reward_lists, state_lists = rec.record_random_trajectories(episodes=episodes)

print(state_lists[0])