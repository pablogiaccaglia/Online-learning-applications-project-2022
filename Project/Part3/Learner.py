import numpy as np


class Learner:

    def __init__(self, n_arms, n_campaigns):
        self.n_arms = n_arms
        self.n_campaigns = n_campaigns
        self.t = 0
        self.rewards_per_arm = [[[] for _ in range(n_arms)] for _ in range(n_campaigns)]  # shape: (n_campaigns,n_arms)
        self.collected_rewards = np.array([])

    def update_observations(self, reward):
        # not sure if this line is necessary, but I don't know how to append arrays starting from an empty one
        if not self.collected_rewards.any():
            self.collected_rewards = np.append(self.collected_rewards, reward)
        # this line stacks the collected rewards with the array of incoming rewards
        else:
            self.collected_rewards = np.append(np.atleast_2d(self.collected_rewards), np.atleast_2d(reward), axis=0)
