import numpy as np


class Learner:

    def __init__(self, n_arms, n_campaigns):
        self.n_arms = n_arms
        self.n_campaigns = n_campaigns
        self.t = 0
        self.rewards_per_arm = [[[] for _ in range(n_campaigns)] for _ in range(n_arms)]  # shape: (n_arms,n_campaigns)
        self.collected_rewards = np.array([])

    def update_observations(self, reward):
        self.collected_rewards = np.append(self.collected_rewards, reward)
