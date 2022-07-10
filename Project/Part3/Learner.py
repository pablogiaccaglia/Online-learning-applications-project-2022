import numpy as np
from CUSUM import CUSUM

"""class Learner:

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
"""


class Learner:
    def __init__(self, n_arms, cusum_args = None):
        self.n_arms = n_arms
        self.t = 0  # current round variable
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])
        self.cd_enabled = False
        self.pulled_arms = []  # One arm for campaign

        if cusum_args:
            self.cd_enabled = True
            self.explorationAlpha = cusum_args['explorationAlpha']
            self.valid_collected_rewards = np.array([])
            self.detections = [[] for _ in range(n_arms)]

            cusum_args = dict(cusum_args)  # avoiding modifying original dict
            del cusum_args['explorationAlpha']
            self.change_detection = [CUSUM(**cusum_args) for _ in range(n_arms)]

    def update_observations(self, pulled_arm, reward):

        if self.cd_enabled:
            if self.change_detection[pulled_arm].update(reward):
                self.detections[pulled_arm].append(self.t)
                self.valid_collected_rewards = []
                self.pulled_arms = []
                self.change_detection[pulled_arm].reset()

        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)  # optimizable

        if self.cd_enabled:
            self.valid_collected_rewards = np.append(self.valid_collected_rewards, reward)  # optimizable
