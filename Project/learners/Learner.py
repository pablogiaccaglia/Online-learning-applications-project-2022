import numpy as np
from CUSUM import CUSUM

class Learner:
    def __init__(self, n_arms, cusum_args = None, needs_boost = False):
        self.n_arms = n_arms
        self.t = 0  # current round variable
        self.rewards_per_arm = [[] for _ in range(self.n_arms)]
        self.collected_rewards = np.array([])
        self.cd_enabled = False
        self.bandit_name = 'Bandit'
        self.needs_boost = needs_boost

        if cusum_args:
            self.cd_enabled = True
            self.explorationAlpha = cusum_args['explorationAlpha']
            self.valid_collected_rewards = np.array([])
            self.detections = [[] for _ in range(n_arms)]

            cusum_args = dict(cusum_args)  # avoiding modifying original dict
            del cusum_args['explorationAlpha']
            self.change_detection = [CUSUM(**cusum_args) for _ in range(n_arms)]

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)  # optimizable

        if self.cd_enabled:
            self.valid_collected_rewards = np.append(self.valid_collected_rewards, reward)  # optimizable

    def reset(self):
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(self.n_arms)]
        self.collected_rewards = np.array([])

        if self.cd_enabled:
            self.valid_collected_rewards = np.array([])
            self.detections = [[] for _ in range(self.n_arms)]

            for cusum in self.change_detection:
                cusum.reset()