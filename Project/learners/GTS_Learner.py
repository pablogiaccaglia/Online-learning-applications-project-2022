from learners.Learner import Learner
import numpy as np
from entities.Utils import BanditNames

class GTS_Learner(Learner):
    """ every arms induces a gaussian distribution over its expected reward """

    def __init__(self, arms, prior_mean, prior_sigma=1, cusum_args = None):
        self.arms = arms
        super().__init__(len(arms), cusum_args = cusum_args)
        self.means = np.ones(self.n_arms) * prior_mean
        self.sigmas = np.ones(self.n_arms) * prior_sigma
        self.prior_mean = prior_mean
        self.prior_sigma = prior_sigma
        self.bandit_name = BanditNames.GTS_Learner.name

    def pull_arm(self) -> np.array:
        """ Pull an arm and the set of value of all the arms"""
        arms_value = np.random.normal(self.means, self.sigmas)
        idx = np.argmax(arms_value)
        return idx, arms_value

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])  # update mean of pulled arm
        n_samples = len(self.rewards_per_arm[pulled_arm])

        if n_samples > 1:  # update std of pulled arm
            self.sigmas[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm]) / n_samples

    def reset(self):
        super(GTS_Learner, self).reset()
        self.means = np.ones(self.n_arms) * self.prior_mean
        self.sigmas = np.ones(self.n_arms) * self.prior_sigma
