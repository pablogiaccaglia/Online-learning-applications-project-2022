from Part3.GTS_Learner import GTS_Learner
import numpy as np


class SwGTSLearner(GTS_Learner):

    def __init__(self, arms, prior_mean, prior_sigma = 1, window_size = 3):
        super().__init__(arms, prior_mean = prior_mean, prior_sigma = prior_sigma)
        self.window_size = window_size
        self.window_collected_rewards_per_arm = [[] for _ in range(len(arms))]

    def reset_window(self):
        # reset samples information
        self.window_collected_rewards_per_arm = [[] for _ in range(self.n_arms)]
        self.pulled_arms = []

    def pull_arm(self) -> np.array:
        """ Pull an arm and the set of value of all the arms"""

        if self.t % self.window_size == 0:
            """
            If a new sliding window starts, empirical means and number of collected samples have to be re-initialized.
            """
            self.reset_window()

        arms_value = np.random.normal(self.means, self.sigmas)
        idx = np.argmax(arms_value)
        return idx, arms_value

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.window_collected_rewards_per_arm[pulled_arm].append(reward)
        self.means[pulled_arm] = np.mean(self.window_collected_rewards_per_arm[pulled_arm])  # update mean of pulled arm
        n_samples = len(self.window_collected_rewards_per_arm[pulled_arm])

        if n_samples > 1:  # update std of pulled arm
            self.sigmas[pulled_arm] = np.std(self.window_collected_rewards_per_arm[pulled_arm]) / n_samples
