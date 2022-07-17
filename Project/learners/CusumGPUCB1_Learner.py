import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from entities.Utils import BanditNames
from learners.GPUCB1_Learner import GPUCB1_Learner


class CusumGPUCB1Learner(GPUCB1_Learner):

    def __init__(self, arms,
                 prior_mean,
                 prior_sigma = 1,
                 samplesForRefPoint = 100,
                 epsilon = 0.05,
                 detectionThreshold = 20,
                 explorationAlpha = 0.01,
                 delta = 0.1):

        cusum_args = {"samplesForRefPoint": samplesForRefPoint,
                        "epsilon":            epsilon,
                        "detectionThreshold": detectionThreshold,
                        "explorationAlpha":   explorationAlpha}

        super().__init__(arms, prior_mean, prior_sigma = prior_sigma, delta = delta, cusum_args = cusum_args)

        self.bandit_name = BanditNames.CusumGPUCB1Learner.name

    # Same as gts_learner
    def pull_arm(self) -> np.array:
        arms_value = np.random.normal(self.means, self.sigmas)

        """ Pull an arm and the set of value of all the arms"""

        """ With probability 1- alpha exploit,
            with probability alpha explore """

        if np.random.binomial(1, 1 - self.explorationAlpha):
            idx = np.argmax(arms_value)
        else:
            idx = np.random.choice(self.n_arms)

        return idx, arms_value

    def update(self, pulled_arm, reward):
        self.t += 1
        if self.change_detection[pulled_arm].update(reward):
                self.detections[pulled_arm].append(self.t)
                self.valid_collected_rewards = []
                self.pulled_arms = []
                self.change_detection[pulled_arm].reset()

        self.update_observations(pulled_arm, reward)
        self.update_model()

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.valid_collected_rewards
        warnings.simplefilter(action = 'ignore', category = ConvergenceWarning)
        self.gp.fit(x, y)  # TODO: y IS NOT NORMALIZED. DO IT MANUALLY IF NECESSARY
        self.means, self.sigmas = self.gp.predict(
                np.atleast_2d(self.arms).T,
                return_std = True)
        # force sigma>0. It shouldn't be an issue anyway
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def reset(self):
        super(CusumGPUCB1Learner, self).reset()