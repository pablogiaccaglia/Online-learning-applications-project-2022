import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from entities.Utils import BanditNames
from learners.Learner import Learner


class GPUCB1_Learner(Learner):
    def __init__(self, arms, prior_mean, prior_sigma = 1, delta = 0.1,
                 cusum_args = None):  # arms are the budgets (e.g 0,10,20...)
        super().__init__(n_arms = len(arms), cusum_args = cusum_args, needs_boost = True)
        self.n_arms = len(arms)
        self.arms = arms
        self.means = np.ones(self.n_arms) * prior_mean
        self.sigmas = np.ones(self.n_arms) * prior_sigma
        self.ucbs = np.ones(self.n_arms) * np.inf
        self.pulled_arms = []
        self.prior_mean = prior_mean
        self.prior_sigma = prior_sigma
        self.bandit_name = BanditNames.GPUCB1_Learner.name

        """controls beta parameter"""
        self.delta = delta

        alpha = 0.5
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))  # to be adjusted
        self.gp = GaussianProcessRegressor(kernel = kernel,
                                           alpha = alpha ** 2,
                                           # normalize_y=True, #  TODO: SKLEARN NORMALIZATION DOES NOT WORK/I AM NOT
                                           #                          USING IT RIGHT. NORMALIZE Y MANUALLY
                                           n_restarts_optimizer = 9)

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])
        """        if not self.pulled_arms.any():
            self.pulled_arms = np.append(self.pulled_arms, gpucb1_super_arm)
        else:
            self.pulled_arms = np.append(np.atleast_2d(self.pulled_arms), np.atleast_2d(gpucb1_super_arm), axis=0)"""

    def update_ucbs(self):
        for armIdx in range(self.n_arms):
            self.ucbs[armIdx] = self.compute_UCB(armIdx)

    def compute_UCB(self, idx):
        if self.t == 0:
            beta = 2*np.log((np.power(1, 2)*np.power(np.pi, 2)*self.n_arms)/(6*self.delta))
        else:
            beta = 2*np.log((np.power(self.t, 2)*np.power(np.pi, 2)*self.n_arms)/(6*self.delta))
        return self.means[idx] + self.sigmas[idx] * np.sqrt(beta)

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        warnings.simplefilter(action = 'ignore', category = ConvergenceWarning)
        self.gp.fit(x, y)  # TODO: y IS NOT NORMALIZED. DO IT MANUALLY IF NECESSARY
        self.means, self.sigmas = self.gp.predict(
                np.atleast_2d(self.arms).T,
                return_std = True)
        # force sigma>0. It shouldn't be an issue anyway
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, rewards):
        self.t += 1
        self.update_observations(pulled_arm, rewards)
        self.update_model()

    # Same as gts_learner
    def pull_arm(self) -> np.array:
        """ Pull an arm and the set of value of all the arms"""
        arms_value = np.random.normal(self.means, self.sigmas)
        idx = np.argmax(arms_value)
        return idx, arms_value

    def reset(self):
        super(GPUCB1_Learner, self).reset()
        self.means = np.ones(self.n_arms) * self.prior_mean
        self.sigmas = np.ones(self.n_arms) * self.prior_sigma
        self.ucbs = np.ones(self.n_arms) * np.inf
        self.pulled_arms = []

        alpha = 0.5
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))  # to be adjusted
        self.gp = GaussianProcessRegressor(kernel = kernel,
                                           alpha = alpha ** 2,
                                           # normalize_y=True, #  TODO: SKLEARN NORMALIZATION DOES NOT WORK/I AM NOT
                                           #                          USING IT RIGHT. NORMALIZE Y MANUALLY
                                           n_restarts_optimizer = 9)
