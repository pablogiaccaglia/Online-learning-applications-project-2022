import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Learner import Learner


class SwGPUCB1_Learner(Learner):
    def __init__(self, arms, prior_mean, prior_sigma = 1, window_size = 3, beta = 50.):  # arms are the budgets (e.g 0,10,20...)
        super().__init__(n_arms = len(arms), needs_boost = True)
        self.n_arms = len(arms)
        self.arms = arms
        self.means = np.ones(self.n_arms) * prior_mean
        self.sigmas = np.ones(self.n_arms) * prior_sigma
        self.prior_sigma = prior_sigma
        self.prior_mean = prior_mean
        self.pulled_arms = []  # One arm for campaign
        self.ucbs = np.ones(self.n_arms) * np.inf
        self.window_size = window_size
        self.window_collected_rewards = np.array([])
        self.bandit_name = 'SW-GP-UCB1'

        """beta (optional): Hyper-parameter to tune the exploration-exploitation
            balance. If beta is large, it emphasizes the variance of the unexplored
            solution solution (i.e. larger curiosity)"""
        self.beta = beta

        alpha = 0.5
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))  # to be adjusted
        self.gp = GaussianProcessRegressor(kernel = kernel,
                                           alpha = alpha ** 2,
                                           # normalize_y=True, #  TODO: SKLEARN NORMALIZATION DOES NOT WORK/I AM NOT
                                           #                          USING IT RIGHT. NORMALIZE Y MANUALLY
                                           n_restarts_optimizer = 9)

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.window_collected_rewards = np.append(self.window_collected_rewards, reward)
        self.pulled_arms.append(self.arms[pulled_arm])
        """        if not self.pulled_arms.any():
            self.pulled_arms = np.append(self.pulled_arms, gpucb1_super_arm)
        else:
            self.pulled_arms = np.append(np.atleast_2d(self.pulled_arms), np.atleast_2d(gpucb1_super_arm), axis=0)"""

    def update_ucbs(self):
        for armIdx in range(self.n_arms):
            self.ucbs[armIdx] = self.compute_UCB(armIdx)

    def compute_UCB(self, idx):
        return self.means[idx] + self.sigmas[idx] * np.sqrt(self.beta)

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.window_collected_rewards
        warnings.simplefilter(action = 'ignore', category = ConvergenceWarning)
        self.gp.fit(x, y)  # TODO: y IS NOT NORMALIZED. DO IT MANUALLY IF NECESSARY
        self.means, self.sigmas = self.gp.predict(
                np.atleast_2d(self.arms).T,
                return_std = True)
        # force sigma>0. It shouldn't be an issue anyway
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_super_arm, rewards):
        self.t += 1
        self.update_observations(pulled_super_arm, rewards)
        self.update_model()

    def reset_window(self):
        # reset samples information
        self.window_collected_rewards = np.array([])
        self.pulled_arms = []

    # Same as gts_learner
    def pull_arm(self) -> np.array:
        """ Pull an arm and the set of value of all the arms"""

        if self.t % self.window_size == 0:
            """
            If a new sliding window starts, collected rewards and pulled arms bandit info have to be re initialized.
            """
            self.reset_window()

        arms_value = np.random.normal(self.means, self.sigmas)
        idx = np.argmax(arms_value)
        return idx, arms_value

    def reset(self):
        super(SwGPUCB1_Learner, self).reset()
        self.means = np.ones(self.n_arms) * self.prior_mean
        self.sigmas = np.ones(self.n_arms) * self.prior_sigma
        self.pulled_arms = []  # One arm for campaign
        self.ucbs = np.ones(self.n_arms) * np.inf
        self.window_collected_rewards = np.array([])

        alpha = 0.5
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))  # to be adjusted
        self.gp = GaussianProcessRegressor(kernel = kernel,
                                           alpha = alpha ** 2,
                                           # normalize_y=True, #  TODO: SKLEARN NORMALIZATION DOES NOT WORK/I AM NOT
                                           #                          USING IT RIGHT. NORMALIZE Y MANUALLY
                                           n_restarts_optimizer = 9)