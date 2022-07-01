import numpy as np
import Learner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Project.Knapsack import Knapsack


class GPTS_Learner(Learner):
    def __init__(self, arms, n_campaigns):  # arms are the budgets (e.g 0,10,20...)
        self.n_arms = len(arms)  # there should be another solution to pass from numerical budget to index
        super().__init__(self.n_arms, n_campaigns)
        self.arms = arms
        self.means = np.ones((self.n_arms, n_campaigns)) * 30  # table of rewards for knapsack. Probably not zero??
        self.sigmas = np.ones((self.n_arms, n_campaigns)) * 20
        self.pulled_arms = []  # bids history. One arm for campaign

        alpha = 1.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))  # to be adjusted
        self.gp = GaussianProcessRegressor(kernel=kernel,
                                           alpha=alpha ** 2,
                                           normalize_y=True,
                                           n_restarts_optimizer=9)

    def update_observations(self, super_arm, rewards):
        for campaign, budget in enumerate(super_arm):
            self.rewards_per_arm[np.where(self.arms == budget)][campaign].append(rewards[campaign])
        super().update_observations(rewards)

    #todo: modify from this method
    # I think I should use a for cycle iterating over the campaign, i.e. x are the pulled arms, y are their rewards
    def update_model(self):
        for campaign in range(self.n_campaigns):
            x = np.array(self.pulled_arms)[:, campaign]
            x = np.atleast_2d(x).T  # todo: test if this still works
            y = self.collected_rewards[:, campaign]
            self.gp.fit(x, y)
            self.means[:, campaign], self.sigmas[:,campaign] = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
            self.sigmas[:, campaign] = np.maximum(self.sigmas, 1e-2)  # force sigma>0. It shouldn't be an issue anyway

    def update(self, pulled_super_arm, rewards):
        self.t += 1
        self.update_observations(pulled_super_arm, rewards)
        self.update_model()

    # Same as gts_learner
    def pull_arm(self) -> np.array:
        k = Knapsack(rewards=np.random.normal(self.means, self.sigmas).clip(0.0), budgets=self.arms)
        k.solve()
        super_arm = k.allocations[-1][-1]
        return super_arm
