import numpy as np
from Learner import Learner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Project.Knapsack import Knapsack


class GPTS_Learner(Learner):
    def __init__(self, arms, n_campaigns):  # arms are the budgets (e.g 0,10,20...)
        self.n_arms = len(arms)  # there should be another solution to pass from numerical budget to index
        super().__init__(self.n_arms, n_campaigns)
        self.arms = arms
        self.means = np.ones((n_campaigns, self.n_arms)) * 30  # table of rewards for knapsack. Probably not zero??
        self.sigmas = np.ones((n_campaigns, self.n_arms)) * 20
        self.pulled_arms = np.array([])  # bids history. One arm for campaign

        alpha = 1.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))  # to be adjusted
        self.gp = GaussianProcessRegressor(kernel=kernel,
                                           alpha=alpha ** 2,
                                           #normalize_y=True, #  TODO: SKLEARN NORMALIZATION DOES NOT WORK/I AM NOT
                                           #                          USING IT RIGHT. NORMALIZE Y MANUALLY
                                           n_restarts_optimizer=9)

    def update_observations(self, super_arm, rewards):
        # I believe rewards_per_arm is useless here
        # for campaign, budget in enumerate(super_arm):
        #     self.rewards_per_arm[campaign][self.arms.index(budget)].append(rewards[campaign])
        super().update_observations(rewards)
        if not self.pulled_arms.any():
            self.pulled_arms = np.append(self.pulled_arms, super_arm)
        else:
            self.pulled_arms = np.append(np.atleast_2d(self.pulled_arms), np.atleast_2d(super_arm), axis=0)

    def update_model(self):
        for campaign in range(self.n_campaigns):
            x = (np.atleast_2d(self.pulled_arms)[:, campaign])  # todo: I removed the transpose .T, was it necessary?
            y = np.atleast_2d(self.collected_rewards)[:, campaign]
            self.gp.fit(X=np.atleast_2d(x).T, y=y)  # TODO: y IS NOT NORMALIZED. DO IT MANUALLY IF NECESSARY
            self.means[campaign, :], self.sigmas[campaign, :] = self.gp.predict(np.atleast_2d(self.arms).T,
                                                                                return_std=True)
            # force sigma>0. It shouldn't be an issue anyway
            self.sigmas[campaign, :] = np.maximum(self.sigmas[campaign, :], 1e-2)

    def update(self, pulled_super_arm, rewards):
        self.t += 1
        self.update_observations(pulled_super_arm, rewards)
        self.update_model()

    # Same as gts_learner
    def pull_arm(self) -> np.array:
        k = Knapsack(rewards=np.array(np.random.normal(self.means, self.sigmas).clip(0.0)), budgets=np.array(self.arms))
        k.solve()
        super_arm = k.allocations[-1][-1][1:]  # last row of allocations minus the no campaign case
        return super_arm
