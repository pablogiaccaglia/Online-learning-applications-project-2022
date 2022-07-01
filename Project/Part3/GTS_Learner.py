import Learner
import numpy as np
from Project.Knapsack import Knapsack
'''

Remark on the framework: BUDGET ASSIGNMENT(OPTIMIZATION)-> PLAY BUDGETS -> GET COMPOSITE REWARD -> UPDATE -> OPTIMIZATION...
OPTIMIZATION = call knapsack --> ok
PLAY BUDGET = play super-arm given by knapsack --> the super-arm is a value of the budget for each campaign
GET COMPOSITE REWARD = get environment feedback --> get rewards for each campaign and update the corresponding value
                                                    of the arms played. e.g I spent 30 for campaign 2, so I update only 
                                                    the corresponding "slot" of the gts_learner
UPDATE = update gts --> update means and sigmas with the rewards received
The call to Knapsack requires the full table to be filled. Therefore we first need to explore by playing randomly and
collecting various rewards.
The number of arms is an array of dimensions (#possible budgets allocations, #products).

WE ASSUME THAT THE BID IS OPTIMAL THROUGHOUT THE DAY, GIVEN THE BUDGET.
'''
# todo: how do I init the knapsack table? It should depend on the budget spent: e.g reward for budget=10< reward for budget=20
# todo: what's the feedback? I think it's the aggregated gross profit for each product (campaign).


class GTS_Learner(Learner):
    def __init__(self, arms, n_campaigns):  # arms are the budgets (e.g 0,10,20...)
        self.n_arms = len(arms)  # there should be another solution to pass from numerical budget to index
        super().__init__(self.n_arms, n_campaigns)
        self.means = np.ones((self.n_arms, n_campaigns)) * 30  # table of rewards for knapsack. Probably not zero??
        self.sigmas = np.ones((self.n_arms, n_campaigns)) * 20

    # The arm will be pulled according to the knapsack solution, pulling one arm (choosing one
    # budget) for each campaign. The last iteration of allocation from knapsack contains the best solution
    def pull_arm(self) -> np.array:
        k = Knapsack(rewards=np.random.normal(self.means, self.sigmas).clip(0.0), budgets=self.arms)
        k.solve()
        super_arm = k.allocations[-1][-1]
        return super_arm

    # super_arm like: [20,0,10,30,40] reward like: [80.0,0.0,101.4,200.3,20.3]
    def update_observations(self, super_arm, rewards):
        for campaign, budget in enumerate(super_arm):
            self.rewards_per_arm[np.where(self.arms == budget)][campaign].append(rewards[campaign])
        super().update_observations(rewards)

    def update(self, super_arm, rewards):
        self.update_observations(super_arm, rewards)
        for campaign, budget in enumerate(super_arm):
            arm = np.where(self.arms == budget)
            self.means[arm][campaign] = np.mean(self.rewards_per_arm[arm][campaign])
            n_samples = len(self.rewards_per_arm[arm][campaign])
            if n_samples > 1:
                self.sigmas[arm][campaign] = np.std(self.rewards_per_arm[arm][campaign])/n_samples
