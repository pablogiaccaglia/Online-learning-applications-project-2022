import Learner
import numpy as np
from Project.Knapsack import Knapsack
'''

Remark on the framework: BUDGET ASSIGNMENT(OPTIMIZATION)-> PLAY BUDGETS -> GET COMPOSITE REWARD -> UPDATE -> OPTIMIZATION...
OPTIMIZATION = call knapsack --> ok
PLAY BUDGET = play arms given by knapsack --> what are the arms? how many?
GET COMPOSITE REWARD = get environment feedback --> ok (???). we don't know the alphas but it should not be a problem.
UPDATE = update gp --> same problem of PLAY BUDGET
The call to Knapsack requires the full table to be filled. therefore we first need to explore by playing randomly and
collecting various rewards.
I think the number of arms is an array of dimensions (#possible budgets allocations, #products) and we need to play
the one yielding the best reward for every product, staying within the limit budget.

WE ASSUME THAT THE BID IS OPTIMAL THROUGHOUT THE DAY, GIVEN THE BUDGET
'''
# todo: how do I init the knapsack table?
# todo: what's the feedback? I think it's the aggregated gross profit for each product (campaign).
#       In this case I would divide by the sum of users' expected profit to estimate the alphas.
# todo: what are arms in this case? If it's a discrete number of bids, GP_Campaign should be modified accordingly.


class GP_Learner(Learner):
    def __init__(self, arms,n_campaigns): # arms are the budgets (e.g 0,10,20...)
        self.n_arms = len(arms)
        super().__init__(self.n_arms)  # not sure we should keep this
        self.means = np.ones((self.n_arms, n_campaigns)) * 3e1  # table of rewards for knapsack. Probably not zero??
        self.sigmas = np.ones((self.n_arms, n_campaigns)) * 2e1

    # todo: fix this. The arm will be pulled according to the knapsack solution, pulling one arm (choosing one
    #       budget) for each campaign. We will have to look at the last iteration of allocation
    def pull_arm(self) -> np.array:
        k = Knapsack(rewards=np.random.normal(self.means,self.sigmas).clip(0.0), budgets=self.arms)
        k.solve()
        idx = k.allocations[-1]
        #idx = np.argmax(np.random.normal(self.means, self.sigmas), axis=1)
        return idx


    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        n_samples = len(self.rewards_per_arm[pulled_arm])

        if n_samples > 1:
            self.sigmas[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm]) / n_samples