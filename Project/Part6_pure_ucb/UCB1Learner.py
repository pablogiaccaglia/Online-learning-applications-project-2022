from Learner import Learner
import math
import numpy as np
from Knapsack import Knapsack


class UCB1Learner(Learner):

    def __init__(self, budgets):
        super().__init__(budgets)

        self.numOfActions = np.zeros(self.n_arms)
        self.empiricalMeans = np.zeros(self.n_arms, dtype = np.float32)

        self.ucbs = np.ones(self.n_arms, dtype = np.float32) * np.inf

        self.K = Knapsack()

    def update_observations(self, pulled_arm, reward) -> None:
        self.t += 1
        super().update_observations(pulled_arm = pulled_arm, reward = reward)
        self.updateMean(budgetIdx = pulled_arm)
        self.numOfActions[pulled_arm] += 1

    def getBestSuperArm(self):

        """the best super arm is retrieved as follows:
         - Knapsack problem is solved with the empirical means of the profits
         - the last row is considered to compute the difference between the total profit and the budget
         - for each of the net profits the cumulative ucb values are considered to determine the best super arm

         NOTE: to have a more fine grained selection of the super arm, from each UCB the value of the budget spent is
         subtracted"""

        self.K.reset(rewards = self.empiricalMeans, budgets = np.array(self.budgets))
        self.K.solve()

        dp_table, allocations = self.K.get_output()
        lastRowAllocations = allocations[-1][1:]
        cumulativeUcbs = np.zeros_like(self.budgets, dtype = np.float32)

        for totalBudgetIdx, budgetAllocation in enumerate(lastRowAllocations):
            for campaignIdx, campaignAllocation in enumerate(budgetAllocation):  # campaign allocation
                budgetIdx = self.budgetsDict[campaignAllocation]
                netUcb = self.ucbs[campaignIdx][
                    budgetIdx]  # - campaignAllocation # TODO UNCOMMENT ONCE UNIFORMED WITH SIMULATION ENV
                # print(netUcb)
                cumulativeUcbs[totalBudgetIdx] += netUcb  # store cumulative ucb of a given super arm

        bestSuperArmIdx = np.argmax(cumulativeUcbs)
        if isinstance(bestSuperArmIdx, np.ndarray):
            bestSuperArmIdx = np.random.choice(bestSuperArmIdx)

        return lastRowAllocations[bestSuperArmIdx].astype(np.int32)

    def updateMean(self, budgetIdx) -> None:
        self.empiricalMeans[budgetIdx] = np.mean(self.rewards_per_arm[budgetIdx])

    def computeUCB(self, budgetIdx):
        return self.empiricalMeans[budgetIdx] + math.sqrt((2 * math.log(self.t)) / self.numOfActions[budgetIdx])

    # the function selects the arm to pull at each time t according to the one with the maximum upper confidence bound
    def pull_arm(self):
        # We want to pull each arm at least once, so at round 0 we pull arm 0 etc..
        if self.t < self.n_arms:
            return self.t

        # if all the arms have been pulled at least once, we select the arm which maximizes the expectedRewards array
        # we are taking the indexes of the arm with the maximum expected reward, but since we could have multiple arms returned
        # we have to pick one randomly to pull

        for i in range(0, self.n_arms):
            self.ucbs[i] = self.computeUCB(i) if self.numOfActions[i] > 0 else np.inf

        idxs = np.argmax(self.ucbs).reshape(-1)
        # print(idxs)
        return np.random.choice(idxs)
