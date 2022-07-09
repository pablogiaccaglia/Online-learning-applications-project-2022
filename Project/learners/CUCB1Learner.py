from learners.Learner import Learner
import math
import numpy as np
import itertools
from Knapsack import Knapsack


class CUCB1Learner(Learner):

    def __init__(self, budgets, nCampaigns):
        super().__init__(budgets, nCampaigns)

        self.numOfArms = self.numOfBudgets * nCampaigns  # overall number of arms is self.numOfBudgets * nCampaigns

        self.numOfActions = np.zeros((nCampaigns, self.numOfBudgets))
        self.empiricalMeans = np.zeros((nCampaigns, self.numOfBudgets), dtype = np.float32)

        self.ucbs = np.ones((nCampaigns, self.numOfBudgets), dtype = np.float32) * np.inf

        # self.superArms = self.__get_super_arms()
        self.K = Knapsack()

    def update_observations(self, pulledSuperArm, rewards) -> None:
        self.t += 1
        super().update_observations(pulledSuperArm = pulledSuperArm, rewards = rewards)
        for campaignIdx, budget in enumerate(pulledSuperArm):
            budgetIdx = self.budgetsDict[budget]
            self.updateMean(campaignIdx = campaignIdx, budgetIdx = budgetIdx)
            self.numOfActions[campaignIdx][budgetIdx] += 1

    def __get_super_arms(self):
        maxBudget = np.max(self.budgets)
        superArms = []
        for v in itertools.product(self.budgets, repeat = self.nCampaigns):
            if sum(v) <= maxBudget:
                superArms.append(v)

        return superArms

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

    def updateMean(self, campaignIdx, budgetIdx) -> None:
        self.empiricalMeans[campaignIdx][budgetIdx] = np.mean(self.rewards[campaignIdx][budgetIdx])

    def computeUCB(self, campaignIdx, budgetIdx):
        return self.empiricalMeans[campaignIdx][
            budgetIdx] + math.sqrt((2 * math.log(self.t)) / self.numOfActions[campaignIdx][budgetIdx])

    def getArbitrarySuperArm(self, idxCampaignToInclude, idxBudgetToPlay):

        """returns a list of arms to pull, where the given arm is included.
           the "arbitrary" super arm is such that all the budget is allocated to the specified campaign.
           the superArm is an array of budgets

           example:
                    idxCampaignToInclude = 0
                    idxBudgetToPlay = 2
                    budgets = [0,10,20,30]
                    [20, 0, 0, ...] -> campaign 0 has budget 20 allocated

           """
        budget = self.budgets[idxBudgetToPlay]
        superArm = [budget if i == idxCampaignToInclude else 0 for i in range(self.nCampaigns)]
        return np.array(superArm)

    def pullSuperArm(self):
        # the function selects the superArm to pull at each time t according to the one with
        # the maximum cumulative upper confidence bound

        # We want to pull each arm at least once, so for each arm i, play an arbitrary super arm S ∈ S
        # such that i ∈ S. At each time step we play a superArm, meaning that we need to perform. So we need a
        # number of warmup rounds equal to total number of arms.

        if self.t < self.numOfArms:

            """
            Each arm is played once at each time step until t = numOfArms-1.
            Rule to retrieve campaign and budget (arm) to play:
            
            - campaign Index = time step / numOfBudgets
            - budget Index = time step % numOfBudgets
            
            """
            campaignIdx = math.floor(self.t / self.numOfBudgets)
            budgetIdx = self.t % self.numOfBudgets
            return self.getArbitrarySuperArm(idxCampaignToInclude = campaignIdx,
                                             idxBudgetToPlay = budgetIdx)

        # if all the arms have been pulled at least once, we select the super arm which maximizes the cumulative
        # upper confidence bound value,
        for campaignIdx in range(self.nCampaigns):
            for budgetIdx in range(self.numOfBudgets):
                self.ucbs[campaignIdx][budgetIdx] = self.computeUCB(campaignIdx = campaignIdx,
                                                                    budgetIdx = budgetIdx) \
                    if self.numOfActions[campaignIdx][budgetIdx] > 0 else np.inf

        # returns the best super arm based on the
        return self.getBestSuperArm()
