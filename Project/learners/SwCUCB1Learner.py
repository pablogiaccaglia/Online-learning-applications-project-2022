from learners.CUCB1Learner import CUCB1Learner
import math
import numpy as np

class SwCUCB1Learner(CUCB1Learner):

    def __init__(self, budgets, nCampaigns, windowSize):
        super().__init__(budgets, nCampaigns)
        self.windowSize = windowSize

    def resetWindow(self):
        self.numOfActions = np.zeros((self.nCampaigns, self.numOfBudgets))
        self.empiricalMeans = np.zeros((self.nCampaigns, self.numOfBudgets), dtype = np.float32)

    def findUnplayedArm(self):
        if np.any(self.numOfActions == 0):
            indexes = np.argwhere(self.numOfActions == 0)
            index = np.random.choice(indexes)
            campaignIndex = index[0]
            budgetIndex = index[1]
            return campaignIndex, budgetIndex

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

        if self.t % self.windowSize == 0:
            """
            If a new sliding window starts, empirical means and number of collected samples have to be re-initialized.
            """
            self.resetWindow()

        """ At each time step, if an arm with 0 collected samples exists, it is played"""
        if np.any(self.numOfActions == 0):
            campaignIndex, budgetIndex = self.findUnplayedArm()
            return self.getArbitrarySuperArm(idxCampaignToInclude = campaignIndex,
                                             idxBudgetToPlay = budgetIndex)

        # if all the arms have been pulled at least once, we select the super arm which maximizes the cumulative
        # upper confidence bound value,
        for campaignIdx in range(self.nCampaigns):
            for budgetIdx in range(self.numOfBudgets):
                self.ucbs[campaignIdx][budgetIdx] = self.computeUCB(campaignIdx = campaignIdx,
                                                                    budgetIdx = budgetIdx) \
                    if self.numOfActions[campaignIdx][budgetIdx] > 0 else np.inf

        # returns the best super arm based on the
        return self.getBestSuperArm()
