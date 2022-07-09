from UCB1Learner import UCB1Learner
import numpy as np

class SwUCB1Learner(UCB1Learner):

    def __init__(self, budgets, windowSize):
        super().__init__(budgets)
        self.windowSize = windowSize

    def resetWindow(self):
        self.numOfActions = np.zeros(self.n_arms)
        self.empiricalMeans = np.zeros(self.n_arms, dtype = np.float32)

    def findUnplayedArm(self):
        if np.any(self.numOfActions == 0):
            indexes = np.argwhere(self.numOfActions == 0)
            index = np.random.choice(indexes)
            budgetIndex = index[1]
            return budgetIndex

    # the function selects the arm to pull at each time t according to the one with the maximum upper confidence bound
    def pull_arm(self):
        # We want to pull each arm at least once, so at round 0 we pull arm 0 etc..
        if self.t < self.n_arms:
            return self.t

        # if all the arms have been pulled at least once, we select the arm which maximizes the expectedRewards array
        # we are taking the indexes of the arm with the maximum expected reward, but since we could have multiple arms returned
        # we have to pick one randomly to pull

        if self.t % self.windowSize == 0:
            """
            If a new sliding window starts, empirical means and number of collected samples have to be re-initialized.
            """
            self.resetWindow()

        for i in range(0, self.n_arms):
            self.ucbs[i] = self.computeUCB(i) if self.numOfActions[i] > 0 else np.inf

        idxs = np.argmax(self.ucbs).reshape(-1)
        # print(idxs)
        return np.random.choice(idxs)
