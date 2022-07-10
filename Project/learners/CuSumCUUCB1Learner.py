from learners.CUCB1Learner import CUCB1Learner
from CUSUM import CUSUM


class CuSumCUB1Learner(CUCB1Learner):

    def __init__(self,
                 budgets,
                 nCampaigns,
                 samplesForRefPoint = 100,
                 epsilon = 0.05,
                 detectionThreshold = 20,
                 explorationAlpha = 0.01):
        super().__init__(budgets, nCampaigns)
        [[[] for _ in range(n_arms)] for _ in range(n_campaigns)]
        self.changeDetection[[CUSUM(samplesForRefPoint, epsilon, detectionThreshold)]]
