import GPUCB1_Learner

class CusumGPUCB1Learner(GPUCB1_Learner):

    def __init__(self, arms, prior_mean, prior_sigma = 1, beta = 100.):
        super().__init__(arms, prior_mean, prior_sigma = 1, beta = 100.)
