import numpy as np

class CUSUM:

    def __init__(self, samplesForRefPoint, epsilon, detectionThreshold):
        self.samplesForRefPoint = samplesForRefPoint
        self.epsilon = epsilon
        self.detectionThreshold = detectionThreshold
        self.referencePoint = 0
        self.gPlus = 0
        self.gMinus = 0
        self.t = 0

    def update(self, sample):
        self.t += 1
        if self.t <= self.samplesForRefPoint:
            self.referencePoint += sample/self.samplesForRefPoint
            return 0

        else:
            sPlus = (sample - self.referencePoint) - self.epsilon
            sMinus = (sample - self.referencePoint) + self.epsilon
            self.gPlus = max(0, self.gPlus + sPlus)
            self.gMinus = max(0, self.gMinus + sMinus)
            return self.gPlus > self.detectionThreshold or self.gMinus > self.detectionThreshold

    def reset(self):
        self.t = 0
        self.gPlus = 0
        self.gMinus = 0






