import numpy as np


class MoMAB():
    def __init__(self, A, A_star):
        self.A = A
        self.A_star = A_star
        self.K = len(self.A)
        self.O = np.array([arm.mean for arm in self.A])
        self.O_star = np.array([self.O[i] for i in self.A_star])
        self.D = len(self.O[0])
