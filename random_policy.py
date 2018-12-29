import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from utils import Pareto_metric

class random_policy(object):
    def __init__(self, A, A_star = None):
        self.t = 0
        self.A = A
        self.A_star = A_star
        self.K = len(self.A)
        self.O = np.array([arm.mean for arm in self.A])
        self.O_star = np.array([self.O[i] for i in self.A_star])
        self.D = len(self.O[0])
        self.sumRewards = np.zeros((self.K,self.D))
        self.ArmsOccurences = np.zeros(self.K)
        self.arms_regrets = np.array([Pareto_metric(mu,self.O_star) for mu in self.O])

    def initialize(self):
        self.ArmsOccurences = np.zeros(self.K)
        for _ in range(self.K):
            rand_ind = np.random.randint(self.K)
            self.sumRewards[rand_ind] = self.A[rand_ind].sample()
            self.ArmsOccurences[rand_ind] += 1
        self.t = self.K

    def update(self):
        i = np.random.randint(self.K)
        self.ArmsOccurences[i] += 1
        self.sumRewards[i] += self.A[i].sample()
        self.t += 1
        return i

    def fairness(self):
        A_star_occurences = np.array([self.ArmsOccurences[i] for i in self.A_star])
        return A_star_occurences.var()

    def regret(self):
        return np.sum(self.ArmsOccurences*self.arms_regrets)
