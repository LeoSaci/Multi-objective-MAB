import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from utils import Pareto_metric

class random_policy(object):
    def __init__(self, A, A_star):
        self.t = 0
        self.A = A
        self.A_star = A_star
        self.K = len(self.A)
        self.O = np.array([arm.mean for arm in self.A])
        self.O_star = np.array([self.O[i] for i in self.A_star])
        self.D = len(self.O[0])
        self.mean_rewards = np.zeros((self.K,self.D))
        self.arms_counter = np.zeros(self.K)
        self.arms_regrets = np.array([Pareto_metric(mu,self.O_star) for mu in self.O])

    def initialize(self):
        self.arms_counter = np.zeros(self.K)
        for _ in range(self.K):
            rand_ind = np.random.randint(self.K)
            self.mean_rewards[rand_ind] = self.A[rand_ind].sample()
            self.arms_counter[rand_ind] += 1
        self.t = self.K

    def update(self):
        i = np.random.randint(self.K)
        self.mean_rewards[i] *= self.arms_counter[i]
        self.mean_rewards[i] += self.A[i].sample()
        self.arms_counter[i] += 1
        self.mean_rewards[i] /= self.arms_counter[i]
        self.t += 1
        return i

    def fairness(self):
        A_star_counter = np.array([self.arms_counter[i] for i in self.A_star])
        return A_star_counter.var()

    def regret(self):
        return np.sum(self.arms_counter*self.arms_regrets)
