import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from utils import Pareto_metric

class random_policy(object):
    def __init__(self, MO_MAB):
        self.MO_MAB = MO_MAB
        self.t = 0
        self.mean_rewards = np.zeros((MO_MAB.K,MO_MAB.D))
        self.arms_counter = np.zeros(MO_MAB.K)
        self.arms_regrets = np.array([Pareto_metric(mu,MO_MAB.O_star) for mu in MO_MAB.O])

    def initialize(self):
        self.arms_counter = np.zeros(self.MO_MAB.K)
        for _ in range(self.MO_MAB.K):
            rand_ind = np.random.randint(self.MO_MAB.K)
            self.mean_rewards[rand_ind] = self.MO_MAB.A[rand_ind].sample()
            self.arms_counter[rand_ind] += 1
        self.t = self.MO_MAB.K

    def update(self):
        i = np.random.randint(self.MO_MAB.K)
        self.mean_rewards[i] *= self.arms_counter[i]
        self.mean_rewards[i] += self.MO_MAB.A[i].sample()
        self.arms_counter[i] += 1
        self.mean_rewards[i] /= self.arms_counter[i]
        self.t += 1
        return i

    def fairness(self):
        A_star_counter = np.array([self.arms_counter[i] for i in self.MO_MAB.A_star])
        return A_star_counter.var()

    def regret(self):
        return np.sum(self.arms_counter*self.arms_regrets)
