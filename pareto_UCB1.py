import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from utils import Pareto_metric



class UCB1Pareto(object):
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
        self.arms_counter = np.ones(self.K)
        for i in range(self.K):
            self.mean_rewards[i] = self.A[i].sample()
        self.t = self.K

    def update(self):
        Pareto_Set = []
        mu = [self.mean_rewards[k]+np.sqrt((2/self.arms_counter[k])*np.log( self.t*(self.D*self.K)**0.25)) for k in range(self.K)]

        for i in range(self.K):
            optimal = True
            l = 0
            while optimal and l<self.K:
                if np.min(mu[l]-mu[i]) >= 0 and l!=i:
                    optimal = False
                l += 1
            if optimal:
                Pareto_Set.append(i)
        if Pareto_Set == []:
            for i in range(self.K):
                optimal = True
                l = 0
                while optimal and l<self.K:
                    if np.min(mu[l]-mu[i]) > 0 and l!=i:
                        optimal = False
                    l += 1
                if optimal:
                    Pareto_Set.append(i)

        i = np.random.choice(Pareto_Set)

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
