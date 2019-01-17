import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from utils import Pareto_metric




class UCB1Pareto():
    def __init__(self, MoMAB):
        self.MoMAB = MoMAB
        self.t = 0
        self.mean_rewards = np.zeros((MoMAB.K,MoMAB.D))
        self.arms_counter = np.zeros(MoMAB.K)
        self.arms_regrets = np.array([Pareto_metric(mu,MoMAB.O_star) for mu in MoMAB.O])

    def initialize(self):
        self.arms_counter = np.ones(self.MoMAB.K)
        for i in range(self.MoMAB.K):
            self.mean_rewards[i] = self.MoMAB.A[i].sample()
        self.t = self.MoMAB.K

    def update(self):
        Pareto_Set = []
        mu = [self.mean_rewards[k]+np.sqrt((2/self.arms_counter[k])*np.log( self.t*(self.MoMAB.D*self.MoMAB.K)**0.25)) for k in range(self.MoMAB.K)]

        for i in range(self.MoMAB.K):
            optimal = True
            l = 0
            while optimal and l<self.MoMAB.K:
                if np.min(mu[l]-mu[i]) >= 0 and l!=i:
                    optimal = False
                l += 1
            if optimal:
                Pareto_Set.append(i)
        if Pareto_Set == []:
            for i in range(self.MoMAB.K):
                optimal = True
                l = 0
                while optimal and l<self.MoMAB.K:
                    if np.min(mu[l]-mu[i]) > 0 and l!=i:
                        optimal = False
                    l += 1
                if optimal:
                    Pareto_Set.append(i)

        i = np.random.choice(Pareto_Set)

        self.mean_rewards[i] *= self.arms_counter[i]
        self.mean_rewards[i] += self.MoMAB.A[i].sample()
        self.arms_counter[i] += 1
        self.mean_rewards[i] /= self.arms_counter[i]

        self.t += 1
        return i

    def fairness(self):
        A_star_counter = np.array([self.arms_counter[i] for i in self.MoMAB.A_star])
        return A_star_counter.var()

    def regret(self):
        return np.sum(self.arms_counter*self.arms_regrets)
