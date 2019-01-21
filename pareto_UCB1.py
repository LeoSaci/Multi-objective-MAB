import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from utils import Pareto_metric,G


class UCB1Pareto():
    def __init__(self, MoMAB):
        self.MoMAB = MoMAB
        self.t = 0
        self.mean_rewards = np.zeros((MoMAB.K,MoMAB.D))
        self.arms_counter = np.zeros(MoMAB.K)
        self.arms_regrets = np.array([Pareto_metric(mu,MoMAB.O_star) for mu in MoMAB.O])
        self.sum_rew = np.zeros(MoMAB.D)

    def initialize(self):
        self.arms_counter = np.ones(self.MoMAB.K)
        self.sum_rew = np.zeros(self.MoMAB.D)
        for i in range(self.MoMAB.K):
            self.mean_rewards[i] = self.MoMAB.A[i].sample()
            self.sum_rew += self.mean_rewards[i]
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
        sample = self.MoMAB.A[i].sample()
        self.mean_rewards[i] += sample
        self.arms_counter[i] += 1
        self.mean_rewards[i] /= self.arms_counter[i]
        self.sum_rew += sample
        self.t += 1
        return i

    def fairness(self):
        A_star_counter = np.array([self.arms_counter[i] for i in self.MoMAB.A_star])
        return A_star_counter.var()

    def regret(self):
        return np.sum(self.arms_counter*self.arms_regrets)

    def regret_ogde(self,ogde):
        mean_reward = self.sum_rew/self.t
        return self.MoMAB.max_obj_fun - G(MO_MAB.w,mean_reward)
