import numpy as np
import mo_arms
import numpy.random as rd
from utils import lin_scal


class scal_UCB1(object):
    def __init__(self, MO_MAB, weights_sets):
        self.MO_MAB = MO_MAB
        self.t = 0
        self.S = len(weights_sets)
        self.weights_sets = weights_sets
        self.scal_rewards = np.array([[lin_scal(self.weights_sets[s],MO_MAB.O[i]) for i in range(MO_MAB.K)] for s in range(self.S)])
        self.mean_rewards = np.zeros((self.S,MO_MAB.K,MO_MAB.D))
        self.arms_regrets = np.array([np.max(self.scal_rewards[s])-self.scal_rewards[s] for s in range(self.S)])

        self.fun_counter = np.zeros(self.S)
        self.counter = np.zeros((self.S,MO_MAB.K))
        self.arms_counter = np.zeros(MO_MAB.K)

    def initialize(self):
        self.fun_counter = self.MO_MAB.K*np.ones(self.S)
        self.counter = np.ones((self.S,self.MO_MAB.K))
        self.arms_counter = self.S*np.ones(self.MO_MAB.K)
        for s in range(self.S):
            for i in range(self.MO_MAB.K):
                self.mean_rewards[s,i] = lin_scal(self.weights_sets[s],self.MO_MAB.A[i].sample())
        self.t = self.MO_MAB.K*self.S

    def update(self):
        s = rd.randint(self.S)
        i = np.argmax([lin_scal(self.weights_sets[s],self.mean_rewards[s,i]) + np.sqrt(2*np.log(self.fun_counter[s])/self.counter[s,i]) for i in range(self.MO_MAB.K)])
        rew = lin_scal(self.weights_sets[s],self.MO_MAB.A[i].sample())
        self.mean_rewards[s,i] = (self.mean_rewards[s,i]*self.counter[s,i]+rew)/(self.counter[s,i]+1)
        self.counter[s,i] += 1
        self.fun_counter[s] += 1
        self.arms_counter[i] += 1
        self.t += 1

        return i

    def fairness(self):
        A_star_occurences = np.array([self.arms_counter[i] for i in self.MO_MAB.A_star])
        return A_star_occurences.var()

    def regret(self):
        return np.sum(self.counter*self.arms_regrets)
