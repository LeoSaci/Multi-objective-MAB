import numpy as np
import mo_arms
import numpy.random as rd
from utils import lin_scal


class scal_UCB1(object):
    def __init__(self, A, weights_sets, A_star = None):
        self.t = 0
        self.A = A
        self.A_star = A_star
        self.K = len(self.A)
        self.S = len(weights_sets)
        self.weights_sets = weights_sets
        self.O = np.array([arm.mean for arm in self.A])
        self.O_star = np.array([self.O[i] for i in self.A_star])
        self.D = len(self.O[0])
        self.scal_rewards = np.array([[lin_scal(self.weights_sets[s],self.O[i]) for i in range(self.K)] for s in range(self.S)])
        self.mean_rewards = np.zeros((self.S,self.K,self.D))
        self.arms_regrets = np.array([np.max(self.scal_rewards[s])-self.scal_rewards[s] for s in range(self.S)])

        self.fun_counter = np.zeros(self.S)
        self.counter = np.zeros((self.S,self.K))
        self.arms_counter = np.zeros(self.K)

    def initialize(self):
        self.fun_counter = self.K*np.ones(self.S)
        self.counter = np.ones((self.S,self.K))
        self.arms_counter = self.S*np.ones(self.K)
        for s in range(self.S):
            for i in range(self.K):
                self.mean_rewards[s,i] = lin_scal(self.weights_sets[s],self.A[i].sample())
        self.t = self.K*self.S

    def update(self):
        s = rd.randint(self.S)
        i = np.argmax([lin_scal(self.weights_sets[s],self.mean_rewards[s,i]) + np.sqrt(2*np.log(self.fun_counter[s])/self.counter[s,i]) for i in range(self.K)])
        rew = lin_scal(self.weights_sets[s],self.A[i].sample())
        self.mean_rewards[s,i] = (self.mean_rewards[s,i]*self.counter[s,i]+rew)/(self.counter[s,i]+1)
        self.counter[s,i] += 1
        self.fun_counter[s] += 1
        self.arms_counter[i] += 1
        self.t += 1

        return i

    def fairness(self):
        A_star_occurences = np.array([self.arms_counter[i] for i in self.A_star])
        return A_star_occurences.var()

    def regret(self):
        return np.sum(self.counter*self.arms_regrets)
