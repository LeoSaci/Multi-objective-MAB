import numpy as np
import mo_arms
from utils import reordonate,simplex_proj



class OGDE(object):
    def __init__(self,A, A_star, w, delta):
        self.t = 0
        self.A = A
        self.A_star = A_star
        self.K = len(self.A)
        self.O = np.array([arm.mean for arm in self.A])
        self.O_star = np.array([self.O[i] for i in self.A_star])
        self.D = len(self.O[0])
        self.w = w.reshape((1,self.D))
        self.delta = delta
        self.mu = np.zeros((self.D,self.K))
        self.alpha = (1/self.K)*np.ones(self.K)
        self.arms_counter = np.zeros(self.K)

    def G_w(mu):
        return (self.w).dot(reordonate(mu).reshape((self.D,1)))[0,0]

    def initialize(self):
        self.arms_counter = np.ones(self.K)
        for i in range(self.K):
            self.mu[:,i] = self.A[i].sample()
        self.t = self.K

    def update(self):
        beta = np.zeros(self.K+1)
        beta[1:] = np.cumsum(self.alpha)
        r = np.random.rand()
        i = np.argmax([int(beta[i] <= r and r <= beta[i+1]) for i in range(self.K)])
        sample = self.A[i].sample()
        self.mu[:,i] = self.mu[:,i]*self.arms_counter[i] + sample
        self.arms_counter[i] += 1
        self.mu[:,i] /= self.arms_counter[i]
        eta = np.sqrt(2)/(1-1/np.sqrt(self.K))*np.sqrt(np.log(2/self.delta)/self.t)
        self.alpha = simplex_proj(eta/self.K,self.alpha + eta*self.w.dot(self.mu)[0])
        self.t += 1
        return i

    def fairness(self):
        A_star_occurences = np.array([self.arms_counter[i] for i in self.A_star])
        return A_star_occurences.var()

    def regret(self):
        return 0
