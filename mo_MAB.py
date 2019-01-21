import numpy as np
from utils import alpha_star,G


class MoMAB():
    def __init__(self, A, A_star):
        self.A = A
        self.A_star = A_star
        self.K = len(self.A)
        self.O = np.array([arm.mean for arm in self.A])
        self.O_star = np.array([self.O[i] for i in self.A_star])
        self.D = self.O.shape[1]
        self.w = np.sort(np.array([2**(1-i) for i in range(self.D)]))
        self.alpha_star = alpha_star(self.O,self.w)
        self.optimal_mixed_rew = sum([ self.alpha_star[k]*self.O[k] for k in range(self.K) ]) #self.alpha_star.dot(self.O)[0]
        self.max_obj_fun = G(self.w,self.optimal_mixed_rew)
