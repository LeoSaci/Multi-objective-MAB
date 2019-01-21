import numpy as np
import mo_arms
from utils import reordonate,simplex_proj,alpha_star,Pareto_metric,decreasing_order,reordonate


class OGDE(object):
    def __init__(self,MoMAB, w, delta):
        self.MoMAB = MoMAB
        self.t = 0
        self.w = MoMAB.w.reshape((1,MoMAB.D))
        self.delta = delta
        self.mu = np.zeros((MoMAB.D,MoMAB.K))
        self.alpha = (1/MoMAB.K)*np.ones(MoMAB.K)
        self.arms_counter = np.zeros(MoMAB.K)
        self.arms_regrets = np.array([Pareto_metric(mu,MoMAB.O_star) for mu in MoMAB.O])

    def G_w(self,mu):
        mu_prime = list(mu)
        mu_prime.sort()
        mu_prime.reverse()
        mu_prime = np.array(mu_prime).reshape((self.MoMAB.D,1))
        return (self.w).dot(mu_prime)[0,0]

    def initialize(self):
        self.arms_counter = np.ones(self.MoMAB.K)
        for i in range(self.MoMAB.K):
            self.mu[:,i] = 1-self.MoMAB.A[i].sample()
        self.t = self.MoMAB.K

    def update(self):
        self.t += 1
        beta = np.zeros(self.MoMAB.K+1)
        beta[1:] = np.cumsum(self.alpha)
        r = np.random.rand()
        i = np.argmax([int(beta[i] < r and r <= beta[i+1]) for i in range(self.MoMAB.K)])
        sample = 1-self.MoMAB.A[i].sample()
        self.mu[:,i] = self.mu[:,i]*self.arms_counter[i] + sample
        self.arms_counter[i] += 1
        self.mu[:,i] /= self.arms_counter[i]
        eta = np.sqrt(2)/(1-1/np.sqrt(self.MoMAB.K))*np.sqrt(np.log(2/self.delta)/self.t)
        sigma = decreasing_order(self.mu,self.alpha)
        sigma = list(sigma)
        sigma.reverse()
        sigma = np.array(sigma)
        self.alpha = simplex_proj(eta/self.MoMAB.K , self.alpha - eta* self.w.dot(reordonate(sigma,self.mu))[0] )
        return i

    def regret(self):
        mean_reward = sum(((1-self.mu)*self.arms_counter).T)/self.t
        return self.MoMAB.max_obj_fun - self.G_w(mean_reward)

    def regret_pareto(self):
        return np.vdot(self.arms_counter,self.arms_regrets)#np.sum(self.arms_counter*self.arms_regrets)
