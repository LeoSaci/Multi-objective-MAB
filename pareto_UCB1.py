import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from utils import Pareto_metric



class UCB1Pareto(object):
    def __init__(self, A, A_star = None):
        self.t = 0
        self.A = A
        self.A_star = A_star
        self.K = len(self.A)
        self.O = np.array([arm.mean for arm in self.A])
        self.O_star = np.array([self.O[i] for i in self.A_star])
        self.D = len(self.O[0])
        self.sumRewards = np.zeros((self.K,self.D))
        self.ArmsOccurences = np.zeros(self.K)
        self.arms_regrets = np.array([Pareto_metric(mu,self.O_star) for mu in self.O])

    def initialize(self):
        self.ArmsOccurences = np.ones(self.K)
        for i in range(self.K):
            self.sumRewards[i] = self.A[i].sample()
        self.t = self.K

    def update(self):
        Pareto_Set = []
        #tab = [np.sqrt((2/self.ArmsOccurences[k])*np.log( self.t*(self.D*self.K)**0.25)) for k in range(self.K)]
        mu = [self.sumRewards[k]/self.ArmsOccurences[k]+np.sqrt((2/self.ArmsOccurences[k])*np.log( self.t*(self.D*self.K)**0.25)) for k in range(self.K)]
        #print(tab)
        #print(mu)
        for i in range(self.K):
            optimal = True
            l = 0
            while optimal and l<self.K:
                #print(i+1,self.K)
                if np.min(mu[l]-mu[i]) >= 0 and l!=i:
                    optimal = False
                l += 1
            if optimal:
                Pareto_Set.append(i)
        #if int(self.t/200)==self.t/200:
        #    print(Pareto_Set)
        if Pareto_Set == []:
            #print('Pareto Set Empty, at t = '+str(self.t))
            for i in range(self.K):
                optimal = True
                l = 0
                while optimal and l<self.K:
                    #print(i+1,self.K)
                    if np.min(mu[l]-mu[i]) > 0 and l!=i:
                        optimal = False
                    l += 1
                if optimal:
                    Pareto_Set.append(i)

        i = np.random.choice(Pareto_Set)

        self.ArmsOccurences[i] += 1
        self.sumRewards[i] += self.A[i].sample()
        self.t += 1
        return i

    def fairness(self):
        A_star_occurences = np.array([self.ArmsOccurences[i] for i in self.A_star])
        return A_star_occurences.var()

    def regret(self):
        return np.sum(self.ArmsOccurences*self.arms_regrets)


def plot_Pareto_frontier(algorithm):
    O = algorithm.O
    arms_regrets = algorithm.arms_regrets
    virtual_rewards = [O[i]+arms_regrets[i] for i in range(len(O))]
    tab = np.array([virtual_rewards[i][1]/virtual_rewards[i][0] for i in range(len(O))])
    pente_max,pente_min = tab[np.argmax(tab)],tab[np.argmin(tab)]
    angles = np.linspace(np.arctan(pente_min),np.arctan(pente_max),100)
    O_star = algorithm.O_star
    frontier_points = np.array([[np.cos(angles[i]),np.sin(angles[i])] for i in range(len(angles))])
    for i in range(len(angles)):
        eps = Pareto_metric(frontier_points[i],O_star)
        frontier_points[i] += eps*np.ones(2)

    frontier_points = frontier_points.T
    plt.figure(0)
    plt.plot(frontier_points[0],frontier_points[1], color = 'g', label = 'Pareto frontier')
    plt.scatter([O[i][0] for i in range(len(O))], [O[i][1] for i in range(len(O))] ,marker = 'o',color = 'r', label = 'Suboptimal arms reward vectors')
    for i in range(len(O)):
        plt.annotate('Arm '+str(i+1),(O[i][0],O[i][1]),(O[i][0]+0.01,O[i][1]+0.01))
        if i < len(O_star)-1:
            plt.scatter([O[i][0]],[O[i][1]],marker = 'o',color='k')
        if i == len(O_star)-1:
            plt.scatter([O[i][0]],[O[i][1]],marker = 'o',color='k', label = 'Optimal arms reward vectors')
    plt.legend()





# while optimal and l<self.K:
#     mu_l = self.sumRewards[l]/self.ArmsOccurences[l]+np.sqrt(2/self.ArmsOccurences[l]*np.log( self.t*(self.D*self.K)**0.25))
#     mu_i = self.sumRewards[i]/self.ArmsOccurences[i]+np.sqrt(2/self.ArmsOccurences[i]*np.log( self.t*(self.D*self.K)**0.25))
#     if l!=i:
#         if np.max(mu_i-mu_l)  < 0:
#             optimal = False
#     l += 1
