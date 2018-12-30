import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import mo_arms




# Generate a bi-objective MO-MAB, K arms with multinomial distributions
def create_momab(ArmClass,K):
    angles = [rd.rand()*np.pi/2 for i in range(K)]
    if ArmClass == 'multinomial':
        A = [mo_arms.ArmMultinomial(mean = rd.rand()*np.array([np.cos(angles[i]),np.sin(angles[i])]), random_state=np.random.randint(1, 312414)) for i in range(K)]
    else:
        A = [mo_arms.ArmExp(L = rd.rand()*np.array([np.cos(angles[i]),np.sin(angles[i])]), random_state=np.random.randint(1, 312414)) for i in range(K)]
    A_star = []
    for k in range(K):
        mu = A[k].mean
        is_optimal = True
        i = 0
        while i < K and is_optimal:
            if np.max(mu-A[i].mean)<=0 and i != k:
                is_optimal = False
            i += 1
        if is_optimal:
            A_star.append(k)

    for i in range(len(A_star)):
        temp = A[i]
        A[i] = A[A_star[i]]
        A[A_star[i]] = temp
    A_star = np.arange(len(A_star))
    return A,A_star


def Pareto_metric(mu,ParetoSet):
    return np.max([np.min(nu-mu) for nu in ParetoSet])


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



def lin_scal(weights,sample):
    n = len(weights)
    weights = weights.reshape((1,n))
    sample = sample.reshape((n,1))
    return weights.dot(sample)[0,0]

def reordonate(mu):
    mu_prime = np.copy(mu.reshape(len(mu)))
    mu_prime.sort()
    mu_prime.reverse()
    return mu_prime

def simplex_proj(eps,x):
    y = x.reshape(len(x))
    n = len(x)
    y = 1/(1-eps*n)*(y-eps*np.ones(n))
    y_sorted = list(np.copy(y))
    y_sorted.sort()
    y_sorted.reverse()
    y_sorted = np.array(y_sorted)
    rho = np.max([j for j in range(1,n+1) if y_sorted[j-1]+(1/j)*(1-np.sum(y_sorted[:j]))])
    lambd = (1/rho)*(1-sum(y_sorted[:rho]))
    gamma = np.array([np.max([x[i]+lambd,0]) for i in range(n)])
    alpha = eps*np.ones(n) + (1-eps*n)*gamma
    return alpha

def plot_histograms(algo,histograms,hist_times,K):
    plt.figure(algo) # Empirical distributions of the selected arms at different times
    plt.subplot(1,3,1)
    t = hist_times[0]
    histogram = histograms[0]
    plt.hist(histogram,4*K, range = (0,K), weights = [1/len(histogram) for i in range(len(histogram))])
    plt.xlim([1,K])
    plt.ylim([0,1])
    plt.ylabel('Probability')
    plt.title('t = '+str(t))

    plt.subplot(1,3,2)
    t = hist_times[1]
    histogram = histograms[1]
    plt.hist(histogram,4*K, range = (0,K), weights = [1/len(histogram) for i in range(len(histogram))])
    plt.xlim([1,K])
    plt.ylim([0,1])
    plt.xlabel('Arms')
    plt.title('t = '+str(t))

    plt.subplot(1,3,3)
    t = hist_times[2]
    histogram = histograms[2]
    plt.hist(histogram,4*K, range = (0,K), weights = [1/len(histogram) for i in range(len(histogram))])
    plt.xlim([1,K])
    plt.ylim([0,1])
    plt.title('t = '+str(t))
