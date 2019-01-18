import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from utils import create_arms,plot_histograms,plot_Pareto_frontier2d,plot_Pareto_frontier3d,alpha_star
from pareto_UCB1 import UCB1Pareto
from linear_scalarization import scal_UCB1
from random_policy import random_policy
from OGDE import OGDE
from mo_MAB import MoMAB
import mo_arms

q = stats.norm.ppf(0.975)

D = 2   # Number of objectives
K = 6   # Number of Arms
ArmClass = ['multinomial','exponential'][0]  # Arms distribution

# Generate a random D-objective MO-MAB, K arms of ArmClass distribution
                      #(with mean vectors of the form r*[x_1,...,x_D], with sum(x_i) = 1 and r in [0.2,0.7] )
A,A_star = create_arms(ArmClass,K,D)
MO_MAB = MoMAB(A,A_star)


# General parameters
T = 5000                   # horizon time
hist_times = [50,500,T]    # Times at which we plot histograms of selected arms
n_itr = 1                  # number of runs for each method


# Parameters for OGDE algorithm
delta = 0.01
w = np.array([2**(1-i) for i in range(D)])

# Set of weights for Linear Scalarization algorithm
weights_sets = [np.array([(i+1)/10,1-(i+1)/10]) for i in range(9)]

# Algorithms
scal = scal_UCB1(MO_MAB, weights_sets)
pareto = UCB1Pareto(MO_MAB)
ogde = OGDE(MO_MAB, w, delta)
rand = random_policy(MO_MAB)

# Choices of the algorithms to run and of the curves to plot
algo_names = ['OGDE','Pareto UCB1','Random policy','Linear scalarization']
algo_list = [ogde,pareto,rand,scal]
[plot_arms,plot_curves,plot_histo] = [True,True,True]

REGRET = []
VAR_REGRET = []

FAIRNESS = []
VAR_FAIRNESS = []

RATE_OPT = []
VAR_RATE_OPT = []

two_regrets = False
if two_regrets:
    REGRET_OGDE = []
    VAR_REGRET_OGDE = []

    REGRET_PARETO = []
    VAR_REGRET_PARETO = []


print(str(D)+'-objective MO-MAB ; '+str(K)+' arms with '+ArmClass+' distributions')
print('Pareto front : '+str(len(A_star))+' optimal arms')
print('   ')
for ind in range(len(algo_list)):
    algo = algo_names[ind]
    algorithm = algo_list[ind]
    plot_fairness = algo_names[ind] in ['Pareto UCB1','Linear scalarization '+r'$(W_1)$','Linear scalarization '+r'$(W_2)$','Linear scalarization '+r'$(W_3)$']
    print('Algorithm : '+algo)
    if plot_fairness:
        fairness = np.zeros((n_itr,T))
    if two_regrets:
        regret_pareto = np.zeros((n_itr,T))
        regret_ogde = np.zeros((n_itr,T))
    else:
        regret = np.zeros((n_itr,T))

    histograms = [[],[],[]]
    opt_arms_rate = np.zeros((n_itr,T))
    for it in tqdm(range(n_itr)):
        opt_arms_rate_it = np.zeros(T)
        opt_arms_rate_it[0:K] = np.arange(K)+1
        temp = K
        algorithm.initialize()
        for n in range(T-K):
            i = algorithm.update() # indice of the arm selected at time t=n+K
            if plot_fairness:
                fairness[it,n+K] = algorithm.fairness()
            if two_regrets:
                if algo == 'OGDE':
                    regret_ogde[it,K+n] = algorithm.regret()
                    regret_pareto[it,K+n] = algorithm.regret_pareto()
                elif algo == 'Pareto UCB1':
                    regret_ogde[it,K+n] = algorithm.regret_ogde(ogde)
                    regret_pareto[it,K+n] = algorithm.regret()

            else:
                regret[it,K+n] = algorithm.regret()
            if n < hist_times[0]-K:
                histograms[0].append(i+1)
                histograms[1].append(i+1)
                histograms[2].append(i+1)
            else:
                if n < hist_times[1]-K:
                    histograms[1].append(i+1)
                    histograms[2].append(i+1)
                else:
                    histograms[2].append(i+1)

            opt_arms_rate_it[n+K] = temp+int(i < len(A_star))
            temp = opt_arms_rate_it[n+K]
        opt_arms_rate_it /= (1+np.arange(T))
        opt_arms_rate[it] = opt_arms_rate_it


    if plot_fairness:
        avg_fairness = sum(fairness)/n_itr
        var_fairness = sum((fairness-avg_fairness)**2)/n_itr
        FAIRNESS.append(avg_fairness)
        VAR_FAIRNESS.append(var_fairness)

    if not two_regrets:
        avg_regret = sum(regret)/n_itr
        var_regret = sum((regret-avg_regret)**2)/n_itr
        REGRET.append(avg_regret)
        VAR_REGRET.append(var_regret)
    else:
        avg_regret_pareto = sum(regret_pareto)/n_itr
        var_regret_pareto = sum((regret_pareto-avg_regret_pareto)**2)/n_itr
        REGRET_PARETO.append(avg_regret_pareto)
        VAR_REGRET_PARETO.append(var_regret_pareto)

        avg_regret_ogde = sum(regret_ogde)/n_itr
        var_regret_ogde = sum((regret_ogde-avg_regret_ogde)**2)/n_itr
        REGRET_OGDE.append(avg_regret_ogde)
        VAR_REGRET_OGDE.append(var_regret_ogde)

    opt_arms_rate_mean = sum(opt_arms_rate)/n_itr
    opt_arms_rate_var = sum((opt_arms_rate - opt_arms_rate_mean)**2)/n_itr


    RATE_OPT.append(opt_arms_rate_mean)
    VAR_RATE_OPT.append(opt_arms_rate_var)

    time = np.arange(T)

    if plot_curves:
        if plot_fairness:
            plt.figure(1)
            plt.plot(time,avg_fairness,label=algo)
            plt.fill_between(time,avg_fairness-(q/np.sqrt(n_itr))*np.sqrt(var_fairness), avg_fairness+(q/np.sqrt(n_itr))*np.sqrt(var_fairness),color='#D3D3D3')
            plt.xlabel('Number of rounds')
            plt.ylabel('Unfairness')
            plt.title('Unfairness averaged over '+str(n_itr)+' runs')
            plt.legend()

        if two_regrets:
            plt.figure(21)
            plt.plot(time,avg_regret_pareto,label=algo)
            plt.fill_between(time,avg_regret_pareto-(q/np.sqrt(n_itr))*np.sqrt(var_regret_pareto), avg_regret_pareto+(q/np.sqrt(n_itr))*np.sqrt(var_regret_pareto),color='#D3D3D3')
            plt.xlabel('Number of rounds')
            plt.ylabel('Regret')
            plt.title('Regret for Pareto definition averaged over '+str(n_itr)+' runs')
            plt.legend()

            plt.figure(22)
            plt.plot(time,avg_regret_ogde,label=algo)
            plt.fill_between(time,avg_regret_ogde-(q/np.sqrt(n_itr))*np.sqrt(var_regret_ogde), avg_regret_ogde+(q/np.sqrt(n_itr))*np.sqrt(var_regret_ogde),color='#D3D3D3')
            plt.xlabel('Number of rounds')
            plt.ylabel('Regret')
            plt.title('Regret for OGDE definition averaged over '+str(n_itr)+' runs')
            plt.legend()
        else:
            plt.figure(2)
            plt.plot(time,avg_regret,label=algo)
            plt.fill_between(time,avg_regret-(q/np.sqrt(n_itr))*np.sqrt(var_regret), avg_regret+(q/np.sqrt(n_itr))*np.sqrt(var_regret),color='#D3D3D3')
            plt.xlabel('Number of rounds')
            plt.ylabel('Regret')
            plt.title('Regret averaged over '+str(n_itr)+' runs')
            plt.legend()

        plt.figure(3)
        plt.plot(np.arange(T),opt_arms_rate_mean,label = algo)
        plt.fill_between(time,opt_arms_rate_mean-(q/np.sqrt(n_itr))*np.sqrt(opt_arms_rate_var), opt_arms_rate_mean+(q/np.sqrt(n_itr))*np.sqrt(opt_arms_rate_var),color='#D3D3D3')
        plt.xlabel('Rounds')
        plt.ylabel('$\%$')
        plt.title('Rate of optimal arms pulling, averaged over '+str(n_itr)+' runs')
        plt.legend()

    if plot_histo:
        fig = plot_histograms(algo,histograms,hist_times,K,A_star)

if plot_arms:
    alpha = ogde.alpha.reshape((1,K))
    mat = np.array([MO_MAB.O[k] for k in range(K)])
    opt_mix = alpha.dot(mat)[0]
    print('Optimal mixed solution for OGDE : '+str(opt_mix))

    if D == 2:
        plot_Pareto_frontier2d(MO_MAB)
    elif D == 3:
        plot_Pareto_frontier3d(MO_MAB,opt_mix)
plt.show()
