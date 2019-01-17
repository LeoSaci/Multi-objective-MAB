import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from utils import create_arms,plot_histograms,plot_Pareto_frontier2d,plot_Pareto_frontier3d
from pareto_UCB1 import UCB1Pareto
from linear_scalarization import scal_UCB1
from random_policy import random_policy
from OGDE import OGDE
from mo_MAB import MoMAB

D = 3
K = 15
ArmClass = ['multinomial','exponential'][0]

# Generate a D-objective MO-MAB, K arms of ArmClass distribution
A,A_star = create_arms(ArmClass,K,D)
MO_MAB = MoMAB(A,A_star)

q = stats.norm.ppf(0.975)


T = 10000
hist_times = [500,5000,T]
n_itr = 5

delta0 = 10**-10
delta1 = 10**-5
delta2 = 0.01
delta3 = 0.1
delta4 = 0.5

w0 = np.array([0.6,0.4])
w1 = np.array([0.75,0.25])
w2 = np.array([0.9,0.1])
w3 = np.array([1,0])
w4 = np.array([2,1])

weights_sets = [np.array([0.5,0.5])]
weights_sets2 = [np.array([i/10,1-i/10]) for i in range(10)]


pareto = UCB1Pareto(MO_MAB)
# scal = scal_UCB1(MO_MAB, weights_sets)
# scal2 = scal_UCB1(MO_MAB, weights_sets2)
rand = random_policy(MO_MAB)
ogde = OGDE(MO_MAB, np.array([2,1,0.5]), delta2)

# ogde0 = OGDE(MO_MAB, w4, delta0)
# ogde1 = OGDE(MO_MAB, w4, delta1)
# ogde2 = OGDE(MO_MAB, w4, delta2)
# ogde3 = OGDE(MO_MAB, w4, delta3)
# ogde4 = OGDE(MO_MAB, w4, delta4)

# algo_names = ['Pareto UCB1','delta0','delta1','delta2']
# algo_list = [pareto,ogde,OGDE(MO_MAB, w0, delta1),OGDE(MO_MAB, w0, delta2)]

# algo_names = ['ogde0','ogde1','ogde2','ogde3','ogde4']
# algo_list = [ogde0,ogde1,ogde2,ogde3,ogde4]
algo_names = ['Pareto UCB1','Random policy','OGDE'] #'Linear scalarization '+r'$(W_1)$','Linear scalarization '+r'$(W_2)$'
algo_list = [pareto,rand,ogde]

[plot_arms,plot_curves,plot_histo] = [True,True,True]
print(str(D)+'-objective MO-MAB ; '+str(K)+' arms with '+ArmClass+' distributions')
print('Pareto front : '+str(len(A_star))+' optimal arms')
print('   ')
for ind in range(len(algo_list)):
    algo = algo_names[ind]
    algorithm = algo_list[ind]
    plot_fairness = algo_names[ind] in ['Pareto UCB1','Linear scalarization '+r'$(W_1)$','Linear scalarization '+r'$(W_2)$']
    print('Algorithm : '+algo)
    if plot_fairness:
        fairness = np.zeros((n_itr,T))
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


    avg_regret = sum(regret)/n_itr
    var_regret = sum((regret-avg_regret)**2)/n_itr

    opt_arms_rate_mean = sum(opt_arms_rate)/n_itr
    opt_arms_rate_var = sum((opt_arms_rate - opt_arms_rate_mean)**2)/n_itr


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
    alpha = algo_list[-1].alpha.reshape((1,K))
    mat = np.array([MO_MAB.O[k] for k in range(K)])
    opt_mix = alpha.dot(mat)[0]
    print('Optimal mixed solution for OGDE : '+str(opt_mix))

    if D == 2:
        plot_Pareto_frontier2d(MO_MAB,opt_mix,ogde_list = [ogde0,ogde1,ogde2,ogde3,ogde4])
    elif D == 3:
        plot_Pareto_frontier3d(MO_MAB,opt_mix)


plt.show()
