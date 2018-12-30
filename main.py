import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from utils import create_momab,plot_histograms,plot_Pareto_frontier
from pareto_UCB1 import UCB1Pareto
from linear_scalarization import scal_UCB1
from random_policy import random_policy
from OGDE import OGDE

D = 2
K = 8
ArmClass = ['multinomial','exponential'][0]
# Generate a D-objective MO-MAB, K arms of ArmClass distribution
A,A_star = create_momab(ArmClass,K)

q = stats.norm.ppf(0.975)


T = 5000
hist_times = [500,2000,T]
n_itr = 2

delta = 0.1


w0 = np.array([0.51,0.49])
w1 = np.array([0.505,0.495])
w2 = np.array([0.5,0.5])

weights_sets = [np.array([0.5,0.5])]
weights_sets2 = [np.array([i/10,1-i/10]) for i in range(10)]


pareto = UCB1Pareto(A, A_star)
scal = scal_UCB1(A, weights_sets, A_star)
scal2 = scal_UCB1(A, weights_sets2, A_star)
rand = random_policy(A, A_star)
ogde = OGDE(A, A_star, w0, delta)

# algo_names = ['w0','w1','w2','Pareto']
# algo_list = [ogde,OGDE(A, A_star, w1, delta),OGDE(A, A_star, w2, delta),pareto]
algo_names = ['Pareto UCB1','Linear scalarization '+r'$(W_1)$','Linear scalarization '+r'$(W_2)$','OGDE','Random policy']
algo_list = [pareto,scal,scal2,ogde,rand]

print(str(D)+'-objective MO-MAB ; '+str(K)+' arms with '+ArmClass+' distributions')
print('Pareto front : '+str(len(A_star))+' optimal arms')
print('   ')
for i in range(0,4):
    algo = algo_names[i]
    algorithm = algo_list[i]
    print('Algorithm : '+algo)
    if algo == algo_names[0]:
        plot_Pareto_frontier(algorithm)

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


    avg_fairness = sum(fairness)/n_itr
    var_fairness = sum((fairness-avg_fairness)**2)/n_itr


    avg_regret = sum(regret)/n_itr
    var_regret = sum((regret-avg_regret)**2)/n_itr

    opt_arms_rate_mean = sum(opt_arms_rate)/n_itr
    opt_arms_rate_var = sum((opt_arms_rate - opt_arms_rate_mean)**2)/n_itr


    time = np.arange(T)

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
    plt.plot(np.arange(T),100*opt_arms_rate_mean,label = algo)
    plt.fill_between(time,opt_arms_rate_mean-(q/np.sqrt(n_itr))*np.sqrt(opt_arms_rate_var), opt_arms_rate_mean+(q/np.sqrt(n_itr))*np.sqrt(opt_arms_rate_var),color='#D3D3D3')
    plt.xlabel('Rounds')
    plt.ylabel('$\%$')
    plt.title('Rate of optimal arms pulling, averaged over '+str(n_itr)+' runs')
    plt.legend()

    plot_histograms(algo,histograms,hist_times,K)

plt.show()
