import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import mo_arms
from scipy.optimize import linprog


# Generate a bi-objective MO-MAB, K arms with multinomial distributions
def create_arms(ArmClass,K,D):
    angles1 = [rd.rand()*np.pi/2 for i in range(K)]
    angles2 = [rd.rand()*np.pi/2 for i in range(K)]
    A = []
    for i in range(K):
        samp = np.random.rand(D)
        samp = samp/np.linalg.norm(samp)
        mod = 0.2+0.5*np.random.rand()
        samp = mod*samp
        A.append(mo_arms.ArmMultinomial(mean = samp, random_state=np.random.randint(1, 312414)))

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

def optimal_mixed_sol(arm_means):
    top_arms = []
    bottom_arms = []
    for i in range(len(arm_means)):
        if arm_means[i][0]-arm_means[i][1]<0:
            top_arms.append(i)
        else:
            bottom_arms.append(i)
    if top_arms == [] or bottom_arms == []:
        print('error')
    else:
        l = []
        for i in range(len(top_arms)):
            for j in range(len(bottom_arms)):
                P = arm_means[top_arms[i]]
                Q = arm_means[bottom_arms[j]]
                l.append((P[0]*((Q[1]-P[1])/(Q[0]-P[0]))-P[1])/((Q[1]-P[1])/(Q[0]-P[0])-1))
    max_value = np.max(l)
    return np.array([max_value,max_value])


def plot_Pareto_frontier2d(MO_MAB,ogde_list = []):
    O = MO_MAB.O
    O_star = MO_MAB.O_star
    fig = plt.figure(0)
    plt.scatter([O[i][0] for i in range(len(O))], [O[i][1] for i in range(len(O))] ,marker = 'o',color = 'r', label = 'Suboptimal arms reward vectors')
    for i in range(len(O)):
        plt.annotate('Arm '+str(i+1),(O[i][0],O[i][1]),(O[i][0]+0.01,O[i][1]+0.01))
        if i < len(O_star)-1:
            plt.scatter([O[i][0]],[O[i][1]],marker = 'o',color='k')
        if i == len(O_star)-1:
            plt.scatter([O[i][0]],[O[i][1]],marker = 'o',color='k', label = 'Optimal arms reward vectors')
    opt_mixed_sol = optimal_mixed_sol(O)
    plt.scatter([opt_mixed_sol[0]],[opt_mixed_sol[1]],marker = 'o',color='b', label = '$\sum_{k=1}^K \\alpha_k^* \mu_k$')
    if ogde_list != []:
        for i,algo in enumerate(ogde_list):
            w = algo.w
            alpha = algo.alpha.reshape((len(algo.alpha),1))
            mu = O.T
            point = mu.dot(alpha).T[0]
            plt.scatter([point[0]],[point[1]],marker = 'o',color='g', label = '$\sum_{k=1}^K \\alpha_k^{(T)} \mu_k$')
            #plt.annotate('ogde '+str(i),(point[0],point[1]),(point[0],point[1]))
    plt.legend()

def plot_Pareto_frontier3d(MO_MAB,opt_mix):
    O = MO_MAB.O
    O_star = MO_MAB.O_star

    fig = pylab.figure(0)
    ax = fig.add_subplot(111, projection = '3d')
    x = O[:,0]
    y = O[:,1]
    z = O[:,2]
    sc1 = ax.scatter(x,y,z,color = 'k',label = 'Suboptimal arms reward vectors')

    x = O_star[:,0]
    y = O_star[:,1]
    z = O_star[:,2]
    sc2 = ax.scatter(x,y,z,color = 'r',label = 'Optimal arms reward vectors')

    ax.scatter(opt_mix[0],opt_mix[1],opt_mix[2], marker = 'o', color = 'g', label = 'Computed optimal mixed solution')

    for i in range(len(O)):
        if i < len(O_star):
            col = 'r'
        else:
            col = 'k'
        ax.text(O[i][0],O[i][1],O[i][2], ' Arm'+ '%s' % (str(i+1)), size=10, zorder=1,  color=col)
    plt.legend()



def lin_scal(weights,sample):
    n = len(weights)
    weights = weights.reshape((1,n))
    sample = sample.reshape((n,1))
    return weights.dot(sample)[0,0]

# return the permutation sorting the components of mat.dot(x) in a decreasing order
def decreasing_order(mat,x):
    x = np.array(x).reshape((len(x),1))
    y = mat.dot(x)
    min_val = np.min(y)
    mat_prim = np.zeros(mat.shape)
    l = []
    for i in range(len(mat)):
        k = np.argmax(y)
        y[k] = min_val - 1
        l.append(k)
    return np.array(l)

# reordonate the rows of the matrix x wrt the permutation sigma
def reordonate(sigma,x):
    return np.array([x[i] for i in sigma])

# Projection on the probability simplex
def projsplx(x):
    x = np.array(x)
    x = x.reshape(len(x))
    n = len(x)
    y = np.copy(x)
    y.sort()
    i = n-1
    t_i = (np.sum(y[i:])-1)/(n-i)
    t_hat = t_i + 1
    while t_hat != t_i and i > 0:
        t_i = (np.sum(y[i:])-1)/(n-i)
        if t_i >= y[i-1]:
            t_hat = t_i
        else:
            i -= 1
            t_hat = t_i + 1
    if i == 0:
        t_hat = (np.sum(x)-1)/n
    return np.array([np.max((x[i]-t_hat,0)) for i in range(n)])


def simplex_proj(eps,x):
    x = np.array(x)
    x = x.reshape(len(x))
    n = len(x)
    y = (x-eps*np.ones(n))/(1-eps*n)
    gamma = projsplx(y)
    alpha = eps*np.ones(n) + (1-eps*n)*gamma
    return alpha

# Plot the empirical distributions of arm selection at different times
def plot_histograms(algo,histograms,hist_times,K,A_star):
    fig, (ax1, ax2,ax3) = plt.subplots(1,3)
    t = hist_times[0]
    histogram = histograms[0]
    N,bins,patches1 = ax1.hist(histogram,bins=K,range = (1,K+1),density=True,edgecolor='white',align = 'mid',color = "#777777")
    ax1.set_ylim([-0.06,1])
    for i,(bin_size, bin, patch) in enumerate(zip(N, bins, patches1)):
        if i < len(A_star):
            patch.set_facecolor("#FF0000")
    for i in range(1,K+1):
        ax1.text((2*i+1)/2,-0.05,str(i),size=5,withdash=True)
    ax1.set_xticks([])
    ax1.set_xlabel('Arm number')
    ax1.set_ylabel('Probability')
    ax1.set_title('t = '+str(t))


    t = hist_times[1]
    histogram = histograms[1]
    N,bins,patches2 = ax2.hist(histogram,bins=K,range = (1,K+1),density=True,edgecolor='white',align = 'mid',color = "#777777")
    ax2.set_ylim([-0.06,1])
    for i,(bin_size, bin, patch) in enumerate(zip(N, bins, patches2)):
        if i < len(A_star):
            patch.set_facecolor("#FF0000")
    for i in range(1,K+1):
        ax2.text((2*i+1)/2,-0.05,str(i),size=5)
    ax2.set_xticks([])
    ax2.set_xlabel('Arm number')
    ax2.set_ylabel('Probability')
    ax2.set_title('t = '+str(t))

    t = hist_times[2]
    histogram = histograms[2]
    N,bins,patches3 = ax3.hist(histogram,bins=K,range = (1,K+1),density=True,edgecolor='white',align = 'mid',color = "#777777")
    ax3.set_ylim([-0.06,1])
    for i,(bin_size, bin, patch) in enumerate(zip(N, bins, patches3)):
        if i < len(A_star):
            patch.set_facecolor("#FF0000")
    for i in range(1,K+1):
        ax3.text((2*i+1)/2,-0.05,str(i),size=5)
    ax3.set_xticks([])
    ax3.set_xlabel('Arm number')
    ax3.set_ylabel('Probability')
    ax3.set_title('t = '+str(t))

    fig.suptitle(algo)
    return fig

def alpha_star(MO_MAB,w):
    O = MO_MAB.O
    D = MO_MAB.D
    K = MO_MAB.K

    w_temp = list(w)[1:]
    w_temp.append(0)
    w_temp = np.array(w_temp)
    w_prime = w-w_temp

    c = np.zeros(K+D+D**2)
    c[K:K+D] = (np.arange(D)+1)*w_prime
    for i in range(D):
        c[K+(i+1)*D:K+(i+2)*D]

    A_eq = np.zeros(K+D+D**2)
    A_eq[:K] = np.ones(K)
    A_eq = A_eq.reshape((1,len(A_eq)))
    b_eq = np.array([1])
    b_u = np.zeros(K+D**2+D**2)

    A_u = np.zeros((K+D+D**2,K+2*D**2)).T
    A_u[:K,:K] = -np.eye(K)
    A_u[K:K+D**2,K+D:] = -np.eye(D**2)

    A_u[K+D**2:,K+D:] = -np.eye(D**2)

    B_2 = np.zeros((D**2,K))
    for j in range(D):
        for i in range(D):
            B_2[j*D+i] = np.array([O[k][j] for k in range(K)])
    A_u[K+D**2:,:K] = B_2

    B_3 = np.zeros((D**2,D))
    for j in range(D):
        B_3[j*D:(j+1)*D] = -np.eye(D)

    A_u[K+D**2:,K:K+D] = B_3
    res = linprog(c=c,A_ub=A_u,b_ub=b_u,A_eq=A_eq,b_eq=b_eq)

    return res['x'][:K]
