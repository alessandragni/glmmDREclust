from scipy.special import softmax
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import numdifftools as nd
import matplotlib.pyplot as plt
import scipy.stats as st
import sys
from math import isnan

np.set_printoptions(precision=5)

from .DataGeneration_class import DataGenerationBernoulli
from .DataGeneration_class import DataGenerationPoisson

class AlgorithmComp:

    def __init__(self, ran_var, ran_int, n_fix, sim, tol, model, fix, lengths, y, N, t):
        # model can assume values 'P' (Poisson) or 'B' (Bernoulli)
        self.ran_var = ran_var
        self.ran_int = ran_int
        self.n_fix = n_fix
        self.sim = sim
        if not self.sim:
            self.N = N
            self.fix = fix
            self.lengths = lengths
            self.y = y
            self.t = t

        self.n_ran = self.ran_int + self.ran_var  # number of total random effects we have in total
        # n_ran can assume values 1 (just intercept or variable) or 2

        # parameters
        self.K = 60
        self.K1 = 20
        self.tolweight = 0.005
        self.tol = tol  # distance between 2 points for making them collapse
        self.tolR = 10**(-8)  # tolerance for the iterative estimation of support points
        self.tolF = 10**(-8)  # tolerance for the iterative estimation of fixed effects
        self.itmax = 20

        if model == 'B':
            self.DG = DataGenerationBernoulli(ran_var, ran_int, n_fix, sim, fix, lengths, y, N, t)
        elif model == 'P':
            self.DG = DataGenerationPoisson(ran_var, ran_int, n_fix, sim, fix, lengths, y, N, t)

        self.DG.set_parameters()
        self.get_parameters()

        self.W = None
        self.hess_ran = None
        self.hess_fix = None

    def get_parameters(self):
        self.nknots = self.DG.nknots
        self.knots = self.DG.knots
        self.par = self.DG.par
        self.weights = self.DG.weights
        self.glm_mat = self.DG.glm_mat
        self.range_min = self.DG.range_min
        self.range_max = self.DG.range_max

        if self.sim:
            self.lengths = self.DG.lengths
            self.N = self.DG.N
            self.y = self.DG.y
            self.t = self.DG.t
            self.fix = self.DG.fix
            self.fix_var_values = self.DG.fix_var_values
            self.ran_var_values = self.DG.ran_var_values
            self.int_values = self.DG.int_values
            self.subgroups_teor = self.DG.subgroups_teor

        return

    def computeD(self):
        if self.n_ran == 1:
            return distance_matrix(self.knots.reshape(-1, 1), self.knots.reshape(-1, 1))
        else:
            return distance_matrix(self.knots, self.knots)

    def prob(self, index, knots_row):
        # p(c(y_i)|beta,c_l) exploiting properties of exponential of sum
        d = 1

        for j in range(self.lengths[index]):
            a = self.y[index][j]

            if self.n_ran == 1:
                if self.ran_int:
                    b = knots_row
                else:
                    b = self.par[0]

                if self.ran_var:
                    b = b + knots_row * self.t[index][j]
            else:  # case n_ran==2
                b = knots_row[0] + knots_row[1] * self.t[index][j]

            if self.ran_int:
                for k in range(self.n_fix):
                    b = b + self.par[k] * self.fix[k][index][j]
            else:
                for k in range(self.n_fix):
                    b = b + self.par[1 + k] * self.fix[k][index][j]

            #print('big d:' + str(d))
            d = self.DG.L_ij(d, a, b)

        return d

    def computeW(self):
        l = np.empty((self.N, self.nknots))
        W_temp = np.empty((self.N, self.nknots))
        for i in range(self.N):
            #print('PAESE: ' + str(i))
            for j in range(self.nknots):
                l[i, j] = self.prob(i, self.knots[j]) 
                W_temp[i, j] = np.log(self.weights[j]) + l[i, j] # because i make use of softmax function
            #print(W_temp[i,:])

        W = np.empty( (self.N, self.nknots) )
        for i in range(self.N):
            W[i,:] = softmax(W_temp[i,:])

        self.W = W
        return W

    def loglikelihood(self, c, par):
        s = np.empty(self.N)  # s <- rep(0,N)
        # param_fixed = par
        # param_random = knots

        for i in range(self.N):
            a = self.y[i]

            if self.n_ran == 1:
                if self.ran_int:
                    b = c
                else:
                    b = par[0]

                if self.ran_var:
                    b = b + c * self.t[i]
            else:
                b = c[0] + c[1] * self.t[i]

            if self.ran_int:
                for k in range(self.n_fix):
                    b = b + par[k] * self.fix[k][i]
            else:
                for k in range(self.n_fix):
                    b = b + par[1 + k] * self.fix[k][i]

            s[i] = np.sum(self.DG.l_ij(a, b))

        return s

    def exp_c(self, c, *args):
        index = args[0]
        # compute expectation random effects
        logl = self.loglikelihood(c, self.par)  # c is a list of [pa_0, pa_1]
        d = np.sum(self.W[:, index] * logl)
        return (-1) * d

    def exp_beta(self, beta, *args):
        c = args[0]
        d = np.empty((self.N, self.nknots))
        for j in range(self.nknots):
            d[:, j] = self.W[:, j] * self.loglikelihood(c[j], beta)
        return (-1) * np.sum(d)

    def optim_ran(self, plot_):
        if self.n_ran == 1:
            d = np.empty(self.nknots)
            hess = np.empty((self.nknots, 1))
        else:
            d = np.empty((self.nknots, 2))
            hess = np.empty((self.nknots, 2, 2))

        for k in range(self.nknots):
            res = minimize(self.exp_c, x0=self.knots[k], args=tuple([k]), method='Nelder-Mead')
            d[k] = res.x
            # print(k)
            f = lambda x: self.exp_c(x, k)
            if self.n_ran == 1:
                Hfun = nd.Hessdiag(f)
            elif self.n_ran == 2:
                Hfun = nd.Hessian(f)
            hess[k] = Hfun(res.x)

            # PLOT
            if plot_:
                if self.n_ran == 1:
                    ts = np.linspace(res.x[0] - 15, res.x[0] + 15, 800)
                    fs = np.array([self.exp_c(t, k) for t in ts])
                    #  tm = (ts[:-1] + ts[1:])/2 #
                    #  fm = np.diff(fs) / np.diff(ts) #
                    plt.plot(ts, fs, label=f'$k={k}$')
                    #  plt.plot(tm, fm, 'k--') #
                    plt.plot(self.knots[k], self.exp_c(self.knots[k], k), 'ro')
                    plt.plot(res.x, self.exp_c(res.x, k), 'go')
                    plt.margins(x=0, tight=True)
                    plt.legend(ncol=2)
                    if self.ran_int:
                        xposition = self.int_values
                    else:
                        xposition = self.ran_var_values
                    for xc in xposition:
                        plt.axvline(x=xc, color='k', linestyle='--')
                    print(d[k])
                    plt.axvline(x=d[k] - st.norm.ppf(1 - 0.05 / 2) * np.sqrt(1 / hess[k]), color='r', linestyle='--')
                    plt.axvline(x=d[k] + st.norm.ppf(1 - 0.05 / 2) * np.sqrt(1 / hess[k]), color='r', linestyle='--')
                    plt.show()

        return d, hess

    def optim_fixed(self):
        res = minimize(self.exp_beta, x0=self.par, args=tuple([self.knots]), method='Nelder-Mead')
        b = res.x
        f = lambda x: self.exp_beta(x, self.knots)
        Hfun = nd.Hessdiag(f)
        hess = Hfun(b)
        return b, hess


    def LogLikelihood(self):
        llik = 0
        LL = 0
        for i in range(self.N):
            for m in range(self.nknots):
                LL = LL + self.weights[m] * np.exp(self.prob(i, self.knots[m])) 
            print(LL)
            llik = llik + (np.log(LL)) #if LL != 0 else np.log(sys.float_info.min)
        return(llik)

        #group = np.array([np.nan if np.sum(self.W[i,:])==0 else np.argmax(self.W[i,:]) for i in range(self.N)])
        #s = []  # s <- rep(0,N)
        # param_fixed = par
        # param_random = knots = c

        #unique_values = np.unique(group[~np.isnan(group)])
        #print(unique_values)
        #for m in unique_values:
        #    temp = 1
        #    positions = np.where(group == m)
        #    print(positions)
        #    for i in positions[0]: 
        #        temp = temp * self.prob(i, self.knots[m])

        #    s.append(self.weights[m] * temp)

        #return np.sum(np.array(s))

    #####################################

    def update_knots(self, knots):
        self.knots = knots
        return

    def update_par(self, par):
        self.par = par
        return

    def update_weights(self, weights):
        self.weights = weights
        return

    def update_W(self, W):
        self.W = W
        return

    def update_nknots(self, nknots):
        self.nknots = nknots
        return

    def update_hess_ran(self, hess_ran):
        self.hess_ran = hess_ran
        return

    def update_hess_fix(self, hess_fix):
        self.hess_fix = hess_fix
        return
    
