from abc import ABC, abstractmethod
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor
from scipy.special import factorial

class DataGeneration(ABC):
    """
    """

    def __init__(self, ran_var, ran_int, n_fix, sim, fix=None, lengths=None, y=None, N=None, t=None):
        self.sim = sim
        self.ran_var = ran_var
        self.ran_int = ran_int
        self.n_fix = n_fix
        if self.sim:
            self.subgroups_teor = np.array([2, 5, 3])  # number of subgroups
            self.groups_teor = len(self.subgroups_teor)  # number of groups created
            self.N = np.sum(self.subgroups_teor)
            self.data = None
            self.eta = None
            self.t = None
        else:
            self.N = N
            self.t = t

        if self.ran_var:
            self.z = 2  # we need z for accessing the fixed parameters of glm_mat
        else:
            self.z = 1

        if self.N > 100:
            self.nknots = 100
        else:
            self.nknots = self.N

        self.weights = np.array([1 / self.nknots] * self.nknots)  
        # creating a np.array of 1/nknots with length nknots

        # from the methods
        self.fix = fix
        self.lengths = lengths
        self.y = y
        self.glm_mat = None
        self.par = None
        self.range_min, self.range_max, self.knots = None, None, None

    def generate_curves(self):
        rude_data = []  # list of eta (total value)
        rude_t = []  # list of random effects
        rude_fix = defaultdict(list)  # list of list of fixed effects

        sd = 1
        mu = 0
        # lengths = np.random.randint(70, 100, size=3)
        lengths = np.array([])
        for i in range(self.groups_teor):
            for j in range(self.subgroups_teor[i]):
                length = np.random.randint(70, 100, size=1)[0]
                lengths = np.append(lengths, length)
                x = np.sort(np.random.normal(loc=mu, scale=sd, size=length))
                f = np.array([np.random.normal(loc=mu, scale=sd, size=length) for k in range(self.n_fix)])
                # x = np.sort(np.random.normal(loc=mu, scale=sd, size=lengths[i]))
                # f = np.array([np.random.normal(loc=mu, scale=sd, size=lengths[i]) for k in range(self.n_fix)])

                eta = self.int_values[i]
                if self.ran_var:
                    eta += self.ran_var_values[i] * x
                    rude_t.append(x)

                for k in range(self.n_fix):
                    eta += self.fix_var_values[k] * f[k]
                    rude_fix[k].append(f[k])

                rude_data.append(eta)

        return rude_data, rude_t, rude_fix, lengths.astype(int)  # np.repeat(lengths, self.subgroups_teor) #lengths
  
    @abstractmethod
    def compute_eta_and_y(self):
        pass

    @abstractmethod
    def compute_glm_mat(self):
        pass

    def compute_par(self):
        par = np.median(self.glm_mat[:, self.z:], axis=0)
        if not self.ran_int:
            par_0 = np.median(self.glm_mat[:, 0])
            par = np.insert(par, 0, par_0)  
            # adding par_0 before position 0 of par
        return par

    @staticmethod
    def tukey_fences(col, val):
        if val == 0.25:
            return np.quantile(col, val) - 1.5 * (np.quantile(col, 0.75) - np.quantile(col, 0.25))
        elif val == 0.75:
            return np.quantile(col, val) + 1.5 * (np.quantile(col, 0.75) - np.quantile(col, 0.25))

    def compute_ranges_knots(self):
        n_col = self.glm_mat.shape[1]
        range_min = np.array([self.tukey_fences(self.glm_mat[:, i], 0.25) for i in range(n_col)])
        range_max = np.array([self.tukey_fences(self.glm_mat[:, i], 0.75) for i in range(n_col)])
        interc = (range_max[0] - range_min[0]) * np.random.uniform(low=0, high=1, size=self.nknots) + range_min[0]
        variab = (range_max[1] - range_min[1]) * np.random.uniform(low=0, high=1, size=self.nknots) + range_min[1]

        if self.ran_int and self.ran_var:
            knots = np.column_stack((interc, variab))
        elif self.ran_int and not self.ran_var:
            knots = interc
        elif not self.ran_int and self.ran_var:
            knots = variab
        else:
            print("You should set at least one among ran_int and ran_var equal to True")
            knots = None
        return range_min, range_max, knots

    def set_parameters(self):

        if self.sim:
            if self.data is None and self.t is None and self.fix is None and self.lengths is None \
                    and self.eta is None and self.y is None:
                self.data, self.t, self.fix, self.lengths = self.generate_curves()
                self.eta, self.y = self.compute_eta_and_y()
                while self.check_balance():
                    self.data, self.t, self.fix, self.lengths = self.generate_curves()
                    self.eta, self.y = self.compute_eta_and_y()

        if self.glm_mat is None:
            self.glm_mat = self.compute_glm_mat()

        if self.par is None:
            self.par = self.compute_par()

        if self.range_min is None and self.range_max is None and self.knots is None:
            self.range_min, self.range_max, self.knots = self.compute_ranges_knots()

    def plots(self):
        for i in range(self.N):
            ax = sns.histplot(self.y[i])
            sns.despine(offset=1, left=False, bottom=False)
            ax.axes.vlines(x=np.mean(self.y[i]), color='red', linewidth=0.8, alpha=.8, ymin=-0.6, ymax=70, ls='--')
            plt.show()
        return

    @abstractmethod
    def L_ij(self, d, a, b):
        pass

    @abstractmethod
    def l_ij(self, a, b):
        pass


class DataGenerationBernoulli(DataGeneration):
    def __init__(self, ran_var, ran_int, n_fix, sim, fix=None, lengths=None, y=None, N=None, t=None):
        super().__init__(ran_var, ran_int, n_fix, sim, fix, lengths, y, N, t)
        if self.sim:
            r = np.random.RandomState(1234)
            self.fix_var_values = np.ceil(r.uniform(-10, 10, n_fix))
            self.ran_var_values = np.array([10, 5, 0])
            if self.ran_int:
                self.int_values = np.array([5, 2, -10])
            else:
                r2 = np.random.RandomState(1235)
                self.int_values = np.repeat(np.ceil(r2.uniform(-10, 10, 1)), self.groups_teor)


    @staticmethod
    def pi(x):
        return np.exp(x) / (1 + np.exp(x))
    
    def compute_eta_and_y(self):
        eta = [self.pi(self.data[i]) for i in range(self.N)]
        y = eta

        for i in range(self.N):
            for j in range(self.lengths[i]):
                y[i][j] = 0 if (np.random.uniform(0, 1, 1)[0] > eta[i][j]) else 1

        return eta, y
    
    def check_balance(self):
        table = [np.array(np.unique(x, return_counts=True)).T for x in self.y]

        for i in range(len(table)):
            if table[i].size == 2:
                return True

            perc = table[i][0, 1] / (table[i][0, 1] + table[i][1, 1])
            if (perc > 0.95) or (perc < 0.05):
                return True

        return False

    def compute_glm_mat(self):
        reg = LogisticRegression(random_state=0)
        if self.ran_var:
            glm_mat = np.array([
                np.concatenate((
                    reg.fit(pd.concat([pd.DataFrame(self.t[i]), pd.DataFrame([v[i] for k, v in self.fix.items()]).T],
                                      axis=1), self.y[i]).intercept_,
                    reg.fit(pd.concat([pd.DataFrame(self.t[i]), pd.DataFrame([v[i] for k, v in self.fix.items()]).T],
                                      axis=1), self.y[i]).coef_[0]))
                for i in range(self.N)])
        else:
            glm_mat = np.array([
                np.concatenate((
                    reg.fit(pd.DataFrame([v[i] for k, v in self.fix.items()]).T, self.y[i]).intercept_,
                    reg.fit(pd.DataFrame([v[i] for k, v in self.fix.items()]).T, self.y[i]).coef_[0]))
                for i in range(self.N)])
        return glm_mat

    def L_ij(self, d, a, b):
        if a == 0:
            #return d * (1 / (1 + np.exp(b)))
            return d - np.log(1 + np.exp(b))
        else:
            #return d * (1 / (1 + 1 / np.exp(b)))
            return d + b - np.log(1 + np.exp(b))

    def l_ij(self, a, b):
        return a * b - np.log(1 + np.exp(b))



class DataGenerationPoisson(DataGeneration):
    def __init__(self, ran_var, ran_int, n_fix, sim, fix=None, lengths=None, y=None, N=None, t=None):
        super().__init__(ran_var, ran_int, n_fix, sim, fix, lengths, y, N, t)

        if self.sim:
            r = np.random.RandomState(1234)
            self.fix_var_values = np.round(r.uniform(0, 1.5, n_fix), 1)
            self.ran_var_values = np.array([0.5, 0.2, 0.1])
            if self.ran_int:
                self.int_values = np.array([2.5, 1, -1])  
                # for the moment no implementation of the other parts

    def compute_eta_and_y(self):
        eta = [np.exp(self.data[i]) for i in range(self.N)]
        y = eta

        for i in range(self.N):
            for j in range(self.lengths[i]):
                h = eta[i][j].astype(np.float64)
                y[i][j] = np.random.poisson(lam=h)

        return eta, y
    
    def check_balance(self):
        for i in self.y:
            for j in i:
                if j >= 100:
                    return True
        return False
    
    def compute_glm_mat(self):
        reg = PoissonRegressor()
        glm_mat = []
        for i in range(self.N):
            if self.ran_var:
                x = pd.concat([pd.DataFrame(self.t[i]), pd.DataFrame([v[i] for k, v in self.fix.items()]).T], axis=1)
            else:
                x = pd.DataFrame([v[i] for k, v in self.fix.items()]).T

            a = reg.fit(x, self.y[i])
            glm_mat = np.append(glm_mat, np.append([a.intercept_], a.coef_))

        glm_mat.shape = (self.N, np.int64(glm_mat.size / self.N))
        return glm_mat

    def L_ij(self, d, a, b):
        if a == 0:
            #return d * np.exp(- np.exp(b))
            return d - np.exp(b) 
        else:
            #return d * np.exp(a * b - np.exp(b)) #* 1 / (np.nan_to_num(factorial(a)))
            return d + a * b - np.exp(b) - np.log(np.nan_to_num(factorial(a)))

    def l_ij(self, a, b):
        return a * b - np.exp(b) - np.log(np.nan_to_num(factorial(a)))

        