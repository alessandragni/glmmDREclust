import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
import scipy.stats as st
from numpy import linalg as la
from scipy.stats import chi2

def check(k_old, k, tol):
    if np.size(k_old) == np.size(k):
        if k.ndim==1:
            return sum(np.absolute(k_old - k) > tol)
        else:
            return sum(np.absolute(k_old - k) > tol).all()
    else:
        return 1


def plot_ellipse(mu, sigma, n_std_tau, ax=None, **kwargs):
    # Usage: https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/plot_ellipse.ipynb
    if ax is None:
        ax = plt.gca()

    ee, V = np.linalg.eigh(sigma)
    e_big = ee[1]
    e_small = ee[0]
    v_big = V[:, 1]
    theta = np.arctan(v_big[1] / v_big[0]) * 180. / np.pi

    long_length = n_std_tau * 2. * np.sqrt(e_big)
    short_length = n_std_tau * 2. * np.sqrt(e_small)

    if not ('facecolor' in kwargs):
        kwargs['facecolor'] = 'none'

    ellipse = Ellipse(mu, width=long_length, height=short_length, angle=theta, **kwargs)
    ax.add_artist(ellipse)


def ellipsoid_intersection_test_helper(sigma_A, sigma_B, mu_A, mu_B):
    lambdas, Phi = eigh(sigma_A, b=sigma_B)
    v_squared = np.dot(Phi.T, mu_A - mu_B) ** 2
    return lambdas, Phi, v_squared


def ellipsoid_K_function(ss, lambdas, v_squared, tau):
    ss = np.array(ss).reshape((-1, 1))
    lambdas = np.array(lambdas).reshape((1, -1))
    v_squared = np.array(v_squared).reshape((1, -1))
    return 1. - (1. / tau ** 2) * np.sum(v_squared * ((ss * (1. - ss)) / (1. + ss * (lambdas - 1.))), axis=1)


def ellipsoid_intersection_test(sigma_A, sigma_B, mu_A, mu_B, tau):
    lambdas, Phi, v_squared = ellipsoid_intersection_test_helper(sigma_A, sigma_B, mu_A, mu_B)
    res = minimize_scalar(ellipsoid_K_function,
                          # bracket=[0.0, 0.5, 1.0],
                          args=(lambdas, v_squared, tau))
    ss = np.linspace(-0.2, 1.2, 1000)
    KK = ellipsoid_K_function(ss, lambdas, v_squared, tau)
    Kmin = np.min(KK)
    #plt.plot(ss, KK)
    #plt.plot(ss, ss * 0.0)
    #plt.title('(Kmin > 0) = ' + str(Kmin > 0))
    #plt.ylabel('K(s)')
    #plt.xlabel('s')
    #plt.ylim([-0.5, 1.5])
    #plt.show()

    return res.fun[0] >= 0


def are_intersected(alpha, hess_ran, n_ran, knots, row, col):
    if n_ran == 1:
        CI_length = np.array(st.norm.ppf(1 - alpha / 2) * np.sqrt(1 / hess_ran))
        a1 = knots[row] - CI_length[row][0]
        b1 = knots[col] - CI_length[col][0]
        a2 = knots[row] + CI_length[row][0]
        b2 = knots[col] + CI_length[col][0]
        c1 = max(a1, b1)
        c2 = min(a2, b2)
        if c1 >= c2:
            return False
        else:
            return True
    elif n_ran == 2:
        mu_A = knots[row].flatten()
        mu_B = knots[col].flatten()
        sigma_A_inv = hess_ran[row].reshape(2, 2)
        sigma_B_inv = hess_ran[col].reshape(2, 2)
        sigma_A = np.linalg.inv(sigma_A_inv)
        sigma_B = np.linalg.inv(sigma_B_inv)

        lamA = la.eigvalsh(sigma_A)
        lamB = la.eigvalsh(sigma_B)

        r = np.sqrt(chi2.ppf(alpha, 2))

        p = np.sum((mu_A - mu_B) ** 2)
        d = np.sqrt(p)

        if max(lamA) < min(lamB):
            lam2 = min(lamB)
            lam1 = max(lamA)
            value = (d < r * (np.sqrt(lam2) - np.sqrt(lam1)))
            # if value:
            #    print('concentric')
        elif max(lamB) < min(lamA):
            lam2 = min(lamA)
            lam1 = max(lamB)
            value = (d < r * (np.sqrt(lam2) - np.sqrt(lam1)))
            # if value:
            #    print('concentric')
        else:
            value = ellipsoid_intersection_test(sigma_A, sigma_B, mu_A, mu_B, r)
            # if value:
            #    print('tested')

        # plt.figure(figsize=(10,10))
        # plot_ellipse(mu_A, sigma_A, n_std_tau=r, facecolor='r', edgecolor='k', alpha=0.5, linewidth=1)
        # plot_ellipse(mu_B, sigma_B, n_std_tau=r, facecolor='b', edgecolor='k', alpha=0.5, linewidth=1)
        # plt.xlim([-50, 50])
        # plt.ylim([-50, 50])
        # plt.gca().set_aspect(1.0)
        # plt.show()

        return value
