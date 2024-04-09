import numpy as np
import pandas as pd


from .Auxiliary_functions import check, are_intersected
from .AlgorithmComp_class import AlgorithmComp


def algorithm_alpha(ran_var, ran_int, n_fix, sim, tol, model, fix, lengths, y, N, t):
    AC = AlgorithmComp(ran_var, ran_int, n_fix, sim, tol, model, fix, lengths, y, N, t)
    # t is the variable for the random slope
    # t = df.groupby('GROUP_VARIABLE')['VARIABLE_NAME'].apply(np.array).values.tolist()

    k = 1
    conv1 = 0
    conv2 = 0
    no_group = 0
    loglik = []
    BIC = []
    AIC = []
    masses = []

    while (conv1 == 0 or conv2 == 0) and (no_group == 0) and (k < AC.K):

        if conv1 == 1 or k >= AC.K1:

            AC.update_W(AC.computeW())

            # estimation of group
            group = np.array([np.nan if np.sum(AC.W[i, :]) == 0 else np.argmax(AC.W[i, :]) for i in range(AC.N)])
            # group is a array of length N
            # setting nan if the sum for the row i of matrix W is zero
            # else setting the nknot (col index) corresponding to the maximum for that row

            no_nan = True
            counter = 0

            for i in range(AC.nknots):
                if AC.weights[i] < AC.tolweight and np.sum(group[np.isfinite(group)] == i) == 0:
                    # if the weights are below a certain threshold and
                    # there is no group for that nknot
                    # (I am checking among the finite groups if the sum of that nknot values is zero)

                    # first set to NA
                    knots = AC.knots
                    if AC.n_ran == 2:
                        knots[i] = np.full(AC.n_ran, np.nan)
                    else:
                        knots[i] = np.nan
                    AC.update_knots(knots)

                    weights = AC.weights
                    weights[i] = np.nan
                    AC.update_weights(weights)

                    counter += 1
                    no_nan = False
                    conv1 = 0

            if no_nan:  # if no nan values have been set
                conv2 = 1

            else:  # if nan values have been set we need to delete them
                # update the value of nknots
                AC.update_nknots(AC.nknots - counter)

                # update the values of knots
                if AC.n_ran == 1:
                    knots = AC.knots[~np.isnan(AC.knots)]
                    AC.update_knots(knots)
                else:
                    knots = AC.knots[~np.isnan(AC.knots).any(axis=1)]
                    AC.update_knots(knots)

                # update the values of weights
                weights = AC.weights[np.isfinite(AC.weights)]  # weights = weights[!is.na(weights)]
                weights = weights / np.sum(weights)
                AC.update_weights(weights)

        if AC.n_ran == 1:
            AC.update_knots(AC.knots.flatten())

        AC.update_W(AC.computeW())

        if AC.nknots != 1:
            colSumsW = np.sum(AC.W, axis=0)

            Wnulli = np.array([j for j in range(len(colSumsW)) if colSumsW[j] == 0])  # <np.finfo(float).eps])
            # print('W nulli: ')
            # print([x for x in Wnulli])

            if len(Wnulli) > 0:
                # print('Warning: there are zero-weight knots')
                Wnulli_list = Wnulli.tolist()
                nknots = AC.nknots - len(Wnulli)
                AC.update_nknots(nknots)

                AC.update_W(np.delete(AC.W, Wnulli_list, axis=1))  # remove columns with index k

                weights_temp = np.delete(AC.weights, Wnulli_list)
                weights = weights_temp / np.sum(weights_temp)
                AC.update_weights(weights)

                knots = np.delete(AC.knots, Wnulli_list, axis=0)
                AC.update_knots(knots)

                if AC.nknots == 1:
                    no_group = 1

        if AC.nknots == 1:
            AC.update_weights(1)
        else:
            colMeansW = np.mean(AC.W, axis=0)
            AC.update_weights(colMeansW / sum(colMeansW))

        ##################

        it = 0  # iteration = 0

        par_old = AC.par
        knots_old = AC.knots

        par_oldss = AC.par
        knots_oldss = AC.knots

        if AC.n_ran == 1:
            AC.update_knots(AC.knots.flatten())

        knots, hess_ran = AC.optim_ran(False)
        AC.update_hess_ran(hess_ran)
        if AC.n_ran == 1:
            AC.update_knots(knots.flatten())
        else:
            AC.update_knots(knots)

        par, hess_fix = AC.optim_fixed()
        AC.update_hess_fix(hess_fix)
        AC.update_par(par)

        while (check(par_old, AC.par, AC.tolF) > 0 or check(knots_old, AC.knots, AC.tolR) > 0) and it < AC.itmax:
            it = it + 1  # new iteration

            par_old = AC.par
            knots_old = AC.knots

            knots, hess_ran = AC.optim_ran(False)
            AC.update_hess_ran(hess_ran)
            if AC.n_ran == 1:
                AC.update_knots(knots.flatten())
            else:
                AC.update_knots(knots)

            par, hess_fix = AC.optim_fixed()
            AC.update_hess_fix(hess_fix)
            AC.update_par(par)

        ##########################################################################################################
        if k > 5:
            # print('new iteration')
            print(AC.knots)
            # knots, hess_ran = AC.optim_ran(True)  # CHANGE HERE TO GET PLOT
            alpha = AC.tol

            D = AC.computeD()
            D_triu = np.triu(D, k=1)
            D_triu[np.tril_indices(D_triu.shape[0], 0)] = np.nan

            not_merged = True

            while not_merged and sum(sum(pd.isna(D_triu))) < D_triu.size:
              print('D_triu')
              print(D_triu)

              (row, col) = np.where(D_triu == np.nanmin(D_triu))
              row = row[0] # in case there is more than 1 minimum
              col = col[0]

              if are_intersected(alpha, AC.hess_ran, AC.n_ran, AC.knots, row, col):
                #print('are intersected')
                # HERE I MAKE THE TWO MASSES CONDENSE
                # two masses are collapsing
                knots_new = AC.knots
                weights_new = AC.weights
                # compute knots_new
                knots_new[row] = (weights_new[row] * knots_new[row] + weights_new[col] * knots_new[col]) / (
                        weights_new[row] + weights_new[col])
                knots_new = np.delete(knots_new, col, axis=0)  # remove row with index col

                if AC.n_ran == 1:
                    knots_new = knots_new.flatten()

                    # compute weights_new
                weights_new[row] = weights_new[row] + weights_new[col]
                weights_new = np.delete(weights_new, col)

                # update nknots, knots, weights
                AC.update_nknots(AC.nknots - 1)
                AC.update_knots(knots_new)
                AC.update_weights(weights_new)

                not_merged = False

              else:
                print('I move on to the check other masses')
                D_triu[row, col] = np.nan


        ############################################################################################

        # estimation of group
        # if AC.nknots == 1:
        #    group = np.ones(AC.N)
        # else:
        #    group = np.array([np.nan if np.sum(AC.W[i, :]) == 0 else np.argmax(AC.W[i, :]) for i in range(AC.N)])

        if check(par_oldss, AC.par, AC.tolF) == 0 and check(knots_oldss, AC.knots, AC.tolR) == 0 and (k>5) and (sum(sum(pd.isna(D_triu))) == D_triu.size):
            conv1 = 1

        masses.append(AC.knots)
        ll = AC.LogLikelihood()
        loglik.append(ll)
        n_params = np.size(AC.knots) + (len(AC.knots) - 1) + len(AC.par)
        BIC.append(n_params * np.log(np.sum(AC.lengths)) - 2*ll)
        AIC.append(n_params * 2 - 2*ll)

        k = k + 1

    return AC.knots, AC.par, AC.W, AC.hess_ran, AC.hess_fix, [masses, loglik, BIC, AIC]
