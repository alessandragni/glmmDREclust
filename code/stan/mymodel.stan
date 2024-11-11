
data {
    int<lower=1> Ntot;
    int<lower=1> Ngroups;
    int<lower=1> x_size;
    int<lower=1> z_size;
    int<lower=1> M_max;
    real<lower=0> alpha; // Typically 1 / M_max

    array[Ntot] int<lower=0> y;              // Updated to new array syntax
    array[Ngroups] int<lower=1> group_sizes; // Updated to new array syntax

    matrix[Ntot, x_size] X;
    matrix[Ntot, z_size] Z;
}

parameters {
    array[M_max] vector[z_size] b; // Changed to array syntax for compatibility
    vector[x_size] regressors;
    simplex[M_max] omega;
}

transformed parameters {
    vector[Ntot] fixed_means;
    for (i in 1:Ntot) {
        fixed_means[i] = dot_product(X[i, :], regressors);
    }
}

model {
    omega ~ dirichlet(rep_vector(1, M_max) * alpha);
    b ~ multi_normal_cholesky(rep_vector(0, z_size), diag_matrix(rep_vector(1, z_size)));
    regressors ~ normal(0, 1);

    int j = 1; // Initialize j before looping over groups
    for (i in 1:Ngroups) {
        vector[M_max] log_probas = log(omega);
        for (k in 1:group_sizes[i]) {
            for (m in 1:M_max) {
                real rate = exp(fixed_means[j] + dot_product(Z[j, :], b[m]));
                log_probas[m] += poisson_lpmf(y[j] | rate);
            }
            j += 1;
        }
        target += log_sum_exp(log_probas);
    }
}

generated quantities {
    vector[Ngroups] clus_allocs;
    int j = 1;

    for (i in 1:Ngroups) {
        vector[M_max] log_probas = log(omega);
        for (k in 1:group_sizes[i]) {
            for (m in 1:M_max) {
                real rate = exp(fixed_means[j] + dot_product(Z[j, :], b[m]));
                log_probas[m] += poisson_lpmf(y[j] | rate);
            }
            j += 1;
        }
        clus_allocs[i] = categorical_rng(softmax(log_probas));
    }
}


