import numpy as np
import csv
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.kernel_ridge import KernelRidge
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import pickle 
import time 
import mkl
from sklearn.neural_network import MLPRegressor
from numpy.linalg import norm
mkl.set_num_threads(1)


def generate_data_interactive(B_true, n, p, polykernel_degree = 3, 
                              noise_half_width = 0.25, constant = 3):
    (d, _) = B_true.shape
    X = np.random.normal(size=(p, n))
    poly = PolynomialFeatures(degree = polykernel_degree, interaction_only = True, include_bias = False)
    X_new = np.transpose(poly.fit_transform(np.transpose(X)))
    
    c_expected = np.matmul(B_true, X_new) + constant 
    epsilon = 1 - noise_half_width + 2 * noise_half_width * np.random.rand(d, n)
    c_observed = c_expected * epsilon

    X_false = np.concatenate((np.reshape(np.ones(n), (1, -1)), X), axis = 0)
    X_true = np.concatenate((np.reshape(np.ones(n), (1, -1)), X_new), axis = 0)
    
    return (X_false, X_true, c_observed, c_expected)

def read_A_and_b(file_path = None):
    if file_path is None:
        file_path = 'A_and_b.csv'

    A_and_b = pd.read_csv(file_path, header = None)
    A_and_b = A_and_b.to_numpy()
    A_mat = A_and_b[:, :-1]
    b_vec = A_and_b[:, -1]
    
    return (A_mat, b_vec)

def Kernel_transformation(X_train, X_val, gamma = 1):
    
    K_mat = rbf_kernel(np.transpose(X_train), gamma = gamma)
    K_mat = (K_mat + np.transpose(K_mat))/2
    eig_val = np.real(np.linalg.eig(K_mat)[0])
    eig_vector = np.real(np.linalg.eig(K_mat)[1])

    eig_val_positive = np.copy(eig_val)
    eig_val_positive[eig_val_positive <= 1e-4] = 0
    eig_val_positive_inv = np.copy(eig_val_positive)
    eig_val_positive_inv[eig_val_positive_inv > 0] = 1/eig_val_positive_inv[eig_val_positive_inv > 0]

    K_mat_sqrt = np.matmul(np.matmul(eig_vector, np.diag(np.sqrt(eig_val_positive))), np.transpose(eig_vector))
    K_mat_sqrt_inv = np.matmul(np.matmul(eig_vector, np.diag(np.sqrt(eig_val_positive_inv))), np.transpose(eig_vector))
    
    K_mat_val = rbf_kernel(np.transpose(X_train), np.transpose(X_val), gamma = gamma)
    K_design_val = np.matmul(K_mat_sqrt_inv, K_mat_val)
    
    return (K_mat_sqrt, K_mat_sqrt_inv, K_design_val)


def generate_sp_oracle(A_mat, b_vec, verbose = False):

    (m, d) = A_mat.shape

    model = gp.Model()
    model.setParam('OutputFlag', verbose)
    model.setParam("Threads", 1)

    w = model.addVars(d, lb = 0, ub = 1, name = 'w')
    model.update()
    for i in range(m):
        model.addConstr(gp.quicksum(A_mat[i, j] * w[j] for j in range(len(w))) == b_vec[i])
    model.update()

    def local_oracle(c):
        if len(c) != len(w):
            raise Exception("Sorry, c and w dimension mismatched")

        obj = gp.quicksum(c[i] * w[i] for i in range(len(w)))
        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        model.optimize()

        w_ast = [w[i].X for i in range(len(w))]
        z_ast = model.objVal

        return (z_ast, w_ast)
    
    return local_oracle

def oracle_dataset(c, oracle):
    (d, n) = c.shape
    z_star_data = np.zeros(n)
    w_star_data = np.zeros((d, n))
    for i in range(n):
        (z_i, w_i) = oracle(c[:,i])
        z_star_data[i] = z_i
        w_star_data[:,i] = w_i
    return (z_star_data, w_star_data)

def spo_linear_predict(B_est, X_val):
    c_hat = np.matmul(B_est, X_val)
    
    return c_hat 

def spo_kernel_predict(v_est, K_design_val):
    c_hat = np.matmul(v_est, K_design_val)
    return c_hat 

def ridge_kernel(X, c, gamma = 1, cur_lambda = 1):
    kr = KernelRidge(alpha = cur_lambda/2, kernel = "rbf", gamma = gamma)
    kr.fit(np.transpose(X), np.transpose(c))
    return kr 

def ridge_linear(X, c, cur_lambda = 1):

    (p, n) = X.shape
    (d, n2) = c.shape
    if n != n2:
        raise Exception("Sorry, c and X dimension mismatched")

    model = Ridge(alpha = cur_lambda/2, fit_intercept = False)
    model.fit(np.transpose(X), np.transpose(c))
        
    return model

def ls_linear(X, c):
    (p, n) = X.shape
    (d, n2) = c.shape
    if n != n2:
        raise Exception("Sorry, c and X dimension mismatched")

    model = LinearRegression(fit_intercept = False)
    model.fit(np.transpose(X), np.transpose(c))

    return model


def SPO_reformulation_kernel(K_mat_sqrt, K_mat_sqrt_inv, X, c, 
                             z_star_train, w_star_train, 
                             A_mat, b_vec, 
                             cur_lambda = 1, verbose = False):

    try:
        v_est = SPO_reformulation_linear(K_mat_sqrt, c, z_star_train, w_star_train, A_mat, b_vec, 
                                   cur_lambda = cur_lambda, verbose = verbose)
    except Exception as e:
        print("SPO_kernel error:", e)
        print("Optimization is unsuccessful with status code", model.status)
    return v_est

def SPO_reformulation_linear(X, c, z_star_train, w_star_train, A_mat, b_vec, 
                                   cur_lambda = 1, verbose = False):

    A_mat_trans = np.transpose(A_mat)
    (n_nodes, n_edges) = A_mat.shape
    (p, n) = X.shape
    (d, n2) = c.shape

    if n != n2:
        raise Exception("Sorry, c and X dimension mismatched")

    try:
        model = gp.Model()
        model.setParam('OutputFlag', verbose)
        model.setParam("Threads", 1)

        p_var = model.addVars(n_nodes, n2, name = 'p')
        B_var = model.addVars(d, p, name = "B")
        model.update()

        for i in range(n):
            for j in range(d):
                constr_lhs = -gp.quicksum(A_mat_trans[j, k] * p_var[k, i] for k in range(n_nodes))
                constr_rhs = c[j, i] - 2 * gp.quicksum(B_var[j, k] * X[k, i] for k in range(p))
                model.addConstr(constr_lhs >= constr_rhs)
        model.update()

        obj_noreg = 0 
        for i in range(n):
            term1 = - gp.quicksum(b_vec[k] * p_var[k, i] for k in range(n_nodes))
            term2 = 2 * gp.quicksum(w_star_train[j, i] * B_var[j, k] * X[k, i] for j in range(d) for k in range(p))
            obj_noreg = obj_noreg + term1 + term2 - z_star_train[i]

        obj = obj_noreg + n*(cur_lambda/2) * gp.quicksum(B_var[i, j]*B_var[i, j] for i in range(d) for j in range(p))
        model.setObjective(obj, GRB.MINIMIZE)

        model.update()

        model.optimize()
        B_est = np.array([[B_var[k, j].X for j in range(p)] for k in range(d)])
    except Exception as e:
        print("SPO_linear error:", e)
        print("Optimization is unsuccessful with status code", model.status)
        # status code table: https://www.gurobi.com/documentation/9.1/refman/optimization_status_codes.html
    return B_est

def spo_loss(c_hat, c_val, z_star_val, w_star_val, sp_oracle):
    n_holdout = len(z_star_val)
    spo_sum = 0
    
    for i in range(n_holdout):
        (z_oracle, w_oracle) = sp_oracle(c_hat[:, i])
        spo_loss_cur = np.dot(c_val[:,i], w_oracle) - z_star_val[i]
        spo_sum = spo_sum + spo_loss_cur
        
    spo_loss_avg = spo_sum/n_holdout
    return spo_loss_avg

def spo_sgd(features, c, sp_oracle, z_star_train, w_star_train, B_init, cur_lambda, 
                 numiter =  1000, batchsize = 10, long_factor = 0.1):
    
    (p, n) = features.shape
    (d, n2) = c.shape

    def subgrad_stochastic(B_new, cur_lambda):
        G_new = np.zeros(B_new.shape)
        for j in range(batchsize):
            i = np.random.randint(n)
            spoplus_cost_vec = 2*np.matmul(B_new, features[:, i]) - c[:,i]
            (z_oracle, w_oracle) = sp_oracle(spoplus_cost_vec)
            w_star_diff = w_star_train[:, i] - np.array(w_oracle)
            G_new = G_new + 2 * w_star_diff[:, np.newaxis] * features[:,i][np.newaxis, :]
        G_new = (1/batchsize)*G_new + cur_lambda*B_new
        return G_new
    subgrad = subgrad_stochastic
    def step_size_long_dynamic(itn, G_new):
        return long_factor/np.sqrt(itn + 1)
    step_size = step_size_long_dynamic
        
    B_iter = B_init
    B_avg_iter = B_init
    step_size_sum = 0

    for itn in range(numiter):
        G_iter = subgrad(B_iter, cur_lambda)
        step_iter = step_size(itn, G_iter)
        
        step_size_sum = step_size_sum + step_iter
        step_avg = step_iter/step_size_sum
        B_avg_iter_temp = (1 - step_avg)*B_avg_iter + step_avg*B_iter

        B_iter = B_iter - step_iter*G_iter

        B_avg_iter = B_avg_iter_temp
        
    return B_avg_iter

def validation_set_alg_kernel_SPO(A_mat, b_vec,
                       X_train, c_train, X_val, c_val, 
                       z_star_train, w_star_train, 
                       z_star_val, w_star_val,
                       sp_oracle = None, 
                       alg_type = "SGD", 
                       numiter = 1000, batchsize = 10, long_factor = 0.1,
                       gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, num_lambda = 10,
                       verbose = False):
    
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    if lambdas is None:
        lambda_min = lambda_max*lambda_min_ratio
        lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
    if gammas is None:
        gammas = np.array([1/2])

    lambdas = lambdas[lambdas > 1e-6]  # remove lambda close to 0 
    lambdas = lambdas[lambdas <= 1]  # remove lambda larger than 1
    num_gamma = len(gammas)
    num_lambda = len(lambdas)
    validation_loss_list = np.zeros((num_gamma, num_lambda))
    
    for i in range(num_gamma):
        gamma = gammas[i]
        (K_mat_sqrt, K_mat_sqrt_inv, K_design_val) = Kernel_transformation(X_train, X_val, gamma = gamma)
        
        for j in range(num_lambda):
            cur_lambda = lambdas[j]
            
            if alg_type == "reformulation":
                v_est = SPO_reformulation_kernel(K_mat_sqrt, K_mat_sqrt_inv, X_train, c_train, 
                                 z_star_train, w_star_train, 
                                 A_mat, b_vec, cur_lambda = cur_lambda)
                c_hat = spo_kernel_predict(v_est, K_design_val)
                validation_loss_list[i, j] = spo_loss(c_hat, c_val, z_star_val, w_star_val, sp_oracle)
            if alg_type == "SGD":
                v_init = np.zeros((c_train.shape[0], K_mat_sqrt.shape[0]))
                v_est = spo_sgd(K_mat_sqrt, c_train, 
                     sp_oracle, z_star_train, w_star_train, v_init, cur_lambda, 
                     numiter =  numiter, batchsize = batchsize, long_factor = long_factor)
                c_hat = spo_kernel_predict(v_est, K_design_val)
                validation_loss_list[i, j] = spo_loss(c_hat, c_val, z_star_val, w_star_val, sp_oracle)
    
    ind_list = list(np.ndindex(validation_loss_list.shape))
    best_ind = ind_list[np.argmin(validation_loss_list)]
    best_gamma = gammas[best_ind[0]]
    best_lambda = lambdas[best_ind[1]]
    
    return (best_gamma, best_lambda, validation_loss_list)

def validation_set_alg_kernel_ridge(A_mat, b_vec,
                       X_train, c_train, X_val, c_val, 
                       z_star_train, w_star_train, 
                       z_star_val, w_star_val,
                       sp_oracle = None,
                       gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, num_lambda = 10,
                       verbose = False):
    
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    if lambdas is None:
        lambda_min = lambda_max*lambda_min_ratio
        lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
    if gammas is None:
        gammas = np.array([1/2])

    lambdas = lambdas[lambdas > 1e-6]  # remove lambda close to 0 
    lambdas = lambdas[lambdas <= 1]  # remove lambda larger than 1
    num_gamma = len(gammas)
    num_lambda = len(lambdas)
    validation_loss_list = np.zeros((num_gamma, num_lambda))
    
    for i in range(num_gamma):
        for j in range(num_lambda):
            gamma = gammas[i]
            cur_lambda = lambdas[j]
            
            kr = ridge_kernel(X_train, c_train, gamma = gamma, cur_lambda = cur_lambda)
            c_hat = kr.predict(np.transpose(X_val))
            err = c_hat - np.transpose(c_val)
            validation_loss_list[i, j] = np.sqrt(np.mean(np.sum(np.power(err, 2), axis = 1)))
    
    ind_list = list(np.ndindex(validation_loss_list.shape))
    best_ind = ind_list[np.argmin(validation_loss_list)]
    best_gamma = gammas[best_ind[0]]
    best_lambda = lambdas[best_ind[1]]
    
    return (best_gamma, best_lambda, validation_loss_list)

def validation_set_alg_linear_ridge(A_mat, b_vec,
                       X_train, c_train, X_val, c_val, 
                       z_star_train, w_star_train, 
                       z_star_val, w_star_val,
                       sp_oracle = None, 
                       lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, num_lambda = 10,
                       verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    if lambdas is None:
        lambda_min = lambda_max*lambda_min_ratio
        lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
    num_lambda = len(lambdas)
    validation_loss_list = np.zeros(num_lambda)
    
    for i in range(num_lambda):
        cur_lambda = lambdas[i]
        if cur_lambda >= 1e-6:
            ridge = ridge_linear(X_train, c_train, cur_lambda = cur_lambda)
            c_hat = ridge.predict(np.transpose(X_val))
            err = c_hat - np.transpose(c_val)
            validation_loss_list[i] = np.sqrt(np.mean(np.sum(np.power(err, 2), axis = 1)))
        if cur_lambda < 1e-6: # if lambda is close to 0, then run linear regressions 
            ls = ls_linear(X_train, c_train)
            c_hat = ls.predict(np.transpose(X_val))
            err = c_hat - np.transpose(c_val)
            validation_loss_list[i] = np.sqrt(np.mean(np.sum(np.power(err, 2), axis = 1)))

    best_ind = np.argmin(validation_loss_list)
    best_lambda = lambdas[best_ind]
    
    return (best_lambda, validation_loss_list)

def validation_set_alg_linear_SPO(A_mat, b_vec,
                       X_train, c_train, X_val, c_val, 
                       z_star_train, w_star_train, 
                       z_star_val, w_star_val,
                       sp_oracle = None, 
                       lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, num_lambda = 10,
                       verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    if lambdas is None:
        lambda_min = lambda_max*lambda_min_ratio
        lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
    num_lambda = len(lambdas)
    validation_loss_list = np.zeros(num_lambda)
    
    for i in range(num_lambda):
        cur_lambda = lambdas[i]
        B_est = SPO_reformulation_linear(X_train, c_train, z_star_train, w_star_train, A_mat, b_vec, cur_lambda = cur_lambda, verbose = verbose)
        c_hat = spo_linear_predict(B_est, X_val)
        validation_loss_list[i] = spo_loss(c_hat, c_val, z_star_val, w_star_val, sp_oracle)


    best_ind = np.argmin(validation_loss_list)
    best_lambda = lambdas[best_ind]
    
    return (best_lambda, validation_loss_list)

def replication_linear_spo(A_mat, b_vec, B_true, 
                             X_train, c_train, c_train_exp,
                             X_val, c_val, c_val_exp,
                             X_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val,
                             z_star_test, w_star_test, 
                             sp_oracle = None, 
                             lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    (best_lambda_spo_linear, validation_loss_list) = validation_set_alg_linear_SPO(A_mat, b_vec,
                           X_train, c_train, X_val, c_val, 
                           z_star_train, w_star_train, 
                           z_star_val, w_star_val,
                           sp_oracle = sp_oracle, 
                           lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                           verbose = verbose)
    spo_linear_best = SPO_reformulation_linear(X_train, c_train, z_star_train, w_star_train, A_mat, b_vec, cur_lambda = best_lambda_spo_linear)
    c_hat_spo_linear = spo_linear_predict(spo_linear_best, X_test)
    regret_spo_linear = spo_loss(c_hat_spo_linear, c_test_exp, z_star_test, w_star_test, sp_oracle)

    return (regret_spo_linear, best_lambda_spo_linear, validation_loss_list)

def replication_linear_ridge(A_mat, b_vec, B_true, 
                             X_train, c_train, c_train_exp,
                             X_val, c_val, c_val_exp,
                             X_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val,
                             z_star_test, w_star_test, 
                             sp_oracle = None, 
                             lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    (best_lambda_ridge_linear, validation_loss_list) = validation_set_alg_linear_ridge(A_mat, b_vec,
                           X_train, c_train, X_val, c_val, 
                           z_star_train, w_star_train, 
                           z_star_val, w_star_val,
                           sp_oracle = sp_oracle, 
                           lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                           verbose = verbose)
    if best_lambda_ridge_linear >= 1e-6:
        ridge_linear_best = ridge_linear(X_train, c_train, cur_lambda = best_lambda_ridge_linear)
        c_hat_ridge_linear = np.transpose(ridge_linear_best.predict(np.transpose(X_test)))
        regret_ridge_linear = spo_loss(c_hat_ridge_linear, c_test_exp, z_star_test, w_star_test, sp_oracle)
            # note that we are using the true regression function output c_test_exp 
            # (without noise) to evaluate the regret, and z_star_test, w_star_test are also generated 
            # from c_test_exp
    if best_lambda_ridge_linear <= 1e-6:  # if lambda is close to 0, then run linear regressions 
        ls_linear_best = ls_linear(X_train, c_train)
        c_hat_ls_linear = np.transpose(ls_linear_best.predict(np.transpose(X_test)))
        # spo_loss(c_hat, c_val, z_star_val, w_star_val, sp_oracle)
        regret_ridge_linear = spo_loss(c_hat_ls_linear, c_test_exp, z_star_test, w_star_test, sp_oracle)

    return (regret_ridge_linear, best_lambda_ridge_linear, validation_loss_list)

def replication_kernel_ridge(A_mat, b_vec, B_true, 
                             X_train, c_train, c_train_exp,
                             X_val, c_val, c_val_exp,
                             X_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val, 
                             z_star_test, w_star_test, 
                             sp_oracle = None, 
                             gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    (best_gamma_ridge_kernel, best_lambda_ridge_kernel, validation_loss_list) = validation_set_alg_kernel_ridge(A_mat, b_vec,
                       X_train, c_train, X_val, c_val, 
                       z_star_train, w_star_train, 
                       z_star_val, w_star_val,
                       sp_oracle = sp_oracle, 
                       gammas = gammas, lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                       verbose = verbose)
    ridge_kernel_best = ridge_kernel(X_train, c_train, gamma = best_gamma_ridge_kernel, cur_lambda = best_lambda_ridge_kernel)
    c_hat_ridge_kernel = np.transpose(ridge_kernel_best.predict(np.transpose(X_test)))
    regret_ridge_kernel = spo_loss(c_hat_ridge_kernel, c_test_exp, z_star_test, w_star_test, sp_oracle)
    
    return (regret_ridge_kernel, best_gamma_ridge_kernel, best_lambda_ridge_kernel, validation_loss_list)

def replication_kernel_spo(A_mat, b_vec, B_true, 
                             X_train, c_train, c_train_exp,
                             X_val, c_val, c_val_exp,
                             X_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val, 
                             z_star_test, w_star_test, 
                             sp_oracle = None, 
                             alg_type = "SGD", 
                             numiter = 1000, batchsize = 10, long_factor = 0.1,
                             gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)

    (best_gamma_spo_kernel, best_lambda_spo_kernel, validation_loss_list) = validation_set_alg_kernel_SPO(A_mat, b_vec,
                           X_train, c_train, X_val, c_val, 
                           z_star_train, w_star_train, 
                           z_star_val, w_star_val, 
                           sp_oracle = sp_oracle, 
                           alg_type = alg_type, 
                           numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                           gammas = gammas, lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                           verbose = verbose)

    (K_mat_sqrt, K_mat_sqrt_inv, K_design_test) = Kernel_transformation(X_train, X_test, gamma = best_gamma_spo_kernel)

    if alg_type == "reformulation":
        spo_kernel_best =  SPO_reformulation_kernel(K_mat_sqrt, K_mat_sqrt_inv, X_train, c_train, z_star_train, w_star_train,  A_mat, b_vec, cur_lambda = best_lambda_spo_kernel)
        c_hat_spo_kernel = spo_kernel_predict(spo_kernel_best, K_design_test)
        regret_spo_kernel = spo_loss(c_hat_spo_kernel, c_test_exp, z_star_test, w_star_test, sp_oracle)
    if alg_type == "SGD":
        v_init = np.zeros((c_train.shape[0], K_mat_sqrt.shape[0]))
        v_est = spo_sgd(K_mat_sqrt, c_train, 
                     sp_oracle, z_star_train, w_star_train, v_init, best_lambda_spo_kernel, 
                     numiter =  numiter, batchsize = batchsize, long_factor = long_factor)
        c_hat_spo_kernel = spo_kernel_predict(v_est, K_design_test)
        regret_spo_kernel = spo_loss(c_hat_spo_kernel, c_test_exp, z_star_test, w_star_test, sp_oracle)

    return (regret_spo_kernel, best_gamma_spo_kernel, best_lambda_spo_kernel, validation_loss_list)
  
def replication_no_kernel(A_mat, b_vec, B_true, 
                             X_false_train, X_true_train, c_train, c_train_exp,
                              X_false_val, X_true_val, c_val, c_val_exp,
                             X_false_test, X_true_test,  c_test, c_test_exp, 
                            gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):

    n_train = X_false_train.shape[1]
    n_holdout = X_false_val.shape[1]
    n_test = X_false_test.shape[1]
    # generate oracle solution 
    sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    (z_star_test, w_star_test) = oracle_dataset(c_test_exp, sp_oracle)
        # note that for test data, we are using the true regression function output c_test_exp 
        # (without noise) to generate the oracle values.
    (z_star_train, w_star_train) = oracle_dataset(c_train, sp_oracle)
    (z_star_val, w_star_val) = oracle_dataset(c_val, sp_oracle)

    # misspecified spo 
    time0 = time.time() 
    (regret_spo_false, best_lambda_spo_linear, validation_loss_spo_linear) = replication_linear_spo(A_mat, b_vec, B_true, 
                     X_false_train, c_train, c_train_exp,
                     X_false_val, c_val, c_val_exp,
                     X_false_test, c_test, c_test_exp,
                     z_star_train, w_star_train, 
                     z_star_val, w_star_val,
                     z_star_test, w_star_test, 
                     sp_oracle = sp_oracle, 
                     lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                   verbose = verbose)
    time1 = time.time()
    spo_linear_time = time1 - time0

    # misspecified eto  
    time0 = time.time() 
    (regret_ridge_linear, best_lambda_ridge_linear, validation_loss_ridge_linear) = replication_linear_ridge(A_mat, b_vec, B_true, 
                     X_false_train, c_train, c_train_exp,
                     X_false_val, c_val, c_val_exp,
                     X_false_test, c_test, c_test_exp,
                     z_star_train, w_star_train, 
                     z_star_val, w_star_val,
                     z_star_test, w_star_test, 
                     sp_oracle = sp_oracle, 
                     lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                   verbose = verbose)
    time1 = time.time()
    ridge_linear_time = time1 - time0
    
    # correct spo 
    time0 = time.time() 
    (regret_spo_correct, best_lambda_spo_correct, validation_loss_spo_correct) = replication_linear_spo(A_mat, b_vec, B_true, 
                     X_true_train, c_train, c_train_exp,
                     X_true_val, c_val, c_val_exp,
                     X_true_test, c_test, c_test_exp,
                     z_star_train, w_star_train, 
                     z_star_val, w_star_val,
                     z_star_test, w_star_test, 
                     sp_oracle = sp_oracle, 
                     lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                   verbose = verbose)
    time1 = time.time()
    spo_correct_time = time1 - time0

    # correct eto  
    time0 = time.time() 
    (regret_ridge_correct, best_lambda_ridge_correct, validation_loss_ridge_correct) = replication_linear_ridge(A_mat, b_vec, B_true, 
                     X_true_train, c_train, c_train_exp,
                     X_true_val, c_val, c_val_exp,
                     X_true_test, c_test, c_test_exp,
                     z_star_train, w_star_train, 
                     z_star_val, w_star_val,
                     z_star_test, w_star_test, 
                     sp_oracle = sp_oracle, 
                     lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                   verbose = verbose)
    time1 = time.time()
    ridge_correct_time = time1 - time0

    regret_all = {"zstar_avg_test": z_star_test.mean(),
                 "SPO_wrong": regret_spo_false,
                "ETO_wrong": regret_ridge_linear,
                 "SPO_correct": regret_spo_correct,
                 "ETO_correct": regret_ridge_correct}
    validation_all = {"best_lambda_spo_linear": best_lambda_spo_linear,
                 "best_lambda_ridge_linear": best_lambda_ridge_linear,
                  "best_lambda_SPO_correct": best_lambda_spo_correct,
                 "best_lambda_ETO_correct": best_lambda_ridge_correct}
    time_all = {"SPO_wrong": spo_linear_time,
                "ETO_wrong": ridge_linear_time,
                "SPO_correct": spo_correct_time,
                 "ETO_correct": ridge_correct_time}
    validation_loss_all = {"SPO_wrong": validation_loss_spo_linear,
                "ETO_wrong": validation_loss_ridge_linear,
                "SPO_correct": validation_loss_spo_correct,
                 "ETO_correct": validation_loss_ridge_correct}
    return (regret_all, validation_all, time_all, validation_loss_all)

def replication_kernel_SGD(A_mat, b_vec, B_true, 
                              X_false_train, X_true_train, c_train, c_train_exp,
                              X_false_val, X_true_val, c_val, c_val_exp,
                             X_false_test, X_true_test,  c_test, c_test_exp, 
                             numiter = 1000, batchsize = 10, long_factor = 1,
                            gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):

    n_train = X_false_train.shape[1]
    n_holdout = X_false_val.shape[1]
    n_test = X_false_test.shape[1]
    # generate oracle solution 
    sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    (z_star_test, w_star_test) = oracle_dataset(c_test_exp, sp_oracle)
        # note that for test data, we are using the true regression function output c_test_exp 
        # (without noise) to generate the oracle values.
    (z_star_train, w_star_train) = oracle_dataset(c_train, sp_oracle)
    (z_star_val, w_star_val) = oracle_dataset(c_val, sp_oracle)

    # kernel spo 
    time0 = time.time() 
    (regret_spo_kernel, best_gamma_spo_kernel, best_lambda_spo_kernel, validation_loss_spo_kernel) = replication_kernel_spo(A_mat, b_vec, B_true, 
                             X_false_train, c_train, c_train_exp,
                             X_false_val, c_val, c_val_exp,
                             X_false_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val,
                             z_star_test, w_star_test, 
                             sp_oracle = sp_oracle, 
                             alg_type = "SGD", 
                             numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                             gammas = gammas, lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                             verbose = verbose)
    time1 = time.time()
    spo_kernel_time = time1 - time0

    # kernel eto  
    time0 = time.time() 
    (regret_ridge_kernel, best_gamma_ridge_kernel, best_lambda_ridge_kernel, validation_loss_ridge_kernel) = replication_kernel_ridge(A_mat, b_vec, B_true, 
                             X_false_train, c_train, c_train_exp,
                             X_false_val, c_val, c_val_exp,
                             X_false_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val,
                             z_star_test, w_star_test, 
                             sp_oracle = sp_oracle, 
                             gammas = gammas, lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                             verbose = verbose)
    time1 = time.time()
    ridge_kernel_time = time1 - time0
    
    

    regret_all = {"SPO_kernel": regret_spo_kernel, 
                "ETO_kernel": regret_ridge_kernel}
    validation_all = {"best_gamma_spo_kernel": best_gamma_spo_kernel, 
                "best_lambda_spo_kernel": best_lambda_spo_kernel,
                 "best_gamma_ridge_kernel": best_gamma_ridge_kernel,
                 "best_lambda_ridge_kernel": best_lambda_ridge_kernel}
    time_all = {"SPO_kernel": spo_kernel_time, 
                "ETO_kernel": ridge_kernel_time}
    validation_loss_all = {"SPO_kernel": validation_loss_spo_kernel, 
                "ETO_kernel": validation_loss_ridge_kernel}
    return (regret_all, validation_all, time_all, validation_loss_all)

def replication_kernel_gurobi(A_mat, b_vec, B_true, 
                             X_false_train, X_true_train, c_train, c_train_exp,
                              X_false_val, X_true_val, c_val, c_val_exp,
                             X_false_test, X_true_test,  c_test, c_test_exp, 
                             numiter = 1000, batchsize = 10, long_factor = 0.1,
                            gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):

    n_train = X_false_train.shape[1]
    n_holdout = X_false_val.shape[1]
    n_test = X_false_test.shape[1]
    # generate oracle solution 
    sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    (z_star_test, w_star_test) = oracle_dataset(c_test_exp, sp_oracle)
        # note that for test data, we are using the true regression function output c_test_exp 
        # (without noise) to generate the oracle values.
    (z_star_train, w_star_train) = oracle_dataset(c_train, sp_oracle)
    (z_star_val, w_star_val) = oracle_dataset(c_val, sp_oracle)

    # kernel spo reformulation
    time0 = time.time() 
    (regret_spo_kernel, best_gamma_spo_kernel, best_lambda_spo_kernel, validation_loss_spo_kernel) = replication_kernel_spo(A_mat, b_vec, B_true, 
                             X_false_train, c_train, c_train_exp,
                             X_false_val, c_val, c_val_exp,
                             X_false_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val,
                             z_star_test, w_star_test, 
                             sp_oracle = sp_oracle, 
                             alg_type = "reformulation", 
                             numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                             gammas = gammas, lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                             verbose = verbose)
    time1 = time.time()
    spo_kernel_time = time1 - time0
    

    regret_all = {"SPO_kernel": regret_spo_kernel}
    validation_all = {"best_gamma_spo_kernel": best_gamma_spo_kernel, 
                "best_lambda_spo_kernel": best_lambda_spo_kernel}
    time_all = {"SPO_kernel": spo_kernel_time}
    validation_loss_all = {"SPO_kernel": validation_loss_spo_kernel}
    return (regret_all, validation_all, time_all, validation_loss_all)



