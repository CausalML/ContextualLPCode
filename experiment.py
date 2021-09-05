from utility_functions import * 
import mkl
mkl.set_num_threads(1)

n_train_seq = np.array([50, 100])
# n_train_seq = np.array([50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
n_holdout_seq = n_train_seq
n_train_max = 1000
n_holdout_max = 1000
n_test = 1000
# runs = 50
# n_jobs = 50
runs = 5
n_jobs = 5

numiter = 1000; batchsize = 10; long_factor = 1;

p = 5
polykernel_degree = 5
grid_dim = 5
noise_half_width = 0.25
constant  = 3
verbose = False 
(A_mat, b_vec) = read_A_and_b()
(n_nodes, n_edges) = A_mat.shape

X = np.random.normal(size=(p, 50))
poly = PolynomialFeatures(degree = polykernel_degree, interaction_only = True, include_bias = False)
X_new = np.transpose(poly.fit_transform(np.transpose(X)))
(p2, _) = X_new.shape

np.random.seed(10)
B_true = np.random.rand(n_edges, p2) 
data_train = [generate_data_interactive(B_true, n_train_max, p, polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_holdout = [generate_data_interactive(B_true, n_holdout_max, p, polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_test = generate_data_interactive(B_true, n_test, p, polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) 

# lambda_max = 100
# num_lambda = 10
# lambda_min_ratio = 1e-3
# lambda_min = lambda_max*lambda_min_ratio
# lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
# lambdas = np.round(lambdas, 2)
# lambdas = np.concatenate((np.array([0, 0.001, 0.01]), lambdas))
# gammas = np.array([0.01, 0.1, 0.5, 1, 2])
lambda_max = 100
num_lambda = 2
lambda_min_ratio = 1e-3
lambda_min = lambda_max*lambda_min_ratio
lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
lambdas = np.round(lambdas, 2)
gammas = np.array([0.1, 0.5])

output = "experiment.txt"
with open(output, 'w') as f:
    print("start", file = f)
    
################
#  No kernel  ##
################
with open(output, 'a') as f:
    print("################" , file = f)
    print("#  no_kernel   #", file = f)
    print("################" , file = f)

regret_all = []
validation_all = []
time_all = []
loss_all = []

for i in range(len(n_train_seq)):
    n_train = n_train_seq[i]
    n_holdout = n_holdout_seq[i] 
    
    with open(output, 'a') as f:
        print("n_train", n_train, file = f)
    time1 = time.time()

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_no_kernel)(
                           A_mat, b_vec, B_true, 
                           data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][2][:, :n_train], data_train[run][3][:, :n_train],
                           data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][2][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
                           data_test[0], data_test[1], data_test[2], data_test[3],
                           # numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                           #   loss_stop = loss_stop, tol = tol, stop = stop,
                           gammas = gammas, lambdas = lambdas, lambda_max = None, lambda_min_ratio = None, num_lambda = None,
                           verbose = verbose) for run in range(runs))
    
    regret_temp = [res_0[0] for res_0 in res_temp]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    validation_temp = [res_0[1] for res_0 in res_temp]
    validation_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in validation_temp]
    validation_temp = pd.concat(validation_temp)
    
    time_temp = [res_0[2] for res_0 in res_temp]
    time_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in time_temp]
    time_temp = pd.concat(time_temp)
    
    loss_temp = [res_0[3] for res_0 in res_temp]
    
    regret_temp["n"] = n_train
    validation_temp["n"] = n_train
    time_temp["n"] = n_train
    
    with open(output, 'a') as f:
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    regret_all.append(regret_temp)
    validation_all.append(validation_temp)
    time_all.append(time_temp)
    loss_all.append(loss_temp)
    
    pd.concat(regret_all).to_csv("regret_no_kernel.csv", index = False)
    pd.concat(validation_all).to_csv("validation_no_kernel.csv", index = False)
    pd.concat(time_all).to_csv("time_no_kernel.csv", index = False)
    pickle.dump(loss_all, open("loss_no_kernel.pkl", "wb"))

################
#  Kernel SGD ##
################
with open(output, 'a') as f:
    print("################" , file = f)
    print("#  kernel_SGD  #", file = f)
    print("################" , file = f)

regret_all = []
validation_all = []
time_all = []
loss_all = []

for i in range(len(n_train_seq)):
    n_train = n_train_seq[i]
    n_holdout = n_holdout_seq[i] 
    
    with open(output, 'a') as f:
        print("n_train", n_train, file = f)
    time1 = time.time()
    

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_kernel_SGD)(
                           A_mat, b_vec, B_true, 
                           data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][2][:, :n_train], data_train[run][3][:, :n_train],
                           data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][2][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
                           data_test[0], data_test[1], data_test[2], data_test[3],
                           numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                           gammas = gammas, lambdas = lambdas, lambda_max = None, lambda_min_ratio = None, num_lambda = None,
                           verbose = verbose) for run in range(runs))
    
    regret_temp = [res_0[0] for res_0 in res_temp]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    validation_temp = [res_0[1] for res_0 in res_temp]
    validation_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in validation_temp]
    validation_temp = pd.concat(validation_temp)
    
    time_temp = [res_0[2] for res_0 in res_temp]
    time_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in time_temp]
    time_temp = pd.concat(time_temp)
    
    loss_temp = [res_0[3] for res_0 in res_temp]
    
    regret_temp["n"] = n_train
    validation_temp["n"] = n_train
    time_temp["n"] = n_train
    
    with open(output, 'a') as f:
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    regret_all.append(regret_temp)
    validation_all.append(validation_temp)
    time_all.append(time_temp)
    loss_all.append(loss_temp)
    
    pd.concat(regret_all).to_csv("regret_kernel_SGD.csv", index = False)
    pd.concat(validation_all).to_csv("validation_kernel_SGD.csv", index = False)
    pd.concat(time_all).to_csv("time_kernel_SGD.csv", index = False)
    pickle.dump(loss_all, open("loss_kernel_SGD.pkl", "wb"))

###################
#  Kernel gruobi ##
###################
with open(output, 'a') as f:
    print("###################", file = f)
    print("#  kernel_gurobi  #", file = f)
    print("###################", file = f)

regret_all = []
validation_all = []
time_all = []
loss_all = []

for i in range(len(n_train_seq)):
    if i > 5:
        break

    n_train = n_train_seq[i]
    n_holdout = n_holdout_seq[i] 
    
    with open(output, 'a') as f:
        print("n_train", n_train, file = f)
    time1 = time.time()
    

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_kernel_gurobi)(
                           A_mat, b_vec, B_true, 
                           data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][2][:, :n_train], data_train[run][3][:, :n_train],
                           data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][2][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
                           data_test[0], data_test[1], data_test[2], data_test[3],
                           numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                           gammas = gammas, lambdas = lambdas, lambda_max = None, lambda_min_ratio = None, num_lambda = None,
                           verbose = verbose) for run in range(runs))
    
    regret_temp = [res_0[0] for res_0 in res_temp]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    validation_temp = [res_0[1] for res_0 in res_temp]
    validation_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in validation_temp]
    validation_temp = pd.concat(validation_temp)
    
    time_temp = [res_0[2] for res_0 in res_temp]
    time_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in time_temp]
    time_temp = pd.concat(time_temp)
    
    loss_temp = [res_0[3] for res_0 in res_temp]
    
    regret_temp["n"] = n_train
    validation_temp["n"] = n_train
    time_temp["n"] = n_train
    
    with open(output, 'a') as f:
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    regret_all.append(regret_temp)
    validation_all.append(validation_temp)
    time_all.append(time_temp)
    loss_all.append(loss_temp)
    
    pd.concat(regret_all).to_csv("regret_kernel_gurobi.csv", index = False)
    pd.concat(validation_all).to_csv("validation_kernel_gurobi.csv", index = False)
    pd.concat(time_all).to_csv("time_kernel_gurobi.csv", index = False)
    pickle.dump(loss_all, open("loss_kernel_gurobi.pkl", "wb"))