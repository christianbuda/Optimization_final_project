import pandas as pd
import numpy as np
import time
import cvxopt.solvers
from cvxopt import matrix, solvers
from itertools import product
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

####################### DATA LOADING ########################################################
def MulticlassSVM_data_loading(l1, l2, l3):
    # data loading routine
    
    # load data
    full_data = pd.read_csv('data.txt')

    # select only the 3 input letters
    data = pd.DataFrame(full_data.loc[(full_data['Y'] == l1) | (full_data['Y'] == l2) | (full_data['Y'] == l3)])

    # replace l1 with 0
    #         l2 with 1
    #         l3 with 2
    data.replace({l1: 0, l2: 1, l3: 2}, inplace = True)

    # the final dataset
    X = data.drop('Y', axis=1).to_numpy()
    y = data['Y'].to_numpy()

    # set seed for the random split
    np.random.seed(42)
    # split in train and test
    test_idx = np.random.choice(len(y), size = int(0.2*len(y)), replace = False)

    X_train = np.delete(X, test_idx, axis = 0)
    y_train = np.delete(y, test_idx, axis = 0)

    X_test = X[test_idx,]
    y_test = y[test_idx,]
    
    return X_train, y_train, X_test, y_test

####################### DATA SCALING ########################################################
class IdentityScaler():
    # this does basically nothing
    def __init__(self):
        return
    
    def fit(self, X):
        return
    
    def fit_transform(self, X):
        return X
    
    def transform(self, X):
        return X
    

    
class StdScaler():
    # standardize input data to have mean 0 unit variance
    def __init__(self):
        self.mu = 0
        self.sd = 1
    
    def fit(self, X):
        # compute mean and standard deviation
        self.mu = X.mean(axis = 0)
        self.sd = np.sqrt(X.var(axis = 0))
        return
    
    def fit_transform(self, X):
        # compute mean and standard deviation
        self.mu = X.mean(axis = 0)
        self.sd = np.sqrt(X.var(axis = 0))
        
        # rescale unlabelled data matrix
        return (X-self.mu)/self.sd
    
    def transform(self, X):
        # rescale unlabelled data matrix
        return (X-self.mu)/self.sd

####################### MODEL EVALUATION ########################################################
def accuracy(y_pred, y_true):
    # simple function to compute accuracy
    return(np.mean(y_pred == y_true))

def plot_confusion_matrices(model, X_train, y_train, X_test, y_test, labels):
    fig,ax = plt.subplots(1,2, figsize = (13,5))

    actual = y_train
    predicted = model.predict(X_train)
    cm_display = metrics.ConfusionMatrixDisplay.from_predictions(actual, predicted, ax = ax[0])
    ax[0].set_xticklabels(labels, rotation = 0)
    ax[0].set_yticklabels(labels, rotation = 0)
    ax[0].set_title('Training Confusion Matrix')


    actual = y_test
    predicted = model.predict(X_test)
    cm_display = metrics.ConfusionMatrixDisplay.from_predictions(actual, predicted, ax = ax[1])
    ax[1].set_xticklabels(labels, rotation = 0)
    ax[1].set_yticklabels(labels, rotation = 0)
    ax[1].set_title('Test Confusion Matrix')


    plt.show()

####################### CROSS VALIDATION ########################################################
def GridSearchCV(X, y, model_generation_fun, param_grid, k = 5, **kwargs):
    # function to automatically do k-fold cross validation over a grid of parameters
    
    ## kwargs: are the other fixed parameters to be passed to model_generation_fun
    ## k:      is the number of "folds"
    
    # k-fold cross validation dataset split
    idx_split = np.array_split(np.random.permutation(len(y)), k)

    # length of the parameter list for each of the parameters
    gridsearch_size = tuple(map(len, param_grid.values()))

    # final validation accuracy matrix
    avg_acc = np.zeros(gridsearch_size)

    # loop over all possible tuple of indices
    for indices in tqdm(product(*map(np.arange, gridsearch_size)), total = np.prod(gridsearch_size)):
        # select current value for each of the parameters
        current_params = map( lambda x, i: x[i] , param_grid.values(),indices )

        # pack values back into a dictionary
        current_params = dict(zip(param_grid.keys(), current_params ))

        # train and compute accuracy for each split
        model_accuracies = list(map(lambda idx: model_generation_fun(**current_params,  **kwargs).split_train_evaluate(X, y, idx), idx_split))

        # store mean accuracy
        avg_acc[indices] = np.mean(model_accuracies)
    
    best_params = np.unravel_index(avg_acc.argmax(), avg_acc.shape)
    # select current value for each of the parameters
    best_params = map( lambda x, i: x[i] , param_grid.values(),best_params )
    # pack values back into a dictionary
    best_params = dict(zip(param_grid.keys(), best_params ))
    
    # initialize the final model
    model = model_generation_fun(**best_params,  **kwargs)
    model.fit(X, y)
    
    return({'best_model' : model, 'best_val_acc':avg_acc.max(), 'best_params':best_params, 'avg_val_acc':avg_acc})

####################### KERNEL FUNCTIONS ########################################################
def linear_kernel_matrix(X, Y):
    # linear kernel function
    return( np.dot(X,Y.T) )

def gaussian_kernel_matrix(X, Y, gamma):
    # gaussian kernel function
    
    if X is Y:
        # checks if the two objects are the same
        # so that the norm is only computed once
        norm_X = np.sum(X*X, axis = 1)
        norm_Y = norm_X[np.newaxis,:]
        norm_X = norm_X[:,np.newaxis]
    else:
        norm_X = np.sum(X*X, axis = 1)[:,np.newaxis]
        norm_Y = np.sum(Y*Y, axis = 1)[np.newaxis,:]
    
    return(np.exp( -gamma * (norm_X + norm_Y - 2*np.dot(X,Y.T)) ))

def polynomial_kernel_matrix(X, Y, p):
    # polynomial kernel function
    return( (np.dot(X,Y.T) + 1)**p )

def get_kernel_func(kernel_type, kernel_params):
    # function that takes a keyword in input
    # and returns the corresponding kernel function in output
    
    available_kernels = {'linear':linear_kernel_matrix,
                         'gaussian':gaussian_kernel_matrix,
                         'polynomial':polynomial_kernel_matrix}
    
    kernel_type = kernel_type.lower()
    if kernel_type not in available_kernels.keys():
        raise ValueError("Kernel type must be one of ['linear', 'gaussian', 'polynomial'].")
    
    selected_kernel = available_kernels[kernel_type]
    
    def kernel_fun(X,Y):
        return selected_kernel(X,Y,**kernel_params)
    
    return kernel_fun

####################### MVP-SMO AUXILIARY FUNCTIONS ########################################################
def compute_RS(lambda_star, C, class_plus, class_minus):
    # computes the sets of indices R and S as boolean masks
    
    mask_lower = np.isclose(lambda_star, 0)  # L
    mask_upper = np.isclose(lambda_star, C)  # U
    mask_mid = np.logical_not(np.logical_or(mask_lower, mask_upper))
    
    L_plus = np.logical_and(mask_lower, class_plus)
    L_minus = np.logical_and(mask_lower, class_minus)
    U_plus = np.logical_and(mask_upper, class_plus)
    U_minus = np.logical_and(mask_upper, class_minus)
    
    R = np.any((L_plus, U_minus, mask_mid), axis=0)
    S = np.any((L_minus, U_plus, mask_mid), axis=0)
    return(R,S)

def find_MVP(R,S, objective, data_idx):
    # computes a most violating pair starting from R and S
    
    ## objective: is the objective function, i.e. -grad(f)/y
    ## data_idx:  original indices of the data
    
    i = objective[R].argmax()
    i = data_idx[R][i]
    m = objective[i]
    
    j = objective[S].argmin()
    j = data_idx[S][j]
    M = objective[j]
    
    return i,j,m,M

def solve_subproblem(W, lambda_star, K, y, C):
    # solves analitically the constrained quadratic optimization subproblem with 2 variables
    
    ## W:           the working set
    ## lambda_star: the multipliers
    ## K:           the kernel matrix
    
    # MVP
    i,j = W
    
    tmp = np.delete(y*lambda_star, W, axis = 0)
    tmp_sum = np.sum(tmp)
    
    a = K[i,i] + K[j,j] - 2*K[i,j]
    b = y[j] * ( y[i]-y[j] + tmp_sum * (K[i,i] - K[i,j]) + np.sum( tmp * np.delete(K[j,]-K[i,], W, axis = 0)) )
    
    # unconstrained minimum of the subproblem in lambda_j (the 2 at the denominator is embedded in a)
    unconstrained_min = -b/a
    
    # enforce that both lambda_i and lambda_j are in [0,C]
    if y[i]*y[j] > 0:
        lower = max([0, - y[i] * tmp_sum - C])
        upper = min([C, - y[i] * tmp_sum])
    else:
        lower = max([0, y[i] * tmp_sum])
        upper = min([C, C + y[i] * tmp_sum])
    
    # constrained solution
    if unconstrained_min>= lower and unconstrained_min<= upper:
        lambda_j = unconstrained_min
    elif unconstrained_min>upper:
        lambda_j = upper
    else:
        lambda_j = lower
    
    lambda_i = - y[i]*y[j] * lambda_j - y[i] * tmp_sum
    
    return lambda_i, lambda_j

####################### THE BASIC SVM MODELS ########################################################
class Model():
    # basic model class object
    def __init__(self):
        pass
    
    def __call__(self, X):
        raise NotImplementedError
        
    def fit(self, X, y):
        raise NotImplementedError
        
    def predict(self, X):
        raise NotImplementedError
        
    def evaluate(self, X, y):
        return(accuracy(self.predict(X), y))
    
    def train_evaluate(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        return(accuracy(self.predict(X_test), y_test))

    def split_train_evaluate(self, X, y, split_idx):
        # split dataset
        X_test = X[split_idx,]
        y_test = y[split_idx,]
        X_train  = np.delete(X, split_idx, axis = 0)
        y_train  = np.delete(y, split_idx, axis = 0)

        return( self.train_evaluate(X_train, y_train, X_test, y_test) )

class kSVM(Model):
    # kSVM model class
    
    def __init__(self, scaler = IdentityScaler(), C = 1, method = 'CVXOPT', kernel = 'linear', kernel_params = {}, tol = 1e-5, max_iter = 1e10):
        
        ## method:   the optimization method used to fit the model
        ## kernel:   the kernel type
        ## tol:      tolerance for the stopping criterion for the MVP method
        ## max_iter: maximum number of iterations for the MVP method
        
        if method not in ['CVXOPT', 'MVP-SMO']:
            raise ValueError("Argument method must be one of ['CVXOPT', 'MVP-SMO']")
        self.method = method
        
        if self.method == 'MVP-SMO':
            # set the extra control parameters for the optimization
            self.tol = tol
            self.max_iter = max_iter
            self.final_violation = 0
        
        self.scaler = scaler
        self.C = C
        self.kernel_fun = get_kernel_func(kernel, kernel_params)
        self.exec_time = 0
        self.n_iter = 0
        self.final_dual_obj = 0
        
        self.b_star = None
        self.lambda_SV = None
        self.X_SV = None
        self.y_SV = None
        self.omega_norm = None
    
    def fit(self, X, y):
        # fit the model to the data
        
        # rescale input data
        X = self.scaler.fit_transform(X)
        
        kernel_matrix = self.kernel_fun(X, X)
        
        # compute the multipliers
        if self.method == 'CVXOPT':
            lambda_star = self.fit_CVXOPT(X,y,kernel_matrix)
        if self.method == 'MVP-SMO':
            lambda_star = self.fit_MVPSMO(X,y,kernel_matrix)
        
        # indices of the support vectors
        SV_idx = np.arange(len(y))[np.logical_not(np.isclose(lambda_star, 0))]
        
        # set the final SVM parameters
        self.lambda_SV = lambda_star[SV_idx]
        self.X_SV = X[SV_idx,]
        self.y_SV = y[SV_idx]
        self.b_star = np.mean(self.y_SV - np.sum(self.lambda_SV * self.y_SV*kernel_matrix[:,SV_idx], axis = 0))
        self.omega_norm = np.sqrt(2 * (np.sum(self.lambda_SV) - self.final_dual_obj))
    
    def fit_CVXOPT(self, X,y,kernel_matrix):
        # fit the model using CVXOPT to solve the dual
        
        start = time.time()
        
        # define the variables for the convex optimization routine
        P = matrix(kernel_matrix.astype('float') * y[np.newaxis,:] * y[:,np.newaxis])
        q = matrix(-np.ones(len(y)).astype('float'))
        
        G = matrix(np.concatenate([ -np.eye(len(y)), np.eye(len(y)) ]).astype('float'))
        h = matrix(np.concatenate([ np.zeros(len(y)), self.C*np.ones(len(y)) ]).astype('float'))
        
        A = matrix(y[np.newaxis,].astype('float'))
        b = matrix(np.array([0]).astype('float'))
        
        ###################### run optimization
        solvers.options['show_progress']=False
        
        sol = solvers.qp(P,q,G,h,A,b)
        self.exec_time = time.time() - start
        
        solvers.options['show_progress']=True
        #######################################
        
        self.n_iter = sol['iterations']
        self.final_dual_obj = -sol['primal objective']
        
        # optimal lambda multipliers
        lambda_star = np.array(sol['x'])[:,0]
        
        return(lambda_star)
    
    def fit_MVPSMO(self, X,y,kernel_matrix):
        # fit the model using the MVP-SMO method
        
        # number of iterations
        k = 0
        
        start = time.time()
        
        # fixed values
        data_idx = np.arange(len(y))
        class_plus = y>0
        class_minus = y<0
        
        # matrix representing the quadratic form
        Q = kernel_matrix * y[np.newaxis,:] * y[:,np.newaxis]
        
        ### first step of the method
        
        # feasible initialization
        lambda_zero = np.zeros(X.shape[0])
        # corresponding gradient
        previous_grad = -np.ones(X.shape[0])

        lambda_star = lambda_zero
        current_grad = previous_grad

        R,S = compute_RS(lambda_star, self.C, class_plus, class_minus)

        # if either R or S is empty we have found the optimal solution
        if (not np.any(R)) or (not np.any(S)):
            return lambda_star
        
        i,j,m,M = find_MVP(R,S, -current_grad/y, data_idx)
        
        ### end of the first step
        
        # run optimization
        while(m > M+self.tol):
            
            # working pair
            W = [i,j]
            
            lambda_i, lambda_j = solve_subproblem(W, lambda_star, kernel_matrix, y, self.C)

            # update lambda_star
            lambda_zero = np.array(lambda_star)
            lambda_star[W] = lambda_i, lambda_j

            # update gradient
            previous_grad = current_grad
            current_grad = previous_grad + np.sum( Q[W,] * (lambda_star[W] - lambda_zero[W])[:,np.newaxis], axis = 0)

            # increase iterations
            k += 1

            # recompute R,S
            R,S = compute_RS(lambda_star, self.C, class_plus, class_minus)
            

            # if either R or S is empty we have found the optimal solution
            if (not np.any(R)) or (not np.any(S)):
                break

            i,j,m,M = find_MVP(R,S, -current_grad/y, data_idx)
            
            if k>self.max_iter:
                break
        
        self.exec_time = time.time() - start
        
        if k > self.max_iter:
            print('Maximum number of iteration reached, maybe you can try with a bigger tolerance.')
        
        self.final_violation = m - M
        self.n_iter = k
        self.final_dual_obj = - np.sum(Q* lambda_star[np.newaxis,:] * lambda_star[:,np.newaxis])/2 + np.sum(lambda_star)
        
        return(lambda_star)
        
    def __call__(self, X):
        if self.b_star is None:
            raise Exception("You must run 'fit' before calling the model!")
        
        # reshape X into a matrix if it is a single row element
        if len(X.shape) == 1:
            X = X[np.newaxis,:]
        
        # rescale input data
        X = self.scaler.transform(X)
        
        return(np.sum(self.kernel_fun(X, self.X_SV) * self.lambda_SV * self.y_SV, axis = 1) + self.b_star)
    
    def predict(self, X):
        return(np.sign(self(X)))
    
    def dist_from_hyperplane(self, X):
        return(self(X)/self.omega_norm)

def get_model_CVXOPT(C, kernel_type, **kwargs):
    # get a standard kSVM model with input kernel
    # trained using the CVXOPT method
    return kSVM(StdScaler(), C = C, method = 'CVXOPT', kernel = kernel_type, kernel_params = kwargs)

def get_model_MVPSMO(C, kernel_type, **kwargs):
    # get a standard kSVM model with input kernel
    # trained using the MVP-SMO method
    return kSVM(StdScaler(), C = C, method = 'MVP-SMO', kernel = kernel_type, kernel_params = kwargs)

####################### THE MULTICLASS SVM MODEL ########################################################
class Multiclass_kSVM(Model):
    def __init__(self, scaler = IdentityScaler(), C = 1, nclass = 3, method = 'MVP-SMO', kernel = 'linear', kernel_params = {}, **kwargs):
        
        self.nclass = nclass
        self.kwargs = kwargs
        self.method = method
        self.scaler = scaler
        self.C = C
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.exec_time = 0
        self.n_iter = []
        self.final_dual_obj = []
        
        self.label_dicts = -np.ones((nclass, nclass), int)
        np.fill_diagonal(self.label_dicts, 1)
        
        self.model_list = []
        for i in range(nclass):
            self.model_list.append(kSVM(IdentityScaler(), C = self.C, method = self.method, kernel = self.kernel, kernel_params = self.kernel_params, **kwargs))
        
        
    def fit(self, X, y):
        # rescale input data
        X = self.scaler.fit_transform(X)
        
        start = time.time()
        
        for i in range(self.nclass):
            self.model_list[i].fit(X, self.label_dicts[i,y])
        self.exec_time = time.time() - start
        
        self.n_iter = list(map(lambda x: x.n_iter, self.model_list))
        self.final_dual_obj = list(map(lambda x: x.final_dual_obj, self.model_list))
        
    def predict(self, X):
        # rescale input data
        X = self.scaler.transform(X)
        
        distances = np.stack(list(map(lambda model: model.dist_from_hyperplane(X), self.model_list)))
        
        return(np.argmax(distances,axis = 0))
    
    def __call__(self, X):
        # rescale input data
        X = self.scaler.transform(X)
        
        distances = np.stack(list(map(lambda model: model.dist_from_hyperplane(X), self.model_list)))
        
        return(distances)
    
def get_model_multiclass(C, nclass, method, kernel_type, **kwargs):
    # get a standard multiclass kSVM model with input kernel
    return Multiclass_kSVM(StdScaler(), C = C, nclass = nclass, method = method, kernel = kernel_type, kernel_params = kwargs)