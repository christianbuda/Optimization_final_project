import pandas as pd
import numpy as np
import time
from itertools import product
from scipy.optimize import minimize
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

####################### DATA LOADING ########################################################
def NN_data_loading(l1, l2):
    # data loading routine
    
    # load data
    full_data = pd.read_csv('data.txt')

    # select only the 2 input letters
    data = pd.DataFrame(full_data.loc[(full_data['Y'] == l1) | (full_data['Y'] == l2)])

    # replace l1 with 0
    #     and l2 with 1
    data.replace({l1: 0, l2: 1}, inplace = True)

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
    # training error matrix
    avg_err = np.zeros(gridsearch_size)

    # loop over all possible tuple of indices
    for indices in tqdm(product(*map(np.arange, gridsearch_size)), total = np.prod(gridsearch_size)):
        # select current value for each of the parameters
        current_params = map( lambda x, i: x[i] , param_grid.values(),indices )

        # pack values back into a dictionary
        current_params = dict(zip(param_grid.keys(), current_params ))
        
        # train and compute accuracy and error for each split
        model_accuracies = list(map(lambda idx: model_generation_fun(**current_params,  **kwargs).split_train_evaluate(X, y, idx, error = True), idx_split))
        
        model_err = list(map(lambda x:x[1], model_accuracies))
        model_accuracies = list(map(lambda x:x[0], model_accuracies))
        
        # store mean accuracy and error
        avg_acc[indices] = np.mean(model_accuracies)
        avg_err[indices] = np.mean(model_err)
    
    best_params = np.unravel_index(avg_acc.argmax(), avg_acc.shape)
    # select current value for each of the parameters
    best_params = map( lambda x, i: x[i] , param_grid.values(),best_params )
    # pack values back into a dictionary
    best_params = dict(zip(param_grid.keys(), best_params ))
    
    # initialize the final model
    model = model_generation_fun(**best_params,  **kwargs)
    model.fit(X, y)
    
    res = {'best_model' : model, 'best_val_acc':avg_acc.max(), 'bestmodel_val_err' : avg_err.flatten()[avg_acc.argmax()], 'best_params':best_params, 'avg_val_acc':avg_acc, 'avg_val_err' : avg_err}
        
    return(res)

####################### ACTIVATION FUNCTIONS ########################################################
class Activation():
    # basic activation function class
    def __init__(self):
        pass
    
    def __call__(self, X):
        raise NotImplementedError
    
    def derivative(self, X):
        raise NotImplementedError

class Activation_linear(Activation):
    # linear activation function
    def __init__(self):
        pass
        
    def __call__(self, X):
        return X
    
    def derivative(self, X):
        return np.ones(X.shape)
    
class Activation_tanh(Activation):
    # tanh activation function
    def __init__(self, sigma = 1):
        self.sigma = sigma
        
    def __call__(self, X):
        return( np.tanh(self.sigma * X) )
    
    def derivative(self, X):
        return self.sigma * (1 - self(X)**2)

class Activation_sigmoid(Activation):
    # sigmoid activation function
    def __init__(self):
        pass
        
    def __call__(self, X):
        return( 1/(1+np.exp(-X)) )
    
    def derivative(self, X):
        sigmoid = self(X)
        return sigmoid * (1 - sigmoid)

####################### LOSS FUNCTIONS ########################################################
class Loss():
    # basic loss function class
    # to be outputted without regularization and unaggregated
    def __init__(self):
        pass
    
    def __call__(self, y, yhat):
        raise NotImplementedError
    
    def derivative(self, y, yhat):
        raise NotImplementedError
        
class Loss_CE(Loss):
    # unregularized CE loss
    def __init__(self, epsilon = 1e-8):
        self.epsilon = epsilon
        
    def __call__(self, y, p):
        return -( (1-y)*np.log(1-p+self.epsilon) + y*np.log(p+self.epsilon) )
    
    def derivative(self, y, p):
        return (1-y)/(1-p+self.epsilon) - y/(p + self.epsilon)
    
class Loss_LS(Loss):
    # unregularized LS loss
    def __init__(self):
        pass
    
    def __call__(self, y, yhat):
        return 0.5*(yhat-y)**2
    
    def derivative(self, y, yhat):
        return (yhat-y)

####################### RBF KERNELS #######################################################
class RBF_Linear(Activation):
    # linear RBF kernel
    def __init__(self):
        pass
        
    def __call__(self, X):
        return X
    
    def derivative(self, X):
        return np.ones(X.shape)

class RBF_Gaussian(Activation):
    # gaussian RBF kernel
    def __init__(self, sigma = 1):
        self.sigma = sigma
        
    def __call__(self, X):
        return np.exp(-(X/self.sigma)**2)
    
    def derivative(self, X):
        tmp = X/(self.sigma**2)
        return - 2 * tmp * np.exp(-tmp*X)
    
class RBF_Multiquadric(Activation):
    # multiquadric RBF kernel
    def __init__(self, sigma = 1):
        self.sigma = sigma
        
    def __call__(self, X):
        return np.sqrt(X**2 + self.sigma**2)
    
    def derivative(self, X):
        return X / self(X)
    
class RBF_InvMultiquadric(Activation):
    # inverse multiquadric RBF kernel
    def __init__(self, sigma = 1):
        self.sigma = sigma
        
    def __call__(self, X):
        return 1/np.sqrt(X**2 + self.sigma**2)
    
    def derivative(self, X):
        return - X / np.power(X**2 + self.sigma**2, 1.5)

def get_rbf_func(rbf_type, params):
    # function to get the rbf kernel starting from a keyword
    
    available_types = {'linear':RBF_Linear,
                       'gaussian':RBF_Gaussian,
                       'multiquadric':RBF_Multiquadric,
                       'invmultiquadric':RBF_InvMultiquadric}
    
    rbf_type = rbf_type.lower()
    if rbf_type not in available_types.keys():
        raise ValueError("The radial basis function must be one of ['linear', 'gaussian', 'multiquadric', 'invmultiquadric'].")
    
    selected_rbf = available_types[rbf_type]
    return selected_rbf(**params)

####################### LAYERS ############################################################
class Layer():
    # basic layer class
    def __init__(self):
        self.weights = []  # list of weights used in the layers
        self.shapes = []   # shape for each element of the weights list
        self.n_params = [] # number of parameters for each element in the weights list
        
        self.lastunactivated = None  # last memorized output of the layer without the activation
        self.lastoutput = None       # last memorized output of the layer with the activation
    
    def update_memory(self, X, W):
        raise NotImplementedError
    
    def __call__(self, X, W = None):
        if W == None:
            W = self.weights
        
        self.update_memory(X,W)
        
        return self.lastoutput
    
    def call_unactivated(self, X, W = None):
        # call the layer without the activation function
        
        # it is assumed that X is a (b,n,1) tensor
        # where n is the number of features
        # and b is the batch size (can also be 1)
        
        if W == None:
            W = self.weights
        
        self.update_memory(X,W)
        
        return self.lastunactivated
    
    def call_unactivated_memoryless(self, X, W = None):
        raise NotImplementedError
    
    def get_params(self):
        # return parameters list
        return self.weights
    
    def set_params(self, W):
        # sets a new parameters list with same shapes
        if len(W)!=len(self.weights):
            raise ValueError('W has wrong length')
        for i in range(len(W)):
            if self.shapes[i]!=W[i].shape:
                ValueError(f'W[{i}] has wrong shape')
        self.weights = W
    
    def reset_parameters(self, W):
        # resets the layer with completely new weights
        self.weights = W
        self.shapes = list(map(lambda x: x.shape, self.weights))
        self.n_params = list(map(lambda x: np.prod(x), self.shapes))
    
    def flat_set_params(self, W):
        # sets the weights from a flattened array
        if len(W) != np.sum(self.n_params):
            raise ValueError('W has wrong length')
        self.weights = self.unflatten_params(W)
    
    def flat_get_params(self):
        # returns the flattened weights list
        return np.concatenate(list(map(lambda x: x.flatten(), self.weights)))
    
    def unflatten_params(self, W):
        # unflatten the weight array according to the current weight structure
        
        unflattened_W = []
        j = 0
        for idx, n_par in enumerate(self.n_params):
            unflattened_W.append(np.reshape(W[j:j+n_par], newshape = self.shapes[idx]))
            j += n_par
        
        return unflattened_W
    
    def vjp_x(self, X, W):
        raise NotImplementedError
    
    def vjp_W(self, X, W):
        raise NotImplementedError

class Dense(Layer):
    # simple neural network dense layer
    def __init__(self, W, activation = Activation_linear(), include_bias = True):
        super().__init__()
        self.activation = activation
        self.include_bias = include_bias
        
        self.reset_parameters(W)
        
        # check correctness of parameters list
        assert len(W) == 1+include_bias
    
    def update_memory(self, X, W):
        # update the memorized output values
        
        # it is assumed that X is a (b,n,1) tensor
        # where n is the number of features
        # and b is the batch size (can also be 1)
        
        # we perform the matrix multiplication for each column vector
        result = np.matmul(W[0],X)
        
        # and we add the bias if needed
        if self.include_bias:
            result = result + W[1]
        
        # update memory
        self.lastunactivated = result
        self.lastoutput = self.activation(result)
    
    def call_unactivated_memoryless(self, X, W = None):
        # calls the layer without the activation and without updating the memorized values
        
        # it is assumed that X is a (b,n,1) tensor
        # where n is the number of features
        # and b is the batch size (can also be 1)
        if W is None:
            W = self.weights
            
        # we perform the matrix multiplication for each column vector
        result = np.matmul(W[0],X)
        
        # and we add the bias if needed
        if self.include_bias:
            result = result + W[1]
        
        return(result)
    
    def vjp_x(self, v, X, W = None):
        # vector-jacobian product wrt x
        
        if W is None:
            W = self.weights
        
        v = self.activation.derivative(self.call_unactivated_memoryless(X, W)) * v
        result = np.matmul(W[0].T, v)
        return result
    
    def vjp_W(self, v, X, W = None):
        # vector-jacobian product wrt the weights
        
        v = self.activation.derivative(self.call_unactivated_memoryless(X, W)) * v
        result = [np.matmul(X, np.moveaxis(v, -1,-2))]
        
        if self.include_bias:
            result.append(v)
        return(result)

class RBF(Layer):
    # rbf layer object
    def __init__(self, W, activation = Activation_linear(), rbf_fn = RBF_Linear(), epsilon = 1e-8):
        super().__init__()
        
        self.epsilon = epsilon   # parameter to avoid division by zero in the derivative
        self.activation = activation
        self.rbf_fn = rbf_fn     # rbf kernel
        
        self.reset_parameters(W)
        
        # we only need two sets of parameters
        # the weights and the centers
        assert len(W) == 2
        
        # we want both the weights and the centers to be matrices
        assert len(W[0].shape) == 2
        assert len(W[1].shape) == 2
        
        # the weights must be column vectors
        assert W[0].shape[1] == 1
        
        # initialize memory for the norm too
        self.lastnorm = None
    
    def call_norm(self, X, W = None):
        # calls the layer and returns the computed norms
        
        # it is assumed that X is a (b,n,1) tensor
        # where n is the number of features
        # and b is the batch size (can also be 1)
        if W == None:
            W = self.weights
        
        self.update_memory(X,W)
        
        return self.lastnorm
    
    def update_memory(self, X, W):
        # update the memorized output values
        
        # it is assumed that X is a (b,n,1) tensor
        # where n is the number of features
        # and b is the batch size (can also be 1)
        
        # compute the norms of the differences between input data and the centers
        norms = np.linalg.norm(X - W[1].T[np.newaxis,:,:], axis = 1)
        
        # apply the radial basis function and aggregate
        rbf_out = np.sum(W[0].T * self.rbf_fn(norms), axis = 1)
        
        # update memory
        self.lastnorm = norms
        self.lastunactivated = rbf_out
        self.lastoutput = self.activation(rbf_out)
    
    def grad_w(self, X, W = None):
        # gradient wrt the weights
        
        if W is None:
            W = self.weights
            
        # compute intermediate values of the function
        self.update_memory(X,W)
        
        # gradient of unactivated rbf
        rbf_gradw = self.rbf_fn(self.lastnorm)
        # derivative of the activation multiplied by the rbf
        result = self.activation.derivative(self.lastunactivated)[:,np.newaxis] * rbf_gradw
        
        return result
    
    def grad_c(self, X, W = None):
        # gradient wrt the centers
        
        if W is None:
            W = self.weights
        
        # compute intermediate values of the function
        self.update_memory(X,W)
        
        # gradient of unactivated rbf
        rbf_gradc = np.moveaxis( (W[1].T[np.newaxis,:,:] - X) , -1,-2)/(self.lastnorm[:,:,np.newaxis] + self.epsilon)
        rbf_gradc = self.rbf_fn.derivative(self.lastnorm)[:,:,np.newaxis] * rbf_gradc
        rbf_gradc = rbf_gradc * W[0][np.newaxis,:,:]
        # derivative of the activation multiplied by the rbf
        result = self.activation.derivative(self.lastunactivated)[:,np.newaxis, np.newaxis] * rbf_gradc
        return result
    
####################### THE MODELS ########################################################
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
    
class NN(Model):
    # simple feedforward neural network model
    def __init__(self, input_dim, H, N, scaler = IdentityScaler(), loss_fn = Loss_CE(1e-8), sigma = 1):
        self.rho = 1e-4
        self.optimization_solver = 'L-BFGS-B'
        self.scaler = scaler
        self.loss_fn = loss_fn
        self.input_dim = input_dim
        self.H = H
        self.N = N
        self.sigma = sigma
        self.exec_time = 0
        self.n_iter = 0
        self.nfev = 0
        self.njev = 0
        self.opt_message = None
        self.layers_list = []
        self.initial_error = 0
        self.initial_accuracy = 0
        self.starting_obj = 0

        # first hidden layer
        self.layers_list.append(Dense(W = [np.random.normal(0,0.01,(self.N, self.input_dim)), np.zeros((self.N,1))], activation = Activation_tanh(sigma = self.sigma)))
        
        # the other hidden layers
        for i in range(self.H - 1):
            self.layers_list.append(Dense(W = [np.random.normal(0,0.01,(self.N, self.N)), np.zeros((self.N,1))], activation = Activation_tanh(sigma = self.sigma)))
        
        # last classification layer
        self.layers_list.append(Dense(W = [np.random.normal(0,0.01,(1,self.N)), np.zeros((1,1))], activation = Activation_sigmoid()))

        # number of parameters for each layer
        self.n_params = list(map(lambda x: sum(x.n_params), self.layers_list))

    def reformat_X(self, X):
        # reshape X into a matrix if it is a single row element
        if len(X.shape) == 1:
            X = X[np.newaxis,:]
        
        X = self.scaler.transform(X)
        
        # reshape X into a tensor with batching dimension as first dimension
        # e.g. a data matrix 5x3 with 5 elements of 3 feature each
        # is reshaped into a tensor 5x3x1 where we have five 3x1 column vectors along the first dimension
        return(X[:,:,np.newaxis])
    
    def get_weights_list(self):
        # returns the list of the weights in each layer
        out = []
        for layer in self.layers_list:
            out.append(layer.get_params())
        
        return(out)
    
    def get_weight_array(self):
        # returns the weights in each layer as a big flat array
        out = []
        for layer in self.layers_list:
            out.append(layer.flat_get_params())
        return(np.concatenate(out))
    
    def set_weights_list(self, W):
        # sets the weights in each layer
        
        for i in range(len(W)):
            self.layers_list[i].weights = W[i]
            
    def unflatten_params(self, W):
        # unflatten the weight array according to the current architecture
        
        unflattened_W = []
        
        j = 0
        for idx, n_par in enumerate(self.n_params):
            unflattened_W.append(self.layers_list[idx].unflatten_params(W[j:j+n_par]))
            j += n_par
        
        return unflattened_W
    
    def set_weight_array(self, W):
        # sets the weights starting from a flat array
        self.set_weights_list(self.unflatten_params(W))
    
    def __call__(self,X):
        # simple call to the model
        
        X = self.reformat_X(X)
        
        # call with builtin parameters
        out = X
        for layer in self.layers_list:
            out = layer(out)
        
        return(out[:,0,0])
    
    def internal_call(self, X):
        # simple call to the model
        # for internal use
        # here X must be properly formatted!!
        
        # call with builtin parameters
        out = X
        for layer in self.layers_list:
            out = layer(out)
        
        return(out[:,0,0])
    
    def loss(self, X, y, regularized = True):
        if regularized:
            # loss with regularization
            return( np.mean(self.loss_fn(y, self.internal_call(X))) + self.rho* np.linalg.norm(self.get_weight_array())**2 )
        else:
            # loss without regularization
            return( np.mean(self.loss_fn(y, self.internal_call(X))) )
    
    def compute_loss(self, X, y, regularized = True):
        X = self.reformat_X(X)
        if regularized:
            # loss with regularization
            return( np.mean(self.loss_fn(y, self.internal_call(X))) + self.rho* np.linalg.norm(self.get_weight_array())**2 )
        else:
            # loss without regularization
            return( np.mean(self.loss_fn(y, self.internal_call(X))) )
        
    def grad_loss(self, X, y, W):
        # gradient of the loss wrt the weight array
        
        flatW = W
        W = self.unflatten_params(W) # unflattened version of the parameters
        
        # store all the intermediate computations in the network
        intermediate_computations = []
        out = X
        for i, layer in enumerate(self.layers_list):
            intermediate_computations.append(out)
            out = layer(out, W[i])
        intermediate_computations.append(out)
        
        # derivative of the loss wrt its argument
        loss_derivative_term = self.loss_fn.derivative(y,intermediate_computations[-1][:,0,0])[:,np.newaxis,np.newaxis]
        
        # initialize the gradient with ones
        grad = []
        for i in range(len(self.layers_list)):
            grad.append([np.ones((1,1))])
        
        # compute all gradients using chain rule
        # starting from the last layer and going up
        for i in range(len(grad)-1, 0, -1):
            tmp = self.layers_list[i].vjp_x(grad[i], intermediate_computations[i], W[i])
            grad[i] = np.concatenate(list(map(lambda x: np.mean(x*loss_derivative_term, axis = 0).T.flatten(), 
                                              self.layers_list[i].vjp_W(grad[i],intermediate_computations[i], W[i]))))
            for j in range(i):
                grad[j] = tmp
                
        grad[0] = np.concatenate(list(map(lambda x: np.mean(x*loss_derivative_term, axis = 0).T.flatten(),
                                          self.layers_list[0].vjp_W(grad[0],intermediate_computations[0], W[0]))))
        
        grad = np.concatenate(grad)
        
        # add the regularization term in the derivative
        grad = grad+2*self.rho*flatW
        
        return(grad)
    
    def fit(self, X, y):
        # fit model to data
        
        # fit rescaling to input data
        self.scaler.fit(X)
        
        # compute starting accuracy and loss
        self.initial_accuracy = self.evaluate(X, y)
        self.initial_error = self.compute_loss(X,y, regularized = False)
        self.starting_obj = self.compute_loss(X,y, regularized = True)
        
        # rescale and reshape data
        X = self.reformat_X(X)
        
        
        def obj(W):
            # loss as a function of the weight list
            self.set_weight_array(W)
            return(self.loss(X,y))
        
        def grad_obj(W):
            # loss gradient as a function of the weight array
            return(self.grad_loss(X, y, W))
        
        start = time.time()
        res = minimize(obj,self.get_weight_array(),method=self.optimization_solver,jac=grad_obj,tol=1e-4)
        self.exec_time = time.time() - start
        
        if not res.success:
                raise RuntimeError('Error in the optimization with message:\n'+res.message)
        
        # set the optimal weights
        self.set_weight_array(res.x)
        self.n_iter = res.nit
        self.nfev = res.nfev
        self.njev = res.njev
        self.opt_message = res.message
    
    def predict(self, X):
        out = self(X) >= 0.5
        return(out.astype('int'))
    
    def train_evaluate(self, X_train, y_train, X_test, y_test, error = False):
        self.fit(X_train, y_train)
        if error:
            return(accuracy(self.predict(X_test), y_test), self.compute_loss(X_train, y_train))
        return(accuracy(self.predict(X_test), y_test))

    def split_train_evaluate(self, X, y, split_idx, error = False):
        # split dataset
        X_train = X[split_idx,]
        y_train = y[split_idx,]
        X_test  = np.delete(X, split_idx, axis = 0)
        y_test  = np.delete(y, split_idx, axis = 0)
        
        return( self.train_evaluate(X_train, y_train, X_test, y_test, error = error) )

class RBFNet(Model):
    # rbf network model
    def __init__(self, input_dim, N, scaler = IdentityScaler(), rbf_type = 'linear', rbf_params = {}, W0 = None, loss_fn = Loss_CE(), epsilon = 1e-8):
        # for numerical stability in the division
        self.epsilon = epsilon
        
        self.rho = 1e-4
        self.optimization_solver = 'L-BFGS-B'
        self.initial_error = 0
        self.initial_accuracy = 0
        self.starting_obj = 0
        
        self.scaler = scaler
        self.loss_fn = loss_fn
        self.input_dim = input_dim
        self.N = N
        
        self.rbf_fn = get_rbf_func(rbf_type, rbf_params)
        
        self.exec_time = 0
        self.n_iter = 0
        self.nfev = 0
        self.njev_w = 0
        self.njev_c = 0
        
        if W0 is None:
            # if the initial weights are not provided
            # generate the randomly
            w = np.random.normal(0,0.01,(self.N,1))
            c = np.random.normal(0,1,(self.N, self.input_dim))
            W0 = [w,c]
        
        # the hidden layer
        self.rbf_layer = RBF(W = W0, activation = Activation_sigmoid(), rbf_fn = self.rbf_fn, epsilon = self.epsilon)
        
        # total number of parameters
        self.n_params = sum(self.rbf_layer.n_params)

    def reformat_X(self, X):
        # reshape X into a matrix if it is a single row element
        if len(X.shape) == 1:
            X = X[np.newaxis,:]
        
        X = self.scaler.transform(X)
        
        # reshape X into a tensor with batching dimension as first dimension
        # e.g. a data matrix 5x3 with 5 elements of 3 feature each
        # is reshaped into a tensor 5x3x1 where we have five 3x1 column vectors along the first dimension
        return(X[:,:,np.newaxis])
    
    def get_weights_list(self):
        # returns the list of the weights in each layer
        return(self.rbf_layer.get_params())
    
    def get_weight_array(self):
        # returns the weights in each layer as a big flat array
        return(self.rbf_layer.flat_get_params())
    
    def set_weights_list(self, W):
        # sets the weights in each layer
        self.rbf_layer.weights = W[i]
            
    def unflatten_params(self, W):
        # unflatten the weight array according to the current architecture
        return self.rbf_layer.unflatten_params(W)
    
    def set_weight_array(self, W):
        # sets the weights starting from a flat array
        self.set_weights_list(self.unflatten_params(W))
    
    def __call__(self, X, W = None):
        # simple call to the model
        X = self.reformat_X(X)
        
        # call with builtin parameters
        out = self.rbf_layer(X, W)
        return(out)
    
    def internal_call(self, X, W = None):
        # simple call to the model
        # for internal use
        # here X must be properly formatted!!
        
        # call with builtin parameters
        out = self.rbf_layer(X, W)
        return(out)
    
    def loss(self, X, y, W = None, regularized = True):
        if W == None:
            W = self.rbf_layer.weights
        
        if regularized:
            # loss with regularization
            return( np.mean(self.loss_fn(y, self.internal_call(X, W) )) + self.rho* (np.linalg.norm(W[0])**2 + np.linalg.norm(W[1])**2 ) )
        else:
            # loss without regularization
            return( np.mean(self.loss_fn(y, self.internal_call(X, W) )) )
        
    def compute_loss(self, X, y, regularized = True):
        X = self.reformat_X(X)
        if regularized:
            # loss with regularization
            return( np.mean(self.loss_fn(y, self.internal_call(X))) + self.rho* np.linalg.norm(self.get_weight_array())**2 )
        else:
            # loss without regularization
            return( np.mean(self.loss_fn(y, self.internal_call(X))) )
    
    def loss_grad_w(self, X, y, W = None):
        # gradient wrt w of the regularized loss
        # computed using the chain rule
        
        if W == None:
            w = self.rbf_layer.weights[0]
        else:
            w = W[0]
            
        gradw = self.rbf_layer.grad_w(X, W)
        gradw = self.loss_fn.derivative(y, self.rbf_layer.lastoutput)[:,np.newaxis] * gradw
        gradw = np.mean(gradw, axis = 0)
        gradw = gradw[:,np.newaxis] + 2 * self.rho * w
        
        return(gradw)
        
    def loss_grad_c(self, X, y, W = None):
        # gradient wrt c of the regularized loss
        # computed using the chain rule
        
        if W == None:
            c = self.rbf_layer.weights[1]
        else:
            c = W[1]
            
        gradc = self.rbf_layer.grad_c(X, W)
        gradc = self.loss_fn.derivative(y, self.rbf_layer.lastoutput)[:,np.newaxis, np.newaxis] * gradc
        gradc = np.mean(gradc, axis = 0)
        gradc = gradc + 2 * self.rho * c
        
        return(gradc)

    def fit(self, X, y, reinitialize = True, maxiter = 1000, tol = 1e-3):
        # fit the model to the data
        
        if reinitialize:
            # new initialization dataset-based
            # seems to work well to speed up training
            c = np.random.choice(np.arange(X.shape[0]), size = self.N, replace = False)
            c = X[c,:]
            self.rbf_layer.weights[1] = c
        
        # fit rescaling to input data
        self.scaler.fit(X)
        
        # compute starting accuracy and loss
        self.initial_accuracy = self.evaluate(X, y)
        self.initial_error = self.compute_loss(X,y, regularized = False)
        self.starting_obj = self.compute_loss(X,y, regularized = True)
        
        # rescale and reshape data
        X = self.reformat_X(X)
        
        def obj_w(w):
            # loss as a function of the weights
            # only accepts w as a flat array
            w = w[:,np.newaxis]
            return(self.loss(X,y, [w, self.rbf_layer.weights[1]]))

        def grad_obj_w(w):
            # loss gradient wrt the weights
            # only accepts w as a flat array
            w = w[:,np.newaxis]
            return(np.array(self.loss_grad_w(X, y, W = [w, self.rbf_layer.weights[1]] )[:,0]))

        def obj_c(c):
            # loss as a function of the centers
            # only accepts c correctly shaped as a matrix
            return(self.loss(X,y, [self.rbf_layer.weights[0], c]))

        def grad_obj_c(c):
            # loss gradient wrt the centers
            # only accepts c correctly shaped as a matrix
            return( self.loss_grad_c(X, y, W = [self.rbf_layer.weights[0], c] ) )
        
        def gradient_norm():
            # current norm of the gradient
            norm_w = np.linalg.norm(self.loss_grad_w(X, y))
            norm_c = np.linalg.norm(self.loss_grad_c(X, y))
            return np.sqrt(norm_w**2 + norm_c**2)
        
        
        start = time.time()
        k = 0
        while  gradient_norm() >= tol:  # stopping condition
            
            ## STEP 1
            # find optimal weights
            res = minimize(obj_w, np.array(self.rbf_layer.weights[0][:,0]), method=self.optimization_solver, jac = grad_obj_w, tol=1e-4)
            
            if not res.success:
                raise RuntimeError('Error in step 1 of optimization with message:\n'+res.message)
                
            # set new weights
            self.rbf_layer.weights[0] = res.x[:,np.newaxis]
            
            # update optimization info
            self.nfev += res.nfev
            self.njev_w += res.nfev

            ## STEP 2
            # armijo line search as in https://www.epfl.ch/labs/anchp/wp-content/uploads/2018/05/part3-1.pdf
            # parameters as the suggested ones
            beta = 0.5
            c1 = 1e-4
            
            # starting values of loss and centers
            starting_loss = res.fun
            starting_c = self.rbf_layer.weights[1]
            
            # compute the descent direction
            current_grad = grad_obj_c(starting_c)
            descent_dir = -current_grad
            descent_dir = descent_dir/np.linalg.norm(descent_dir)
            
            # armijo line search
            j = 0
            alpha = beta
            while ( (obj_c(starting_c + alpha * descent_dir) - starting_loss) > - c1 * alpha * np.linalg.norm(current_grad) ):
                j += 1
                alpha = alpha * beta
                self.nfev += 1
            self.njev_c += 1
            
            # update the centers with a GD step
            self.rbf_layer.weights[1] = starting_c + alpha * descent_dir
            
            ## INCREASE ITER COUNT
            self.n_iter += 1
            k += 1
            if k > maxiter:
                break
            ## END OF OPTIMIZATION STEP
            
        self.exec_time = time.time() - start
        
        if k > maxiter:
            print('Maximum number of iteration reached, you can try running fit() again or maybe try with a bigger tolerance.')
        
    def predict(self, X):
        out = self(X) >= 0.5
        return(out.astype('int'))
    
    def train_evaluate(self, X_train, y_train, X_test, y_test, error = False):
        self.fit(X_train, y_train)
        if error:
            return(accuracy(self.predict(X_test), y_test), self.compute_loss(X_train, y_train))
        return(accuracy(self.predict(X_test), y_test))

    def split_train_evaluate(self, X, y, split_idx, error = False):
        # split dataset
        X_train = X[split_idx,]
        y_train = y[split_idx,]
        X_test  = np.delete(X, split_idx, axis = 0)
        y_test  = np.delete(y, split_idx, axis = 0)
        
        return( self.train_evaluate(X_train, y_train, X_test, y_test, error = error) )
                                

def get_model_MLP(input_dim, H, N, sigma):
    # get a standard NN model with input parameters
    return NN(input_dim = input_dim, H = H, N = N, scaler = StdScaler(), loss_fn = Loss_CE(1e-8), sigma = sigma)

def get_model_RBF(input_dim, N, rbf_type, **kwargs):
    # get a standard RBF model with input parameters
    return RBFNet(input_dim = input_dim, N = N, scaler = StdScaler(), rbf_type = 'gaussian', rbf_params = kwargs, loss_fn = Loss_CE(1e-8))
