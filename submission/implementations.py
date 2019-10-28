""" Code containing working implementations of ML algorithms """

import numpy as np
import math

# least squares
def least_squares(y, tx):
    """ Linear regression using normal equations """
    a = tx.T@tx
    b = tx.T@y
    # solve normal equations
    w = np.linalg.solve(a,b)
    loss = compute_mse(y, tx, w)
    return w, loss

# lS-GD
def least_squares_GD(y, tx, initial_w, max_iters, gamma, method):
    """ Linear regression using gradient descent """
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma, method)
    return w, loss

# LD-SGD
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma,method):
    """ Linear regression using stochastic gradient descent """
    w, loss = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, method)
    return w, loss

# ridge regression
def ridge_regression(y, tx, lambda_):
    """ Ridge regression """
    a_reg = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + a_reg
    b = tx.T.dot(y)
    # solve normal equations
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w) + lambda_ * w.T @ w
    return w, loss

# log
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD"""
    method = "log"
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma, method)
    return w, loss

# regularized-log
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent or SGD """
    method = "regularized-log"
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma, method, lambda_)
    return w, loss

# other functions like log-newton, 
def MAE_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent with mean absolute error as the cost function"""
    method = "mae"
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma, method)
    return w, loss
    
def logistic_regression_Newton(y, tx, initial_w, max_iters, gamma):
    " Logistic regression using Newton's method "
    method = "log-newton"
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma, method)
    return w, loss
   
def calculate_hessian(y, tx, w):
    """Returns the hessian of the loss function."""
    S = sigmoid(tx@w) * (1-sigmoid(tx@w))
    H = (tx.T*S) @ tx
    return H

def compute_mse(y, tx, w):
    """Computes the loss by mse."""
    mse = 1/(2*len(y)) * np.sum((y-tx@w)**2)
    return mse
    
def sigmoid(t):
    """Applies sigmoid function on t."""
    return (1 / (1 + np.exp(-t)))

def gradient_descent(y, tx, initial_w, max_iters, gamma, method, lambda_ = 0):
    """Returns w and loss calculated using gradient descent with given parameters"""
    w = initial_w
    for n_iter in range(max_iters):
        # perform 1 iteration of gradient descent with the chosen method
        gradient = compute_gradient(y, tx, w, method, lambda_)
        loss = compute_loss(y, tx, w, method, lambda_)
        
        if (method == 'log-newton') : 
            hessian = calculate_hessian(y, tx, w)
            b = hessian @ w - gamma * gradient
            w = np.linalg.solve(hessian,b)
            
        else :
            w = w - gamma * gradient
        
    return w, loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, method):
    """Gradient descent using minibatches"""
    w = initial_w
    # iterates repeatedly over subsets of the examples
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        gradient = compute_gradient(minibatch_y, minibatch_tx, w, method)
        loss = compute_loss(minibatch_y, minibatch_tx, w, method)
        w = w - gamma * gradient
    return w, loss


# """ Grouping all functions into one """"

def ML_methods(y, tx, method, initial_w, batch_size = 1, max_iters = 1, gamma = 0 , lambda_ = 0):
    """ All methods grouped into one function """
    if(method == 'least-squares'):
        w, loss = least_squares(y, tx)
    elif(method == 'least-squares-GD'):
        w, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma, method)
    elif(method == 'least-squares-SGD'):
        w, loss = least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma, method)
    elif(method == 'mae'):
         w, loss = MAE_GD(y, tx, initial_w, batch_size, max_iters, gamma)
    elif(method == 'log'):
        w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)
    elif(method == 'regularized-log'):
        w, loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma) 
    elif(method == 'ridge-regression'):
        w, loss = ridge_regression(y, tx, lambda_)
    elif(method == 'log-newton'):
        w, loss = logistic_regression_Newton(y, tx, initial_w, max_iters, gamma)
    return w, loss


def compute_loss(y, tx, w, method, lambda_=0):
    """Returns the loss for given y, tx, w, method and lambda"""
    # calculate error
    predictions = tx@w
    error = y-predictions
    # depending on method, the returned loss is calculated
    if (method == 'least-squares') or (method == 'least-squares-GD') or (method == 'least-squares-SGD') or (method =='ridge-regression'):
        loss = 1/(2*len(y)) * np.sum(error**2) + lambda_ * w.T @ w
    elif(method == 'mae'):
        loss = 1/(len(y)) * np.sum(np.abs(error))
    elif(method == 'log') or (method == 'regularized-log') or (method == 'log-newton'):
        predictions = sigmoid(tx@w)
        epsilon = 1e-5   # parameter to ensure a real value is returned
        #loss = -1/(len(y)) * (y.T @ np.log(predictions + epsilon) + (1-y).T @ np.log(1-predictions + epsilon)) + lambda_ * w.T @ w
        loss = -1/len(y) * (y.T @ np.log(predictions + epsilon) + (1-y).T @ np.log(1-predictions + epsilon) - lambda_ * w.T @ w)
        
    return loss

def compute_gradient(y, tx, w, method, lambda_=0):
    """Returns the gradient for gradient descent"""
    # calculate error
    if(method=='log') or (method=='regularized-log'):
        prediction = sigmoid(tx@w)
    else:
        prediction = tx@w
    error = y-prediction
    # depending on method, the returned gradient is calculated
    if (method == 'least-squares') or (method == 'least-squares-GD') or (method == 'least-squares-SGD') or (method =='ridge-regression'): 
        gradient = -1/(len(y))*tx.T@error + 2*lambda_*w
    elif(method == 'mae'):
        gradient = - 1/(len(y))* tx.T@np.sign(error)
    elif(method == 'log') or (method == 'regularized-log') or (method == 'log-newton'):
        gradient = 1/len(y) * (tx.T@error + 2*lambda_*w)
    return gradient
    