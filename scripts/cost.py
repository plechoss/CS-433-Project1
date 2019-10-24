"""Functions to compute the cost."""

import numpy as np
import math
from global_variables import pos_weight, lambda_rr

def least_squares(y, tx):
    """ Linear regression using normal equations """
    a = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(a,b)
    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent """
    method = "mse"
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma, method)
    return w, loss

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """ Linear regression using stochastic gradient descent """
    method = "mse"
    #batch_size = 32   # todo: choose and explain suitable batch_size
    w, loss = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, method)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """ Ridge regression """
    a_reg = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + a_reg
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD"""
    method = "log"
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma, method)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent or SGD """
    method = "regularized-log"
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma, method, lambda_)
    return w, loss

def ML_methods(y, tx, method, initial_w, batch_size = 1, max_iters = 1, gamma = 0 , lambda_ = 0):
    """ All methods grouped into one function """
    if(method == 'least-squares'):
        w, loss = least_squares(y, tx)
    elif(method == 'least-squares-GD'):
        w, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)
    elif(method == 'least-squares-SGD'):
        w, loss = least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma)
    elif(method == 'mae'):
        raise NotImplementedError  
    elif(method == 'log'):
        w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)
    elif(method == 'regularized-log'):
        w, loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma) 
    elif(method == 'ridge-regression'):
        w, loss = ridge_regression(y, tx, lambda_)
    if(type(loss) is list and len(loss)>1):
        return w, loss[-1]
    return w, loss   # only return final values of w and loss
    
def sigmoid(z):
    if(z < 0):
        return 1 - 1/(1 + math.exp(z))
    else:
        return 1/(1 + math.exp(-z))

def predict(x, w):
    temp = x@w
    ind = temp.argmax()
    sigmoid_vec = np.vectorize(sigmoid)
    return sigmoid_vec(temp)

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_loss(y, tx, w, method, lambda_=0):
    predictions = tx@w
    error = predictions - y
    if(method == 'mse'):
        return 1/(2*y.shape[0])*np.sum(error*error)
    elif(method == 'mae'):
        return 1/(y.shape[0])*np.sum(np.abs(error))
    elif(method == 'log'):
        #loss = np.abs(np.sum(np.log(1+np.exp(predictions)) - y@predictions.T)/y.shape[0]) # formula from course, gives negative losses?
        predictions = predict(tx, w)
        loss = -np.sum(y*np.log(predictions) + (1-y)*np.log(1-predictions))/y.shape[0]
        return loss
    elif(method == 'regularized-log'):
        predictions = predict(tx, w)
        lambdaTerm = lambda_*np.sum(w**2)/2
        return -np.sum(y*np.log(predictions) + (1-y)*np.log(1-predictions))/y.shape[0] + lambdaTerm
    elif(method == 'ridge-regression'):
        return 1/(2*y.shape[0])*np.sum(error*error)+ lambda_*np.linalg.norm(w)**2 
    
def compute_gradient(y, tx, w, method, lambda_=0):
    if(method=='log' or method=='regularized-log'):
        prediction = predict(tx, w)
    else:
        prediction = tx@w
    #import pdb; pdb.set_trace()
    error = prediction - y
    if(method=='mse') or (method =='ridge-regression'): # TODO: rename methods, mse is probably not a good name 
        gradient = 1/y.shape[0]*tx.T@error + 2*lambda_*w
    elif(method == 'mae'):
        gradient =  1/y.shape[0]* tx.T@np.sign(error)
    elif(method == 'log'):
        gradient =  1/y.shape[0]*tx.T@error
    elif(method == 'regularized-log'):
        gradient =  1/y.shape[0]*tx.T@error + lambda_*w
    return gradient

def gradient_descent(y, tx, initial_w, max_iters, gamma, method, lambda_=0):
    initial_w = initial_w/100 # initialize at small weights else gradient descent might explode at first iteration
    w_res = initial_w
    loss_hist = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w, method, lambda_)
        loss = compute_loss(y, tx, w, method, lambda_)
        w = w - gamma * gradient
        # store w and loss
        w_res = w
        # check that cost decreases on every iteration
        #if(i>0):
        #   print("difference to last loss " + str(loss_hist[-1]-loss))
        loss_hist.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss))
    return w_res, loss_hist

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
    w_res = initial_w
    loss_res = 0
    w = initial_w
    ti = max_iters
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        gradient = compute_gradient(minibatch_y, minibatch_tx, w, 'mse')
        loss = compute_loss(minibatch_y, minibatch_tx, w, 'mse')
        w = w - gamma * gradient
        # store w and loss
        w_res = w
        loss_res = loss
    return w_res, loss_res
