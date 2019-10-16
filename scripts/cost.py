"""Functions to compute the cost."""

import numpy as np
import math
from global_variables import pos_weight

def least_squares(y, tx):
    """ Linear regression using normal equations """
    w = np.linalg.inv((tx.T @ tx)) @ tx.T @y
    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent """
    method = "mse"
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma, method)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent """
    method = "mse"
    batch_size = 32   # todo: choose and explain suitable batch_size
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
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma, method)
    return w, loss
    
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(x, w):
    temp = x@w
    sigmoid_vec = np.vectorize(sigmoid)
    return sigmoid_vec(temp)

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_loss(y, tx, w, method, lambda_=0):
    predictions = tx@w
    error = y-predictions
    if(method == 'mse'):
        return 1/(2*y.shape[0])*np.sum(error*error)
    elif(method == 'mae'):
        return 1/(y.shape[0])*np.sum(np.abs(error))
    elif(method == 'log'):
        predictions = predict(tx, w)
        return -np.sum(y*np.log(predictions)*pos_weight + (1-y)*np.log(1-predictions))/y.shape[0]
    elif(method == 'regularized-log'
         predictions = predict(tx, w)
         lambdaTerm = lambda_*np.sum(w**2)/2
         return (-np.sum(y*np.log(predictions)*pos_weight + (1-y)*np.log(1-predictions))+ lambdaTerm)/y.shape[0]
    elif(method == 'ridge-regression'):
        return 1/(2*y.shape[0])*np.sum(error*error)+np.linalg.norm(w)**2 
    # Missing lambda in this expression
    
def compute_gradient(y, tx, w, method, lambda_=0):
    if(method=='log') or (method=='regularized-log'):
        prediction = predict(tx, w)
    else:
        prediction = tx@w
    error = y-prediction
    if(method=='mse'):
        gradient = -1/y.shape[0]*tx.T@error
    elif(method == 'mae'):
        gradient = - 1/y.shape[0]* tx.T@np.sign(error)
    elif(method == 'log'):
        gradient = tx.T@error
    elif(method == 'regularized-log'):
        gradient = tx.T@error + lambda_*w
    return gradient

def gradient_descent(y, tx, initial_w, max_iters, gamma, method, lambda_=0):
    i = 0
    initial_w = initial_w/100 # initialize at small weights else gradient descent might explode at first iteration
    w_res = initial_w
    loss_hist = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w, method, lambda_)
        loss = compute_loss(y, tx, w,method, lambda_)
        w = w - gamma * gradient
        # store w and loss
        w_res = w
        loss_hist.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss))
        # Log Progress
        i = i + 1
        if i % 1000 == 0:
            print("iter: " + str(i) + " loss: "+str(loss_hist[-1]))

    return loss_hist, w_res

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
        gradient = compute_gradient(minibatch_y, minibatch_tx, w, method)
        loss = compute_loss(minibatch_y, minibatch_tx, w, method)
        w = w- gamma * gradient
        # store w and loss
        w_res = w
        loss_res = loss
    return w_res, loss_res
