"""Functions to compute the cost."""

import numpy as np
import math
from global_variables import pos_weight

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.inv((tx.T @ tx)) @ tx.T @y
    mse = 1/(2*len(y)) * np.sum((y-tx@w)**2)
    return mse, w

"""Ridge regression """
def ridge_regression(y, tx, lambda_):
    a_reg = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + a_reg
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

""" Logistic regression """
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(x, w):
    temp = x@w
    sigmoid_vec = np.vectorize(sigmoid)
    return sigmoid_vec(temp)

def compute_loss(y, tx, w,method):
    predictions = tx@w
    error = y-predictions
    if(method == 'mse'):
        return 1/(2*y.shape[0])*np.sum(error*error)
    elif(method == 'mae'):
        return 1/(y.shape[0])*np.sum(np.abs(error))
    elif(method == 'cross-enthropy'):
        predictions = predict(tx, w)
        return -np.sum(y*np.log(predictions)*pos_weight + (1-y)*np.log(1-predictions))/y.shape[0]
    elif(method == 'ridge-regression'):
        return 1/(2*y.shape[0])*np.sum(error*error)+np.linalg.norm(w)**2 
    # Missing lambda in this expression
    
def compute_gradient(y, tx, w, method):
    if(method=='cross-enthropy'):
        prediction = predict(tx, w)
    else:
        prediction = tx@w
    error = y-prediction
    if(method=='mse'):
        gradient = -1/y.shape[0]*tx.T@error
    elif(method == 'mae'):
        gradient = - 1/y.shape[0]* tx.T@np.sign(error)
    elif(method == 'cross-enthropy'):
        gradient = tx.T@error
    return gradient

def gradient_descent(y, tx, initial_w, max_iters, gamma, method):
    i = 0
    initial_w = initial_w/100 # initialize at small weights else gradient descent might explode at first iteration
    w_res = initial_w
    loss_hist = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w, method)
        loss = compute_loss(y, tx, w,method)
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
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32, max_iters):
        gradient = compute_gradient(minibatch_y, minibatch_tx, w, method)
        loss = compute_loss(minibatch_y, minibatch_tx, w, method)
        w = w- gamma * gradient
        # store w and loss
        w_res = w
        loss_res = loss
    return loss_res, w_res
