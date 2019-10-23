import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from global_variables import * 


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def load_data():
    Y = np.genfromtxt(training_data, delimiter=',', dtype=None, skip_header=1, usecols=[1], converters={1: lambda x: 0 if b'b'==x else 1})    
    data = np.genfromtxt(training_data, delimiter=',', skip_header=1)
    X = data[:, 2:]    
    return X, Y

def clean_and_standardize_features(X):
    X_clean = np.delete(X,[0,4,5,6,12,23,24,25,26,27,28], axis=1)
    X_standardized = (X_clean - X_clean.mean(axis=0))/X_clean.std(axis = 0)
    X_standardized = np.insert(X_standardized, 0, 1, axis=1)
    return X_standardized

def standardize(x_tr, x_te):
    mu = np.mean(x_tr, axis=0)
    sigma = np.std(x_tr, axis=0)
    std_x_tr = (x_tr - mu)/sigma
    std_x_te = (x_te - mu) / sigma   
    return std_x_tr, std_x_te

def correlation_heatmap(X):
    correlations = np.corrcoef(X[:15000,:], rowvar=False)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();
    
    
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    random_idx = np.arange(x.shape[0])
    np.random.shuffle(random_idx)
    x = x[random_idx]
    y = y[random_idx]
    val_split = int(ratio*x.shape[0])
    x_tr = x[:val_split]
    y_tr = y[:val_split]
    x_val = x[val_split:]
    y_val = y[val_split:]
    return (x_tr, y_tr, x_val, y_val)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x = np.reshape(x,(-1,1))
    phi = x**0
    for j in range(1, degree + 1):
        phi =  np.concatenate((phi,x**j),axis=1)
    return(phi)
