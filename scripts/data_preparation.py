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

def clean_features(X, X_val, Y, method=''):
    #remove_cols = [0,4,5,6,12,23,24,25,26,27,28]
    #remove_cols = [0,2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29]
    # no preprocessing    
    if (method == 'raw'):
        return X, X_val

    X_clean = np.copy(X)
    X_clean_val = np.copy(X_val)

    if(method!=''):
        X_clean[X_clean==-999] = np.nan
        X_clean_val[X_clean_val==-999] = np.nan
    
    # replace -999s with 0s
    if(method=='0'):
        X_clean = np.nan_to_num(X_clean)
        X_clean_val = np.nan_to_num(X_clean_val)
    else:
        # replace -999s with the column mean of the training set
        if(method=='mean'):    
            replacements = np.nanmean(X_clean, axis=0)
        
        # replace -999s with the column median of the training set
        elif(method=='median'):
            replacements = np.nanmedian(X_clean, axis=0)
        inds = np.where(np.isnan(X_clean))
        inds_val = np.where(np.isnan(X_clean_val))
        X_clean[inds] = np.take(replacements, inds[1])
        X_clean_val[inds_val] = np.take(replacements, inds_val[1])
        
    return X_clean, X_clean_val

def add_bias(x):
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    return tx

def standardize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return mu, sigma

#returns a 1-D array of the values of correlations between features and the output
def feature_corrs(X, Y):
    return np.corrcoef(np.column_stack([Y,X])[:15000], rowvar=False)[0,1:]

#returns a 1-D array of indices of features to be removed
def features_to_remove(X, Y):
    correlations = feature_corrs(X, Y)
    return np.argwhere(np.abs(correlations)>0.15).flatten()

def correlation_heatmap(X):
    correlations = np.corrcoef(X[:15000,:], rowvar=False)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();

def build_poly(x, degree):
    # removes the first column which in our case is just 1s
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    n = x.shape[1]
    columns = []
    for col in range(n):
        temp = np.repeat(x[:,col].reshape(x.shape[0],1).T,degree, axis=0).T
        for i in range(degree):
            temp[:,i] = temp[:,i]**(i)
        columns.append(temp)
    return np.column_stack(columns)
