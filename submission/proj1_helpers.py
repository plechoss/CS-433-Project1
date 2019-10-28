# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np
from data_preparation import *

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def predict_with_divided_W(Xs, Ws, X):
    """Returns prediction values for input divided by the PRI_jet_num column values"""
    temp = np.copy(Xs)
    temp = preprocX(temp, X)
    res = np.zeros(Xs.shape[0])
    for i in range(Xs.shape[0]):
        res[i] = temp[i]@(Ws[Xs[i,22]].T)
    return res

