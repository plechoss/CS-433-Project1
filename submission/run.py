""" Complete code generating best submission to AIcrowd challenge """
# Useful starting lines
import numpy as np
import csv

from global_variables import *
from data_preparation import * 
from cost import * 
from cross_validation import *
from performances import * 
from proj1_helpers import * 

X, Y = load_data()

jet_values = np.unique(X[:,22])
masks = {}

X_divided = {}
Y_divided = {}
W = {}

for value in jet_values:
    masks[value] = (X[:,22]==value)
    X_temp = preprocX(X[masks[value]])
    X_divided[value] = X_temp
    Y_temp = Y[masks[value]]
    Y_divided[value] = Y_temp
    W[value], loss = least_squares(Y_temp, X_temp)

pred_div = predict_with_divided_W(X, W)

test_data = np.genfromtxt(testing_data, delimiter=',', skip_header=1)
test_X = test_data[:, 2:]
test_ids = range(350000,918238)

predictions = predict_with_divided_W(test_X, W)

test_predictions = label_results(predictions)
test_results = np.column_stack([test_ids, test_predictions])

np.savetxt('submission.csv', test_results, fmt="%d", delimiter=",", header="Id,Prediction", comments='')