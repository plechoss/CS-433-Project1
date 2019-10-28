""" Complete code generating best submission to AIcrowd challenge """
# Useful starting lines
import numpy as np
import csv

from global_variables import *
from data_preparation import * 
from cross_validation import *
from performances import * 
from proj1_helpers import *
from implementations import *

X, Y = load_data()
X_1, X_2 = clean_features(X, X, method)

jet_values = np.unique(X[:,22])
masks = {}

test_data = np.genfromtxt(testing_data, delimiter=',', skip_header=1)
test_X = test_data[:, 2:]
test_ids = range(350000,918238)

method = 'median' 
X_divided = {}
Y_divided = {}
W = {}
X_1, X_test = clean_features(X, test_X, method)
    
for value in jet_values:
    masks[value] = (X_1[:,22]==value)
    X_temp = preprocX(X_1[masks[value]], X_1)
    X_divided[value] = X_temp
    Y_temp = Y[masks[value]]
    Y_divided[value] = Y_temp
    W[value], loss = least_squares(Y_temp, X_temp)

pred_div = predict_with_divided_W(X_1, W, X_1)
perf = performance(Y,pred_div)
    
predictions = predict_with_divided_W(X_test, W, X_1)

test_predictions = label_results(predictions)
test_results = np.column_stack([test_ids, test_predictions])
np.savetxt('submission.csv', test_results, fmt="%d", delimiter=",", header="Id,Prediction", comments='')