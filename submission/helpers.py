# -*- coding: utf-8 -*-

""" Global variables """

data_folder = "../data/"
training_data = data_folder + 'train.csv'
testing_data = data_folder + 'test.csv'
pos_weight = 1
lambda_rr = 0

methods = ['mse', 'mae', 'least-squares', 'least-squares-GD', 'least-squares-SGD', 'log', 'regularized-log', 'ridge-regression', 'log-newton']


"""some helper functions for project 1."""


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
