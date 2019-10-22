data_folder = "../data/"
training_data = data_folder + 'train.csv'
testing_data = data_folder + 'test.csv'
pos_weight = 1
lambda_rr = 3

methods = ['mse', 'mae', 'least-squares', 'least-squares-GD', 'least-squares-SGD', 'log', 'regularized-log', 'ridge-regression']