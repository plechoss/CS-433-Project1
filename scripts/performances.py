import numpy as np 

#takes as an argument pure, unlabeled Y and Y_predicted
def performance(Y, Y_predictions):
    best_t = 0
    best_perf = 0
    for i in range(0, 100):
        perf = np.sum(label_results(Y) == label_results(Y_predicted, threshold=i/100))/Y.shape[0]
        if(perf>best_perf):
            best_t = i/100
            best_perf = perf
    return best_t, best_perf    

def evaluate_performance(Y, Y_predicted):
    false_negatives = 0
    true_negatives = 0
    false_positives = 0
    true_positives = 0
    for i in range(Y.shape[0]):
        if(Y[i]==-1):
            if(Y_predicted[i]==-1):
                true_negatives += 1
            else:
                false_positives += 1
        else:
            if(Y_predicted[i]==-1):
                false_negatives += 1
            else:
                true_positives += 1
    #print('True positives: ' + str(true_positives))
    #print('False positives: ' + str(false_positives))
    #print('True negatives: ' + str(true_negatives))
    #print('False negatives: ' + str(false_negatives))
    
def label_results(Y_predicted, threshold=0.5):
    f = lambda x: -1 if x<threshold else 1
    f_vec = np.vectorize(f)
    return f_vec(Y_predicted) 

def test_polynomial_performance(X, Y, max_degree=15, method='least_squares'):
    # tests the performance of different degree polynomials, using least_squares
    for i in range(max_degree+1):
        X_test = build_poly(X, i+1)
        N = X_test.shape[1]
        
        w_test, loss_test = ML_methods(Y, X_test, method, max_iters=500,gamma=0.1)
        
        prediction = label_results(predict(X_test, w_test))
        perf = performance(prediction, label_results(Y))
        
        print("Performance for X of polynomial of degree " + str(i) + " is " + str(perf))