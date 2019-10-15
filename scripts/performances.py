import numpy as np 

def performance(Y, Y_predicted):
    return np.sum(Y == Y_predicted)/Y.shape[0]

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
    
#set the threshold to 0.618 for mse, 0.458 for cross-enthropy
def label_results(Y_predicted, threshold=0.51):
    f = lambda x: -1 if x<threshold else 1
    f_vec = np.vectorize(f)
    return f_vec(Y_predicted)
    
def find_best_threshold(Y_predicted, Y_labeled):
    best_t = 0.5
    best_perf = performance(Y_labeled, label_results(Y_predicted, threshold=best_t))
    for i in range(0, 100):
        perf = performance(Y_labeled, label_results(Y_predicted, threshold=i/100))>best_perf
        print('threshold is: ' + str(i/100))
        print('performance is: ' + str(perf))
        if(perf>best_perf):
            best_t = i/100
            best_perf = perf
    print('best threshold is: ' + str(best_t))
    print('best performance is: ' + str(best_perf))
    return best_t, best_perf    