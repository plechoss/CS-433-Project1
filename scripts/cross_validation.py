""" Cross-validation """
from cost import *
from data_preparation import *
from performances import *

def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, method, batch_size =1, max_iters = 1, gamma = 0 , lambda_ = 0, clean_method=''):
    """ Perform cross validation on selected ML technique. """
    err_cv_tr = []
    err_cv_te = []
    accuracy = []
    
    # get k'th subgroup in test, others in train
    for k_fold in range(k) : 
        idx_tr, idx_val = k_indices[np.r_[0:k_fold,(k_fold+1):k]].ravel(), k_indices[k_fold]
        x_tr, x_val = x[idx_tr,:], x[idx_val,:]
        y_tr, y_val = y[idx_tr], y[idx_val]
        
        # Preprocessing
        # Handling of corrupted data according to clean_method
        x_tr_clean, x_val_clean = clean_features(x_tr, x_val, clean_method)
        
        # Standardize data
        mu, sigma = standardize(x_tr_clean)
        x_tr_std = (x_tr_clean-mu)/sigma
        x_val_std = (x_val_clean-mu)/sigma
        
        # Remove features that don't correlate with result
        #toRemove = features_to_remove(x_tr_std, y_tr)
        #x_tr_rem = np.delete(x_tr_std, toRemove, 1)
        #x_val_rem = np.delete(x_val_std, toRemove, 1)
        
        # Add bias term 
        x_tr_sel = add_bias(x_tr_std)
        x_val_sel = add_bias(x_val_std)
        
        # Initiate weights at random values
        initial_w = np.random.rand(x_tr_sel.shape[1])
        
        # Test ML algorithm
        w_tr, loss_tr = ML_methods(y_tr, x_tr_sel, method, initial_w, batch_size, max_iters, gamma, lambda_)
        loss_te = compute_loss(y_val, x_val_sel, w_tr, method, lambda_)
        
        # Calculate the loss for train and test data
        err_cv_tr.append(loss_tr)
        err_cv_te.append(loss_te)
        
        # Evaluate the accuracy for test data
        if (method == 'log') or (method == 'log-newton') or (method == 'regularized-log'):
            y_pred = sigmoid(x_val_sel@w_tr)
        else:
            y_pred = x_val_sel@w_tr
        threshold, accuracy = performance(y_val, y_pred)

    return err_cv_tr, err_cv_te, accuracy 
