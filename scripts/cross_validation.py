""" Cross-validation """
from cost import *
from data_preparation import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, method, initial_w = None, batch_size =1, max_iters = 1, gamma = 0 , lambda_ = 0, clean_method=''):
    err_cv_tr = []
    err_cv_te = []
    acc_cv_tr = []
    acc_cv_te = []
    
    # get k'th subgroup in test, others in train: 
    for k_fold in range(k) : 
        idx_tr, idx_val = k_indices[np.r_[0:k_fold,(k_fold+1):k]].ravel(), k_indices[k_fold]
        x_tr, x_val = x[idx_tr,:], x[idx_val,:]
        y_tr, y_val = y[idx_tr], y[idx_val]
        
        # Normalize data with respect to training set parameters
        #x_tr, x_val = standardize(x_tr, x_val)
        
        # Preprocessing
        # Handling of corrupted data according to clean_method
        x_tr_clean, x_val_clean = clean_features(x_tr, x_val, y_tr, clean_method)
        
        # Standardize data
        mu, sigma = standardize(x_tr_clean)
        x_tr_std = (x_tr_clean-mu)/sigma
        x_val_std = (x_val_clean-mu)/sigma
        
        # Select important features
        toRemove = features_to_remove(x_tr_std, y_tr)
        x_tr_sel = np.delete(x_tr_std, toRemove, 1)
        x_val_sel = np.delete(x_val_std, toRemove, 1)
        
        # todo: other pre-processing steps
        # Add bias term 
        x_tr_sel = add_bias(x_tr_sel)
        x_val_sel = add_bias(x_val_sel)
        
        initial_w = np.ones(x_tr_sel.shape[1]) # see if random init
        
        # Test ML algorithm
        w_tr, loss_tr = ML_methods(y_tr, x_tr_sel, method, initial_w, batch_size, max_iters, gamma, lambda_)
        loss_te = compute_loss(y_val, x_val_sel, w_tr, method, lambda_)
        
        # calculate the loss for train and test data: 
        err_cv_tr.append(loss_tr)
        err_cv_te.append(loss_te)
        
        # evaluate the accuracy for train and test data:
        # TO DO
        
    return err_cv_tr, err_cv_te 
