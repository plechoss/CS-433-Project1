""" Cross-validation """
""" Only implemented for Ridge regression for now """
from cost import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_,method = 'ridge'):
    """ Only implemented for ridge for now """
    """return the loss of ridge regression."""
    rmse_cv_tr = []
    rmse_cv_te = []
    
    # get k'th subgroup in test, others in train: 
    for k_fold in range(k) : 
        idx_tr, idx_val = k_indices[np.r_[0:k_fold,(k_fold+1):k]], k_indices[k_fold]
        x_tr, x_val = x[idx_tr].ravel(), x[idx_val].ravel()
        y_tr, y_val = y[idx_tr].ravel(), y[idx_val].ravel()   
       
    # form data with polynomial degree: 
        x_poly_tr = build_poly(x_tr, degree)
        x_poly_val = build_poly(x_val, degree)
        
    # ridge regression:    
        loss_tr, w_tr = ridge_regression(y_tr, x_poly_tr, lambda_)
        loss_te = 1/(2*len(y_val)) * np.sum((y_val-x_poly_val@w_tr)**2)
        
    # calculate the loss for train and test data: 
        rmse_cv_tr.append(np.sqrt(2*loss_tr))
        rmse_cv_te.append(np.sqrt(2*loss_te))
    return np.mean(rmse_cv_tr), np.mean(rmse_cv_te), np.var(rmse_cv_tr), np.var(rmse_cv_te)