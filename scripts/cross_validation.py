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

def cross_validation(y, x, k_indices, k, method, initial_w = None, batch_size =1, max_iters = 1, gamma = 0 , lambda_ = 0):
    if (initial_w is None):
        initial_w = np.ones(x.shape[1]) # see if random init
        
    err_cv_tr = []
    err_cv_te = []
    
    # get k'th subgroup in test, others in train: 
    for k_fold in range(k) : 
        idx_tr, idx_val = k_indices[np.r_[0:k_fold,(k_fold+1):k]].ravel(), k_indices[k_fold]
        x_tr, x_val = x[idx_tr,:], x[idx_val,:]
        y_tr, y_val = y[idx_tr], y[idx_val]
        
        # Normalized data 
        x_tr, x_val = standardize(x_tr, x_val)
        
        # Do the preprocessing, e.g 
    # form data with polynomial degree: 
    #    x_poly_tr = build_poly(x_tr, degree)
    #    x_poly_val = build_poly(x_val, degree)
        
    # Applying method : 
        w_tr, loss_tr = ML_methods(y_tr, x_tr, method, initial_w, batch_size, max_iters, gamma, lambda_)
        loss_te = compute_loss(y_val, x_val, w_tr, method, lambda_)
        
    # calculate the loss for train and test data: 
        err_cv_tr.append(loss_tr)
        err_cv_te.append(loss_te)
    return np.mean(err_cv_tr), np.mean(err_cv_te), np.var(err_cv_tr), np.var(err_cv_te)
