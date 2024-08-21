import numpy as np
import scipy
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import sys
base_dir = '/home2/ebrahim/neural_seq_decoder/' 
sys.path.append(f"{base_dir}/scripts/")
from preprocessing import zscore_by_blocks, get_trial_numbers
from load_data import store_data_to_pandas, select_block_ids
from scipy.io import loadmat
from himalaya.ridge import Ridge, RidgeCV
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import himalaya
import torch
backend = himalaya.backend.set_backend("torch_cuda")
device = 2
torch.cuda.set_device(device)
from sklearn.base import BaseEstimator
from copy import deepcopy
from reg_funcs import blockCV, create_block_sess_dict

# Define the parameter grid for alpha (regularization strength)
alphas = np.exp2(np.arange(-5, 40, 1))
alphas = np.hstack((0,alphas))
param_grid = {'alpha': alphas}
n_splits = 6
model_name = '44-6v-8_20'

savePath = f"/data/LLMs/willet/regression_results/{model_name}/"
os.makedirs(f"{savePath}", exist_ok=True)

block_sess_all = np.load('/data/LLMs/willet/block_sess.npy')
area_6v_data = np.load('/data/LLMs/willet/smooth_ba6v_tx.npy')
area_44_data = np.load('/data/LLMs/willet/smooth_ba44_tx.npy')

sessions_all = np.array([x.split('_')[1] for x in block_sess_all])

X = area_44_data
y = area_6v_data

sess_number = 0

for sess in np.unique(sessions_all):
    
    print("Session: ", sess)
    
    print(f"Session number: {sess_number} out of {np.unique(sessions_all).shape[0]}")
    
    sess_number+=1
    
    s_idxs = np.argwhere(sessions_all==sess).squeeze()
    
    block_sess = block_sess_all[s_idxs].squeeze()

    block_sess_dict = create_block_sess_dict(np.unique(block_sess))

    # Create the outer cross-validation based on sentence IDs
    outer_cv = blockCV(block_sess, block_sess_dict, n_splits)

    # Prepare to store results
    cv_results = []

    fold_number = 0
    r2_folds = []

    y_pred_all = []
    y_pred_intercept_all = []
    y_test_all = []
    best_alpha = []

    for train_idx, test_idx in outer_cv.split(None):
        
        # Initialize the Ridge regression model
        ridge = Ridge()

        print(f"Fold number: {fold_number}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        block_sess_train = block_sess[train_idx].squeeze()
        block_sess_train_dict = create_block_sess_dict(np.unique(block_sess_train))

        inner_cv = blockCV(block_sess_train, block_sess_train_dict, n_splits=n_splits-1)
        
        # Set up the grid search with cross-validation
        grid_search = GridSearchCV(ridge, param_grid, cv=inner_cv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        # Evaluate the best model on the outer test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_null = np.repeat(np.expand_dims(np.mean(X_train, axis=0),axis=0), X_test.shape[0], axis=0)
        
        y_pred_all.append(y_pred)
        y_pred_intercept_all.append(y_pred_null)
        y_test_all.append(y_test)
        
        r2 = r2_score(y_test, y_pred)
        
        print("Average r2 for fold: ", r2)
        
        best_alpha.append(grid_search.best_params_['alpha'])
        
        print(f"Best alpha for this fold: {grid_search.best_params_['alpha']}")
        
        fold_number += 1

    y_pred_all_np = np.vstack(y_pred_all)
    y_test_all_np = np.vstack(y_test_all)
    y_intercept_all_np = np.vstack(y_pred_intercept_all)

    r2_oos = 1 - mean_squared_error(y_test_all_np, y_pred_all_np, multioutput='raw_values')/mean_squared_error(y_test_all_np, y_intercept_all_np, multioutput='raw_values')

    r, p = pearsonr(y_test_all_np, y_pred_all_np, axis=0)

    r2 = r2_score(y_test_all_np, y_pred_all_np, multioutput='raw_values')

    print(np.mean(r2_oos), np.mean(r2), np.mean(r))

    np.save(f'{savePath}y_pred_{sess}', y_pred_all_np)
    np.save(f'{savePath}y_test_{sess}', y_test_all_np)
    np.save(f'{savePath}y_intercept_{sess}', y_intercept_all_np)
    np.save(f'{savePath}r2_oos_{sess}', r2_oos)
    np.save(f'{savePath}r_{sess}', r)
    np.save(f'{savePath}r2_{sess}', r2)


