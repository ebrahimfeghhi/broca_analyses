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
from preprocessing import create_lagged_data, preprocess_himalayas
from scipy.io import loadmat
from himalaya.ridge import Ridge, RidgeCV, GroupRidgeCV
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import himalaya
import torch
backend = himalaya.backend.set_backend("torch_cuda")
device = 1
torch.cuda.set_device(device)
from sklearn.base import BaseEstimator
from copy import deepcopy
from reg_funcs import blockCV, create_block_sess_dict
import argparse
import json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument("--fname", type=str, required=True, help="Folder to save")
parser.add_argument("--X", type=str, help="if True, fit regression from 6v to 6v")
parser.add_argument("--y", type=str, help="if True, fit regression from 44 to 44")
parser.add_argument("--load_data_folder", type=str, help="where to load X and y from", default='/data/LLMs/willet/regression_data/')
parser.add_argument("--f_list", default=[], help="list containing number of features in each space", type=json.loads)
parser.add_argument("--standardize", action='store_true', default=False)
parser.add_argument("--niter", type=int, default=1000)

args = parser.parse_args()
fname = args.fname
X = args.X
y = args.y
load_data_folder = args.load_data_folder
features_list = args.f_list
standardize = args.standardize
n_iter = args.niter

# Define the parameter grid for alpha (regularization strength)
alphas = np.exp2(np.arange(-5, 40, 1))
alphas = np.hstack((0,alphas))
param_grid = {'alpha': alphas}
n_splits = 6

X = np.load(f"{load_data_folder}{X}.npy").astype('float32')
y = np.load(f"{load_data_folder}{y}.npy").astype('float32')

if len(X.shape) == 1:
    X = np.expand_dims(X, axis=-1)

model_name = f'{fname}'

savePath = f"/data/LLMs/willet/regression_results/{model_name}/"
os.makedirs(f"{savePath}", exist_ok=True)

block_sess_all = np.load('/data/LLMs/willet/block_sess.npy')

sessions_all = np.array([x.split('_')[1] for x in block_sess_all])

sess_number = 0

feature_grouper = preprocess_himalayas(features_list, use_kernelized=False)

for sess in np.unique(sessions_all):
    
    print("Session: ", sess)
    
    print(f"Session number: {sess_number} out of {np.unique(sessions_all).shape[0]}")
    
    sess_number+=1
    
    s_idxs = np.argwhere(sessions_all==sess).squeeze()
    
    block_sess = block_sess_all[s_idxs].squeeze()

    block_sess_dict = create_block_sess_dict(np.unique(block_sess))

    # Create the outer cross-validation based on sentence IDs
    outer_cv = blockCV(block_sess, block_sess_dict, n_splits)

    fold_number = 0
    r2_folds = []

    y_pred_all = []
    y_test_all = []
    r2_fold = []
    for train_idx, test_idx in outer_cv.split(None):
        
        # Initialize the Ridge regression model
        
        print(f"Fold number: {fold_number}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        #if standardize: 
        #    scaler = StandardScaler()
        #    X_train = scaler.fit_transform(X_train)
        #    X_test = scaler.transform(X_test)
            
        
        print(X_train.shape, y_train.shape)
        
        block_sess_train = block_sess[train_idx].squeeze()
        block_sess_train_dict = create_block_sess_dict(np.unique(block_sess_train))

        inner_cv = blockCV(block_sess_train, block_sess_train_dict, n_splits=n_splits-1)
        
        n_alphas_batch = len(alphas)
        targets_batch = y_train.shape[1]
        
        print("features list: ", features_list)

        if len(features_list) > 1:
        # banded ridge regression
            print("banded ridge")
            model = GroupRidgeCV(groups="input", fit_intercept=True, cv=inner_cv, Y_in_cpu=False,
                            solver_params={'alphas': alphas, 'n_iter': n_iter, 'warn': False, 
                                        'n_alphas_batch': n_alphas_batch, 'n_targets_batch': targets_batch})
            pipe = make_pipeline(feature_grouper, model)
            
        else:
            # vanilla ridge regression
            print("regular ridge")
            model = RidgeCV(alphas=alphas, fit_intercept=True, cv=inner_cv)
            pipe = make_pipeline(model)
            
        
        _ = pipe.fit(X_train, y_train)
 
        y_pred = pipe.predict(X_test)       
        y_pred_all.append(y_pred)
        y_test_all.append(y_test)
        
        r2 = r2_score(y_test, y_pred)
        r2_fold.append(r2)
        print("Average r2 for fold: ", r2)
    
        fold_number += 1

    y_pred_all_np = np.vstack(y_pred_all)
    y_test_all_np = np.vstack(y_test_all)
    
    r, p = pearsonr(y_test_all_np, y_pred_all_np, axis=0)
    r2 = r2_score(y_test_all_np, y_pred_all_np, multioutput='raw_values')
    print("R2: ", r2.mean())

    np.save(f'{savePath}y_pred_{sess}', y_pred_all_np)
    np.save(f'{savePath}y_test_{sess}', y_test_all_np)
    np.save(f'{savePath}r_{sess}', r)
    np.save(f'{savePath}r2_{sess}', r2)


