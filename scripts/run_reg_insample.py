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
from preprocessing import create_lagged_data
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
import argparse

parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument("--fname", type=str, required=True, help="Folder to save")
parser.add_argument("--X", type=str, help="if True, fit regression from 6v to 6v")
parser.add_argument("--y", type=str, help="if True, fit regression from 44 to 44")
parser.add_argument("--load_data_folder", type=str, help="where to load X and y from", default='/data/LLMs/willet/')

args = parser.parse_args()
fname = args.fname
X = args.X
y = args.y
load_data_folder = args.load_data_folder

# Define the parameter grid for alpha (regularization strength)
alphas = np.exp2(np.arange(-5, 40, 1))
alphas = np.hstack((0,alphas))
param_grid = {'alpha': alphas}
n_splits = 6
    
X = np.load(f"{load_data_folder}{X}.npy")
y = np.load(f"{load_data_folder}{y}.npy")

model_name = f'{fname}'

savePath = f"/data/LLMs/willet/regression_results/{model_name}/"
os.makedirs(f"{savePath}", exist_ok=True)

block_sess_all = np.load('/data/LLMs/willet/block_sess.npy')

sessions_all = np.array([x.split('_')[1] for x in block_sess_all])

sess_number = 0

for sess in np.unique(sessions_all):
    
    print("Session: ", sess)
    
    print(f"Session number: {sess_number} out of {np.unique(sessions_all).shape[0]}")
    
    sess_number+=1
    
    s_idxs = np.argwhere(sessions_all==sess).squeeze()
    
    block_sess = block_sess_all[s_idxs].squeeze()

    block_sess_dict = create_block_sess_dict(np.unique(block_sess))
    
    X_sess = X[s_idxs].squeeze()
    y_sess = y[s_idxs].squeeze()
    
    if len(X_sess.shape) == 1:
        X_sess = np.expand_dims(X_sess, axis=-1)
   
    # Initialize the Ridge regression model
    ridge = Ridge(alpha=1e-5)
    
    ridge.fit(X_sess, y_sess)
        
    y_pred = ridge.predict(X_sess)
   
    r2 = r2_score(y_sess, y_pred, multioutput='raw_values')

    print(np.mean(r2))

    np.save(f'{savePath}y_pred_{sess}', y_pred)



