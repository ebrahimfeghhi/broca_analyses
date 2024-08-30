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
from preprocessing import preprocess_himalayas
import himalaya
import torch
backend = himalaya.backend.set_backend("torch_cuda")
from sklearn.base import BaseEstimator
from copy import deepcopy
from reg_funcs import blockCV, create_block_sess_dict
import argparse
import json
from sklearn.pipeline import make_pipeline
from run_reg_func import run_regression

parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument("--fname", type=str, required=True, help="Folder to save")
parser.add_argument("--X", type=str)
parser.add_argument("--y", type=str, default='smooth_ba6v_pow')
parser.add_argument("--load_data_folder", type=str, help="where to load X and y from", default='/data/LLMs/willet/regression_data/')
parser.add_argument("--f_list", default=[], help="list containing number of features in each space", type=json.loads)
parser.add_argument("--niter", type=int, default=1)
parser.add_argument("--exist_ok", action='store_true', default=False)
parser.add_argument("--load_X_by_sess", action='store_true', default=False)
parser.add_argument("--val_sess", action='store_true', default=False)
parser.add_argument("--device", type=int, default=2)

args = parser.parse_args()
fname = args.fname
X_file = args.X
y = args.y
load_data_folder = args.load_data_folder
features_list = args.f_list
n_iter = args.niter
exist_ok = args.exist_ok
load_X_by_sess = args.load_X_by_sess
val_sess = args.val_sess
device = args.device

torch.cuda.set_device(device)

# Define the parameter grid for alpha (regularization strength)
alphas = np.exp2(np.arange(-5, 40, 1))
alphas = np.hstack((0,alphas))
param_grid = {'alpha': alphas}
n_splits = 6
y = np.load(f"{load_data_folder}{y}.npy").astype('float32')

if load_X_by_sess == False:
    
    X = np.load(f"{load_data_folder}{X_file}.npy").astype('float32')
    print(f"loaded data")

    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=-1)

else:
     X = None

model_name = f'{fname}'

savePath = f"/data/LLMs/willet/regression_results/{model_name}/"

os.makedirs(f"{savePath}", exist_ok=exist_ok)

block_sess_all = np.load('/data/LLMs/willet/block_sess.npy')

sessions_all = np.array([x.split('_')[1] for x in block_sess_all])

feature_grouper = preprocess_himalayas(features_list, use_kernelized=False)

if val_sess:
    validation_sessions = np.load("/data/LLMs/willet/val_sess.npy")
else:
    validation_sessions = None
    

run_regression(X, y, sessions_all, val_sess, validation_sessions, block_sess_all, n_splits, 
                load_X_by_sess, load_data_folder, X_file, alphas, features_list, savePath, feature_grouper, n_iter)