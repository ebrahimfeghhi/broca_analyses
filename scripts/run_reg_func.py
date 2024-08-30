import numpy as np
import numpy as np
import sys
base_dir = '/home2/ebrahim/neural_seq_decoder/' 
sys.path.append(f"{base_dir}/scripts/")
from himalaya.ridge import RidgeCV, GroupRidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import himalaya
import torch
backend = himalaya.backend.set_backend("torch_cuda")
device = 2
torch.cuda.set_device(device)
from copy import deepcopy
from reg_funcs import blockCV, create_block_sess_dict
from sklearn.pipeline import make_pipeline


def run_regression(X, y, sessions_all, val_sess, validation_sessions, block_sess_all, n_splits, 
                   load_X_by_sess, load_data_folder, X_file, alphas, features_list, savePath, feature_grouper, n_iter):

            
    sess_number = 0
    
    for sess in np.unique(sessions_all):
        
        if val_sess:
            if sess not in validation_sessions:
                continue
        
        print("Session: ", sess)
        
        print(f"Session number: {sess_number} out of {np.unique(sessions_all).shape[0]}")
        
        sess_number+=1
        
        s_idxs = np.argwhere(sessions_all==sess).squeeze()
        
        block_sess = block_sess_all[s_idxs].squeeze()

        block_sess_dict = create_block_sess_dict(np.sort(np.unique(block_sess)))

        # Create the outer cross-validation based on sentence IDs
        outer_cv = blockCV(block_sess, block_sess_dict, n_splits)

        fold_number = 0
        y_pred_all = []
        y_test_all = []
        r2_fold = []
        
        y_sess = y[s_idxs].squeeze()
        
        if load_X_by_sess:
            
            X_sess = np.load(f"{load_data_folder}{X_file}_{sess}.npy").squeeze().astype('float32')
            
        else:
                
            X_sess = X[s_idxs].squeeze()

        for train_idx, test_idx in outer_cv.split(None):
            
            # train_idx and test_idx are based on the session indexed data (X_sess and y_sess)

            print(f"Fold number: {fold_number}")
            
            X_train, X_test = X_sess[train_idx], X_sess[test_idx]
            y_train, y_test = y_sess[train_idx], y_sess[test_idx]
            
            print(f"X train shape: {X_train.shape}")
            
            block_sess_train = block_sess[train_idx].squeeze()
            block_sess_train_dict = create_block_sess_dict(np.unique(block_sess_train))

            inner_cv = blockCV(block_sess_train, block_sess_train_dict, n_splits=n_splits-1)
            
            # when there's many features, restrict how many alphas
            if X_train.shape[1] > 250:
                n_alphas_batch = 1
                targets_batch = y_train.shape[1]
            else:
                n_alphas_batch = len(alphas)
                targets_batch = y_train.shape[1]
                

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
                model = RidgeCV(alphas=alphas, fit_intercept=True, cv=inner_cv, solver_params={'n_alphas_batch': n_alphas_batch, 'n_targets_batch': targets_batch})
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
        np.save(f'{savePath}r2_{sess}', r2)


