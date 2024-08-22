import numpy as np
import scipy
import pandas as pd
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.kernel_ridge import Kernelizer
from himalaya.ridge import ColumnTransformerNoStack

def zscore_by_blocks(neural_data, blockNum):
    
    '''
    :param ndarray neural_data: shape -> time x channels
    :param ndarray blockNum: shape -> time, where index i denotes the block
    number for time point i
    
    Z-scores neural data within each block.
    '''

    neural_data_z_scored = np.zeros_like(neural_data)
    for block in np.unique(blockNum):
        block_idxs = np.argwhere(blockNum==block).squeeze()
        neural_data_block = neural_data[block_idxs].squeeze()
        zscored_dat = (neural_data_block - np.mean(neural_data_block, axis=0))/np.std(neural_data_block,axis=0)
        neural_data_z_scored[block_idxs] = zscored_dat
  
    return neural_data_z_scored

def get_trial_numbers(session_X):
    
    '''
    :param dict session_X:
    
    Returns a list of length number of bins, where each value indicates
    the trial that the bin belongs to. 
    '''
    
    trialState = session_X['trialState']
    trialNumber = 0
    trialNumber_store = []
    
    for i,t in enumerate(trialState):
        
        trialNumber_store.append(trialNumber)
        
        # last trial
        if i == len(trialState)-1:
            trialNumber += 1
            
        # if the next timestep is a delay, then a new trial has started.
        # I do greater than 0 (instead of equal to 1) because trialState can equal 2 
        # , which indicates return period on tuning tasks. Also trialState can equal 3 
        # on sentences to indicate the patient output is being red by google text2speech.
        elif t > 0 and trialState[i+1] == 0:
            trialNumber+=1
            
    return np.array(trialNumber_store)

def create_lagged_data(X,block_sess):
    
    '''
    Lags neural data by 1 timestep and adds it as a predictor.
    '''
    
    X_lagged = np.zeros((X.shape[0], X.shape[1]*2))
    zeros_features = np.zeros((1, X.shape[1]))

    for bs in np.unique(block_sess):
        
        bs_idxs = np.argwhere(block_sess==bs).squeeze()
        
        # to create the lagged features, we make the first timepoint 
        # in the block a vector of zeros since there is no previous time
        # that would be helpful to predict. Then, we start from the first index
        # and go until the second to last index. The lagged features are then 
        # hstacked with the non-lagged features to create our new X matrix. 
        X_bs_lagged = np.vstack((zeros_features, X[bs_idxs][:-1]))
        X_lagged[bs_idxs] = np.hstack((X[bs_idxs], X_bs_lagged))
        
    return X_lagged

def preprocess_himalayas(n_features_list, use_kernelized):
        
    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices = [
        slice(start, end)
        for start, end in zip(start_and_end[:-1], start_and_end[1:])
    ]
        
    feature_names = [f'feat_{i}' for i in range(len(n_features_list))]

    if use_kernelized:
        print("USING KERNEL")
        kernelizers = [(name, Kernelizer(kernel='linear'), slice_)
                    for name, slice_ in zip(feature_names, slices)]
        column_kernelizer = ColumnKernelizer(kernelizers)
        return column_kernelizer
    
    else:
        print("USING REGULAR RIDGE")
        scalers = [(name,'passthrough', slice_)
                    for name, slice_ in zip(feature_names, slices)]
        column_scaler = ColumnTransformerNoStack(scalers)
        return column_scaler
