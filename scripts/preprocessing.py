import numpy as np
import scipy
import pandas as pd


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