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

def bin_relative_to_go_cue(trialNumber, trialState):
    
    '''
    :param list trialNumber:  
    :param list trialState: 
    
    trialNumber and trialState should contain the trial numbers and trial state for lists within a block.
    This function will return the bin relative to go cue for each trial.
    '''
            
    assert trialNumber.shape[0] == trialState.shape[0], print("Shapes are mismatched")
    trialNum_unq, trialNum_count = np.unique(trialNumber, return_counts=True)
    
    bil_rel_to_go = []
    
    for tn in trialNum_unq:
        
        # get indices for given trial
        tn_idxs = np.argwhere(trialNumber==tn).squeeze()

        # store number of time bins for that trial
        num_bins_trial = tn_idxs.shape[0]
        
        # get trialState for selected trial
        trialState_tn = trialState[tn_idxs].squeeze()
        
        # find index where go cue occurs
        first_go_idx = np.argwhere(trialState_tn==1).squeeze()[0]

        # add list that increases by 1 each element, and element at first_go_idx is 0
        bil_rel_to_go.extend(np.arange(num_bins_trial) - first_go_idx)
        
    return bil_rel_to_go

def get_trial_numbers(session_X):
    
    '''
    :param dict session_X:
    
    Assign each bin a trial number. 
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
        # I do greater than 0 (instead of equal to 1) because sometimes 
        # trialState is equal to 3 (for reasons unknown...)
        elif t > 0 and trialState[i+1] == 0:
            trialNumber+=1
            
    return np.array(trialNumber_store)
            
def select_block_ids(session_dict, select_block):
    
    '''
    :param dict session_dict: session data loaded using scipy  
    :param str select_block: selected block, can be Switchboard or Chang

    Returns block numbers corresponding to the selected block.
    '''
    
    blockType = session_dict['blockTypes'].squeeze()
    blockList = session_dict['blockList'].squeeze()
    
    selected_block_numbers = []
    
    for bt, bl in zip(blockType, blockList):
        
        if select_block in bt[0]:
            selected_block_numbers.append(bl)
            
    return selected_block_numbers

def sentences_num_trials(trialNumber, sentences):
    
    '''
    :param list trialNumber: trial number (integer) for selected trials for each 20 ms bin
    :param list sentences: list of length number of total trials, indicating sentence shown
    on that trial.
    
    Currently, a single sentence is only provided for each trial. This function 
    returns a list of of length trialNumber, which contains the sentence for each bin. 
    '''
    
    # obtain unqiue trial numbers and counts
    # the counts indicate the number of 20ms bins that belong to that trial
    trialNumber_unique, trialNumber_counts = np.unique(trialNumber, return_counts=True)
    
    # select sentences corresponding to trials
    sentences_selected = sentences[trialNumber_unique] 
    
    # map sentences to ids 
    sentence_repeated = []
    
    for i, s in enumerate(sentences_selected):
        sentence_num_bins = trialNumber_counts[i] # number of time bins for that trial
        s_repeated = np.repeat(s, sentence_num_bins) # repeat sentence num bins times
        sentence_repeated.extend(s_repeated) # add it to list 

    return np.array(sentence_repeated)


def zscore_data(neural_data):
    
    '''
    :param ndarray neural_data: num_time_bins x channels, 
    
    Z score each channel
    '''
    
    channel_means = np.mean(neural_data, axis=0) # take mean across time
    channel_stds = np.std(neural_data, axis=0) # take std across time 
    
    # Avoid division by zero by setting a minimum threshold for standard deviation
    min_std_threshold = 1e-6
    channel_stds = np.where(channel_stds < min_std_threshold, 1, channel_stds)
    
    return (neural_data - channel_means)/channel_stds


def print_lengths_before_adding(ba, i, num_time_bins, spikePow_sb, threshCross_sb, trialState_sb, 
                                sb_repeated, session_name_repeated, trialNumber_sb, sentence_sb, bin_relative_to_go_cue_list):
    # Calculate lengths
    brain_region_length = len(np.repeat(ba, num_time_bins))
    elec_num_length = len(np.repeat(i, num_time_bins))
    pow_length = len(spikePow_sb[:, i])
    tx_length = len(threshCross_sb[:, i])
    trialState_length = len(trialState_sb)
    blockNum_length = len(sb_repeated)
    session_length = len(session_name_repeated)
    trialNumber_length = len(trialNumber_sb)
    sentences_length = len(sentence_sb)
    bin_rel_go_length = len(bin_relative_to_go_cue_list)
    
    # Print lengths
    print(f"Length of 'brain_region': {brain_region_length}")
    print(f"Length of 'elec_num': {elec_num_length}")
    print(f"Length of 'pow-{ba}_{i}': {pow_length}")
    print(f"Length of 'tx-{ba}_{i}': {tx_length}")
    print(f"Length of 'trialState': {trialState_length}")
    print(f"Length of 'blockNum': {blockNum_length}")
    print(f"Length of 'session': {session_length}")
    print(f"Length of 'trialNumber': {trialNumber_length}")
    print(f"Length of 'sentences': {sentences_length}")
    print(f"Length of 'bin_rel_go': {bin_rel_go_length}")

def store_data_to_pandas(session_dict, selected_block_numbers, session_name):
    
    '''
    :param dict session_dict: session data loaded using scipy 
    :param list selected_block_numbers: which block numbers to store into dataframe
    :param str session_name: name of session data 
    
    Returns a pandas df with the following column
        blockNum: the block number (integer)
        session: session name (str)
        spikePow: the power in 20 ms bins after high pass filtering 250 Hz for each of the 256 channels
        threshCross:  number of binned threshold crossings (4.5 x RMS threshold) for each of the 256 channels
        trialState: 0 for delay period, 1 for go period
        sentence: sentence corresponding to each time bin (str)
        trialNumber: trial number corresponding to each bin (int)
    '''
    
    trialNumbers = get_trial_numbers(session_dict)
    blockNum = session_dict['blockNum'].squeeze()
    spikePow = session_dict['spikePow'].squeeze()
    threshCross = session_dict['tx2'].squeeze()
    trialState = session_dict['trialState'].squeeze()
    sentences = session_dict['sentences'].squeeze()
    
    store_data_in_dict = {'blockNum': [], 'session': [], 'trialState': [], 
    'sentences': [], 'trialNumber': [], 'bin_rel_go': []}

    
    num_channels = spikePow.shape[1]
    
    for i in range(num_channels):
        ba = map_from_channel_index_to_brain_area(i)
        store_data_in_dict[f'pow-{ba}{i}'] = []
        store_data_in_dict[f'tx-{ba}-{i}'] = []
        
    for sb in selected_block_numbers:
        
        print(f"{session_name}, {sb}")
        
        # get idxs corresponding to the specified block (sb)
        sb_idxs = np.argwhere(blockNum==sb).squeeze()
        
        num_time_bins = sb_idxs.shape[0]
        
        # select neural data (both spikePow and threshold crossings)
        spikePow_sb = zscore_data(spikePow[sb_idxs].squeeze())
        threshCross_sb = zscore_data(threshCross[sb_idxs].squeeze())
        
        # trialState is 0 (delay) or 1 (go cue)
        trialState_sb = trialState[sb_idxs].squeeze()
        
        # repeat block number and session name to store in long format 
        sb_repeated = np.repeat(sb, num_time_bins)
        session_name_repeated = np.repeat(session_name, num_time_bins)
        
        # neural data is stored in 20ms bins. trialNumbers indicates that trial
        # that each bin belongs to
        trialNumber_sb = trialNumbers[sb_idxs].squeeze()
        
        # for each trial, store bin idx relative to go cue
        bin_relative_to_go_cue_list = bin_relative_to_go_cue(trialNumber_sb, trialState_sb)
        
        sentence_sb = sentences_num_trials(trialNumber_sb, sentences)
        
        for i in range(num_channels):
            
            ba = map_from_channel_index_to_brain_area(i)
            
            # store high pass power and threshold crossings
            store_data_in_dict[f'pow-{ba}-{i}'].extend(spikePow_sb[:, i])
            store_data_in_dict[f'tx-{ba}-{i}'].extend(threshCross_sb[:, i])
            
            
        store_data_in_dict['trialState'].extend(trialState_sb)
        store_data_in_dict['blockNum'].extend(sb_repeated)
        store_data_in_dict['session'].extend(session_name_repeated)
        store_data_in_dict['trialNumber'].extend(trialNumber_sb)
        store_data_in_dict['sentences'].extend(sentence_sb)
        store_data_in_dict['bin_rel_go'].extend(bin_relative_to_go_cue_list)
        
        #print_lengths(store_data_in_dict)
        
    return pd.DataFrame(store_data_in_dict)
