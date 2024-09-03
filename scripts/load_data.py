import numpy as np
import pandas as pd
from preprocessing import zscore_by_blocks, get_trial_numbers

def skip_non_selected_files(file, selected_files):
    
    
    # return False if file is in selected_files,
    # or if selected_files is empty
    
    if len(selected_files) > 0:
        for s in selected_files:
            if s in file:
                return False
            
    elif len(selected_files) == 0:
        return False
    
    return True

def map_from_channel_index_to_brain_area(index):
    if 0 <= index <= 31:
        return 'ip_6v'
    elif 32 <= index <= 63:
        return 'sa_6v'
    elif 64 <= index <= 95:
        return 'sp_6v'
    elif 96 <= index <= 127:
        return 'ia_6v'
    elif 128 <= index <= 159:
        return 'ia_44'
    elif 160 <= index <= 191:
        return 'sp_44'
    elif 192 <= index <= 223:
        return 'sa_44'
    elif 224 <= index <= 255:
        return 'ip_44'
    else:
        raise ValueError("Index out of range. Valid indices are between 0 and 255.")
    
def remove_delay_mismatch_trials(trialState_tn, trialDelays, tn, remove_thresh=0.1):
    
    '''
    :param list trialState: the trial state for every timestep for the trial at index tn
    :param list trialDelays: precomputed trialDelays for every trial
    :param int tn: trialnum index
    :param float remove_thresh: if the diff is greater than remove_thresh seconds, remove
    
    Simple function which checks if the delay computed using trialState matches the 
    delay at index tn for trialDelays. 
    '''
    
    remove_trial = False
    
    td = trialDelays[tn] # trial delay time for index tn
    delay_time_from_trialState = (np.argwhere(trialState_tn==0).shape[0])*20/1000 # convert to sec
    
    if np.abs(delay_time_from_trialState - td) > remove_thresh:
        remove_trial = True
        
    return remove_trial
    
def bin_relative_to_go_cue(trialNumber, trialState, trialDelays):
    
    '''
    :param list trialNumber: indicates the trial each time bin belongs to 
    :param list trialState: indicates whether each time bin is delay (0), go (1), or google text2speech (3)
    :param list trialDelays: delay length for each trial
    
    This function will return the bin relative to go cue for each trial, as well as indices corresponding
    to trials which need to be removed due to incorrect delay timing. 
    '''
            
    assert trialNumber.shape[0] == trialState.shape[0], print("Shapes are mismatched")
    trialNum_unq, _ = np.unique(trialNumber, return_counts=True)
    
    bil_rel_to_go = []
    
    # there are some trials where the precomputed delay times don't match the trialState
    # times. Based on Frank's recommendation, we'll remove those. 
    remove_trial_idxs = [] # stores trialNumbers that should be removed
    num_trials_removed = 0
    for tn in trialNum_unq:
        
        # get indices for given trial
        tn_idxs = np.argwhere(trialNumber==tn).squeeze()

        # store number of time bins for that trial
        num_bins_trial = tn_idxs.shape[0]
        
        # get trialState for selected trial
        trialState_tn = trialState[tn_idxs].squeeze()
     
        first_go_idx = np.argwhere(trialState_tn==1).squeeze()[0]

        # add list that increases by 1 each element, and element at first_go_idx is 0
        bil_rel_to_go.extend(np.arange(num_bins_trial) - first_go_idx)
            
    return bil_rel_to_go
            
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

def stimuli_num_trials(trialNumber, stimuli):
    
    '''
    :param list trialNumber: trial number (integer) for selected trials for each 20 ms bin
    :param list stimuli: list of length number of total trials, indicating stimuli shown
    on that trial.
    
    Currently, a single stimuli is only provided for each trial. This function 
    returns a list of of length trialNumber, which contains the stimuli for each bin. 
    '''
    
    # obtain unqiue trial numbers and counts
    # the counts indicate the number of 20ms bins that belong to that trial
    trialNumber_unique, trialNumber_counts = np.unique(trialNumber, return_counts=True)
    
    # select stimuli corresponding to trials
    stimuli_selected = stimuli[trialNumber_unique] 
    
    # map stimuli to ids 
    stimuli_repeated = []
    
    for i, s in enumerate(stimuli_selected):
        stimuli_num_bins = trialNumber_counts[i] # number of time bins for that trial
        s_repeated = np.repeat(s, stimuli_num_bins) # repeat stimuli num bins times
        stimuli_repeated.extend(s_repeated) # add it to list 

    return np.array(stimuli_repeated)


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



def store_data_to_pandas(session_dict, session_name, selected_block_numbers=[],
                         stimuliKey='sentences', zscore_bool=True):
    
    '''
    :param dict session_dict: session data loaded using scipy 
    :param str session_name: name of session data
    :param list selected_block_numbers: which block numbers to store into dataframe, if empty list
    collect all blocks. 
    :param str stimuliKey: the key used to index the stimuli for that session 
    :param zscore_bool: if False, don't zscore
    
    Returns a pandas df with the following column
        blockNum: the block number (integer)
        session: session name (str)
        spikePow: the power in 20 ms bins after high pass filtering 250 Hz for each of the 256 channels
        threshCross:  number of binned threshold crossings (4.5 x RMS threshold) for each of the 256 channels
        trialState: 0 for delay period, 1 for go period
        stimuli: stimuli to produce, corresponding to each time bin (str). For sentence trials this is the 
        sentence that was repeated, for orofacial its the orofacial movement the patient was instructed to produce,
        etc...
        trialNumber: trial number corresponding to each bin (int)
    '''
    
    trialNumbers = get_trial_numbers(session_dict)
    blockNum = session_dict['blockNum'].squeeze()
    spikePow = session_dict['spikePow'].squeeze()
    threshCross = session_dict['tx2'].squeeze()
    trialState = session_dict['trialState'].squeeze()
    stimuli = session_dict[stimuliKey].squeeze()
    blockType = session_dict['blockTypes']
    trialDelayTimes = session_dict['trialDelayTimes'].squeeze()
    audioEnvelope = session_dict['audioEnvelope'].squeeze()
    
    map_blocknum_to_blocktype = {}
    for bn, bt in zip(np.unique(blockNum), blockType):
        map_blocknum_to_blocktype[bn] = bt[0]
        
    
    store_data_in_dict = {'blockNum': [], 'blockName': [], 'session': [], 'trialState': [], 
    'stimuli': [], 'trialNumber': [], 'bin_rel_go': [], 'audioEnvelope':[]}
    
    num_channels = spikePow.shape[1]
    
    for i in range(num_channels):
        ba = map_from_channel_index_to_brain_area(i)
        store_data_in_dict[f'pow-{ba}-{i}'] = []
        store_data_in_dict[f'tx-{ba}-{i}'] = []
        
    # select all blocks if selected_block_numbers is []
    if len(selected_block_numbers) == 0:
        selected_block_numbers = np.unique(blockNum)
    
    for sb in selected_block_numbers:
        
        # get idxs corresponding to the specified block (sb)
        sb_idxs = np.argwhere(blockNum==sb).squeeze()
        
        # trialState is 0 (delay) or 1 (go cue), or 3 (google text2speech)
        trialState_sb = trialState[sb_idxs].squeeze()
        
        # neural data is stored in 20ms bins. trialNumbers indicates that trial that each bin belongs to
        trialNumber_sb = trialNumbers[sb_idxs].squeeze()
        
        # for each trial, store bin idx relative to go cue
        bin_relative_to_go_cue_list = bin_relative_to_go_cue(trialNumber_sb, trialState_sb, trialDelayTimes)
        
        # add the indices corresponding to the trial to be removed 
        # using the session level indices (because that's what sb_idxs refers to)
        #remove_trial_idxs = []
        #for trial in remove_trials:
        #    remove_trial_idxs.extend(np.argwhere(trialNumbers==trial))
        
        #print(f'removing {nt_removed} trials out of {np.unique(trialNumber_sb).shape[0]} trials')
        
        # remove the mismatch trials from the selected block idxs
        #sb_idxs = np.setdiff1d(sb_idxs, remove_trial_idxs)
        
        # reselect trialState and trialNumber now that we've removed trials 
        trialState_sb = trialState[sb_idxs].squeeze()
        trialNumber_sb = trialNumbers[sb_idxs].squeeze()
                
        num_time_bins = sb_idxs.shape[0]
        
        if zscore_bool:        
            # select neural data (both spikePow and threshold crossings)
            spikePow_sb = zscore_data(spikePow[sb_idxs].squeeze())
            threshCross_sb = zscore_data(threshCross[sb_idxs].squeeze())
        else:
            spikePow_sb = spikePow[sb_idxs].squeeze()
            threshCross_sb = threshCross[sb_idxs].squeeze()
       
        # repeat block number and session name to store in long format 
        sb_repeated = np.repeat(sb, num_time_bins)
        block_name_repeated = np.repeat(map_blocknum_to_blocktype[sb], num_time_bins)
        session_name_repeated = np.repeat(session_name, num_time_bins)
        
        stimuli_sb = stimuli_num_trials(trialNumber_sb, stimuli)
        
        audioEnvelope_sb = audioEnvelope[sb_idxs].squeeze()
        
        for i in range(num_channels):
            
            ba = map_from_channel_index_to_brain_area(i)
            
            # store high pass power and threshold crossings
            store_data_in_dict[f'pow-{ba}-{i}'].extend(spikePow_sb[:, i])
            store_data_in_dict[f'tx-{ba}-{i}'].extend(threshCross_sb[:, i])
            
                # Get lengths of all arrays
        lengths = [
            len(trialState_sb),
            len(sb_repeated),
            len(block_name_repeated),
            len(session_name_repeated),
            len(trialNumber_sb),
            len(stimuli_sb),
            len(bin_relative_to_go_cue_list),
            len(audioEnvelope_sb)
        ]

        # Check if all lengths are equal
        if not all(length == lengths[0] for length in lengths):
            breakpoint()
           
        store_data_in_dict['audioEnvelope'].extend(audioEnvelope_sb) 
        store_data_in_dict['trialState'].extend(trialState_sb)
        store_data_in_dict['blockNum'].extend(sb_repeated)
        store_data_in_dict['blockName'].extend(block_name_repeated)
        store_data_in_dict['session'].extend(session_name_repeated)
        store_data_in_dict['trialNumber'].extend(trialNumber_sb)
        store_data_in_dict['stimuli'].extend(stimuli_sb)
        store_data_in_dict['bin_rel_go'].extend(bin_relative_to_go_cue_list)
        
    return pd.DataFrame(store_data_in_dict)


def dataframe_size_in_gb(df):
    # Get the memory usage of the DataFrame in bytes
    memory_usage_bytes = df.memory_usage(deep=True).sum()
    # Convert bytes to gigabytes
    memory_usage_gb = memory_usage_bytes / (1024 ** 3)
    return memory_usage_gb
