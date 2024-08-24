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
from load_data import store_data_to_pandas
from copy import deepcopy

load_data_from_pickle = True # if true, load pre-existing pandas data

if load_data_from_pickle:
    
    all_sessions = pd.read_pickle(f"/data/LLMs/willet/all_sessions_pd.pkl")  
      
else:

    session_pd_store = []
    counter = 0

    for file in os.listdir(f"{base_dir}sentences/"):
        
        print("loading session data")
        session_dict = scipy.io.loadmat(f'{base_dir}sentences/{file}')
        session_name = file.split('_')[0].replace('t12.2022.', '')

        session_pd = store_data_to_pandas(session_dict, session_name=session_name)
        
        session_pd_store.append(session_pd)
        
    all_sessions = pd.concat(session_pd_store)
    all_sessions.to_pickle('/data/LLMs/willet/all_sessions_pd.pkl')
    
all_sessions['block+sess'] = all_sessions['blockNum'].astype(str) + '_' + all_sessions['session'] 
all_sessions['tn+block+sess'] = all_sessions['trialNumber'].astype(str) + '_' + all_sessions['blockNum'].astype(str) + '_' + all_sessions['session']

# put a 1 where the go cue is, and a 2 when a new trial starts
bin_rel_go_np = np.array(all_sessions['bin_rel_go'])
go_end_cue = np.zeros_like(bin_rel_go_np)
go_end_cue[np.argwhere(bin_rel_go_np==0)] = 1
end_cue_indices = [i for i in range(bin_rel_go_np.shape[0]-1) if bin_rel_go_np[i]>0 and bin_rel_go_np[i+1]<0]
go_end_cue[end_cue_indices] = 2

block_sess = deepcopy(np.unique(all_sessions['block+sess']))
session_names, block_counts = np.unique([s.split('_')[1] for s in block_sess], return_counts=True)
plt.figure(figsize=(15,10))
plt.bar(session_names, block_counts)
plt.ylabel("Number of blocks", fontsize=16)
plt.xlabel("Session label", fontsize=16)
plt.savefig("/home2/ebrahim/neural_seq_decoder/figures/num_blocks_per_session")
plt.show()

# each key is a session, and each value is a 
# list containing the block numbers for that session
block_sess_dict = {}
block_sess = deepcopy(np.unique(all_sessions['block+sess']))
for sb in block_sess:
    sess = sb.split('_')[1]
    b = sb.split('_')[0]
    if sess not in block_sess_dict:
        block_sess_dict[sess] = []
    if b not in block_sess_dict[sess]:
        block_sess_dict[sess].append(b)
        
column_names = ['tx-', 'pow-']

for select_column in column_names:
    
    all_sessions_neural_data = all_sessions.filter(regex=select_column)
    ba_44_neural = np.array(all_sessions_neural_data.filter(regex='44-')).astype("float32")
    ba_6v_neural = np.array(all_sessions_neural_data.filter(regex='6v-')).astype("float32")

    c = 0
    for bs in block_sess:
        block_sess_idxs = np.argwhere(all_sessions['block+sess']==bs) 

        # smooth neural data within each block using a gaussian filter with 40ms SD
        ba_44_neural[block_sess_idxs] = scipy.ndimage.gaussian_filter1d(ba_44_neural[block_sess_idxs], axis=0, sigma=2)
        ba_6v_neural[block_sess_idxs] = scipy.ndimage.gaussian_filter1d(ba_6v_neural[block_sess_idxs], axis=0, sigma=2)
        c+=1
        
        print(f"Block number {c} out of {block_sess.shape[0]}")
        
    np.save(f'/data/LLMs/willet/smooth_ba44_{select_column[:-1]}', ba_44_neural)
    np.save(f'/data/LLMs/willet/smooth_ba6v_{select_column[:-1]}', ba_6v_neural)
    
# Ensure the column is of type str
all_sessions['block+sess'] = all_sessions['block+sess'].astype(str)

# Convert to numpy array
np.save('/data/LLMs/willet/block_sess', np.array(all_sessions['block+sess'], dtype=str))
np.save('/data/LLMs/willet/tn_block_sess', np.array(all_sessions['tn+block+sess'], dtype=str))