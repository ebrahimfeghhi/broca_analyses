import numpy as np
from scipy.io import loadmat 
import sys
sys.path.append("/home2/ebrahim/neural_seq_decoder/scripts/")
from load_data import store_data_to_pandas


fiftyWords = loadmat('/home2/ebrahim/neural_seq_decoder/tuningTasks/t12.2022.05.03_fiftyWordSet.mat')

blockNum = fiftyWords['blockNum']
fiftyWords['blockTypes'] = np.expand_dims(np.repeat('Words', blockNum.shape[0]),axis=-1)

words_pd = store_data_to_pandas(session_dict=fiftyWords, session_name='Words', stimuliKey='trialCues')

breakpoint()