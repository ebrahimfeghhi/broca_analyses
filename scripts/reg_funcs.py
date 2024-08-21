
from sklearn.base import BaseEstimator
import numpy as np

def create_block_sess_dict(block_sess):
    '''
    :param ndarray block_sess: unique block_session-names 
    '''
    
    block_sess_dict = {}
    for sb in block_sess:
        sess = sb.split('_')[1]
        b = sb.split('_')[0]
        if sess not in block_sess_dict:
            block_sess_dict[sess] = []
        if b not in block_sess_dict[sess]:
            block_sess_dict[sess].append(b)
            
            
    for sess, val in block_sess_dict.items():
        block_sess_dict[sess] = sorted(val)
            
    return block_sess_dict

class blockCV(BaseEstimator):
    
    def __init__(self, block_sess, block_sess_dict, n_splits):
        
        '''
        :param ndarray: numpy array where each element is a string of format BN_SESS, where BN is the block number
        and SESS is the session name
        :param dict block_sess_dict: each key is a session name, and the values are lists containing the block numbers for that seesssion
        :param int n_splits: the number of k-fold splits 
        '''
        
        self.block_sess = block_sess
        self.block_sess_dict = block_sess_dict
        self.n_splits = n_splits
            
    def split(self, X, y=None, groups=None):
        
        '''
        This function yields train and test indices such that 1/n_splits blocks from each session are placed into test
        and the remaining blocks are placed into the training set.
        '''

        # Precompute block-to-index mappings for fast lookup
        block_sess_dict_rev = {}
        for idx, block_sess in enumerate(self.block_sess):
            block, sess_name = block_sess.split('_', 1)
            if (block, sess_name) not in block_sess_dict_rev:
                block_sess_dict_rev[(block, sess_name)] = []
            block_sess_dict_rev[(block, sess_name)].append(idx)

        for split_idx in range(self.n_splits):
            
            train_idx, test_idx = [], []

            for sess_name, blocks in self.block_sess_dict.items():
                blocks = np.array(blocks)
                
                # Split the blocks into n_splits parts
                splits = np.array_split(blocks, self.n_splits)
                
                # Test blocks for this split
                test_blocks = splits[split_idx]
                # Training blocks are all other blocks
                train_blocks = np.concatenate(splits[:split_idx] + splits[split_idx + 1:])
            
                # Find the indices in the original array corresponding to these blocks
                test_block_indices = [block_sess_dict_rev[(block, sess_name)] for block in test_blocks]
                train_block_indices = [block_sess_dict_rev[(block, sess_name)] for block in train_blocks]

                test_idx.extend(np.concatenate(test_block_indices))
                train_idx.extend(np.concatenate(train_block_indices))
                
            yield np.array(train_idx), np.array(test_idx)
        
       
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits