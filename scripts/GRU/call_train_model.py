import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
device = 2

datadir = "/data/LLMs/willet/delay_decode/moses_data/"

X = np.load(f'{datadir}X.npy')
y = np.load(f'{datadir}y.npy')
block_nums = np.load(f'{datadir}blockNums.npy')
batchSize = 64
from train_model import trainModel
args = {}
args = {}
args['batchSize'] = 64 # number of words included on each minibatch 
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 256
args['nBatch'] = 2000 
args['nLayers'] = 4
args['nClasses'] = 51
args['nInputFeatures'] = 256 # spike pow + tx crossings
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0 # smooths data with gaussian kernel (40 ms std)
args['kernelLen'] = 14 # kernel size, which is how many time bins are stacked together and passed to RNN (14 = 280 ms)
args['bidirectional'] = True
args['l2_decay'] = 1e-5
args['strideLen'] = 4 # how many times forward the RNN skips forward every step 
args['outputDir'] = "/data/LLMs/willet/delay_decode/moses_data/run1/"

class NeuralDataDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # or torch.float32 if y is continuous

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        return (
            self.X[idx],
            self.y[idx].to(torch.int64), 
        )

def getDatasetLoaders(train_ds, test_ds, batchSize):


    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


# Outer cross-validation loop: iterate over blocks for testing
block_nums_unq_sorted = np.sort(np.unique(block_nums))

for seed in range(10):
    
    acc_folds = []
    targets_folds = []
    predictions_folds = []
        
    args['seed'] = seed
    
    for i in range(0, len(block_nums_unq_sorted), 1):
        
        print("block number: ", block_nums_unq_sorted[i])
        # find indices corresponding to test block
        test_block_idxs = np.where(block_nums == block_nums_unq_sorted[i]) 
        # remove these from the training indices 
        train_block_idxs = np.setdiff1d(np.arange(len(block_nums)), test_block_idxs)

        X_test, y_test = X[test_block_idxs], y[test_block_idxs]
        X_train, y_train = X[train_block_idxs], y[train_block_idxs]
        
        print(X_train.shape, X_test.shape) 
        
        train_ds = NeuralDataDataset(X_train, y_train)
        test_ds = NeuralDataDataset(X_test, y_test)
        
        train_loader, test_loader = getDatasetLoaders(train_ds, test_ds, batchSize)
        
        accuracy, targets, predictions = trainModel(train_loader, test_loader, args)
        
        acc_folds.append(accuracy)
        targets_folds.append(targets)
        predictions_folds.append(predictions)

    np.save(f"{args['outputDir']}/accuracy_{args['seed']}", acc_folds)
    np.save(f"{args['outputDir']}/targets_{args['seed']}", torch.stack(targets_folds).cpu().numpy())
    np.save(f"{args['outputDir']}/preds_{args['seed']}", torch.stack(predictions_folds).cpu().numpy())
        
    
        