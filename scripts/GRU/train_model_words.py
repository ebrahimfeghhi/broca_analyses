from sklearn.model_selection import KFold
import torch
import numpy as np
import pickle
import os
import time
from model import GRUDecoder
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NeuralDataDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # or torch.float32 if y is continuous

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def trainModel(args, datadir='/data/LLMs/willet/delay_decode/moses_data/'):
    
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    testLoss = []
    testCER = []
    
    X = np.load(f'{datadir}X_np.npy')
    y = np.load(f'{datadir}y_np.npy')
    block_nums = np.load(f'{datadir}blockNums.npy')

    # Outer cross-validation loop: iterate over blocks for testing
    for test_blocks in range(0, len(np.unique(block_nums)), 2):
        
        test_block_idxs = np.where((block_nums == test_blocks) | (block_nums == test_blocks + 1))[0]
        train_block_idxs = np.setdiff1d(np.arange(len(block_nums)), test_block_idxs)

        X_test, y_test = X[test_block_idxs], y[test_block_idxs]
        X_train, y_train = X[train_block_idxs], y[train_block_idxs]
        
        # Instantiate the dataset
        dataset = NeuralDataDataset(X_train, y_train)
        
        data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0)

        model = GRUDecoder(
            neural_dim=args["nInputFeatures"],
            n_classes=args["nClasses"],
            hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"],
            dropout=args["dropout"],
            device=device,
            strideLen=args["strideLen"],
            kernelLen=args["kernelLen"],
            gaussianSmoothWidth=args["gaussianSmoothWidth"],
            bidirectional=args["bidirectional"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=args["l2_decay"],
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args["lrEnd"] / args["lrStart"],
            total_iters=args["nBatch"],
        )
        loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # Training loop on the fold
        for batch in range(args["nBatch"]):
            model.train()

            X_batch, y_batch = X_train, y_train
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Noise augmentation, if applicable
            if args["whiteNoiseSD"] > 0:
                X_batch += torch.randn(X_batch.shape, device=device) * args["whiteNoiseSD"]

            if args["constantOffsetSD"] > 0:
                X_batch += torch.randn([X_batch.shape[0], 1, X_batch.shape[2]], device=device) * args["constantOffsetSD"]

            # Compute prediction and loss
            pred = model.forward(X_batch)
            loss = loss_ctc(torch.permute(pred.log_softmax(2), [1, 0, 2]), y_batch)
            loss = torch.sum(loss)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        # Evaluate the model on the test set
        with torch.no_grad():
            model.eval()
            test_pred = model.forward(X_test.to(device))
            test_loss = loss_ctc(torch.permute(test_pred.log_softmax(2), [1, 0, 2]), y_test.to(device))
            testLoss.append(test_loss.cpu().detach().numpy())

            # You can also compute CER (character error rate) if needed and add it to testCER

        print(f"Test blocks {test_blocks}-{test_blocks+1}, Test Loss: {test_loss:>7f}")

    # Saving results
    with open(args["outputDir"] + "/finalStats", "wb") as file:
        pickle.dump({"testLoss": testLoss, "testCER": testCER}, file)


