import os
import pickle
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from model import GRUDecoder

def compute_accuracy(y, preds):
    """
    Computes accuracy given the true labels and predicted logits.

    Args:
        y (torch.Tensor): Ground truth labels of shape (batch_size,)
        preds (torch.Tensor): Predicted logits of shape (batch_size, n_classes)

    Returns:
        float: Accuracy as a percentage
    """
    # Get the predicted class labels by applying argmax on the logits
    _, predicted = torch.max(preds, dim=1)  # predicted has shape (batch_size,)

    # Compute the number of correct predictions
    correct_predictions = (predicted == y).sum().item()

    # Compute accuracy
    accuracy = correct_predictions / y.size(0)  # y.size(0) is the batch size

    return accuracy

def trainModel(trainLoader, testLoader, args):

    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda:2"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

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

    loss_ce = torch.nn.CrossEntropyLoss()
    
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
    
    for batch in range(args["nBatch"]):
        
        model.train()

        X, y = next(iter(trainLoader))
        X, y  = (
            X.to(device),
            y.to(device))

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Compute prediction error
        pred = model.forward(X) # batch_size x (X_len - model.kernelLen)/model.strideLen x n_classes
        
        y_ce = y[:, 0].long() # same class label repeated across time
        pred_last = pred[:, -1]  # take the prediction at the last timestep

        loss = loss_ce(pred_last, y_ce)
        
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Eval
        if batch % 100 == 0 or batch==args["nBatch"]-1:
            
            with torch.no_grad():
                
                model.eval()
                
                for X, y in testLoader:
                    
                    X, y = (
                        X.to(device),
                        y.to(device)
                    )

                    pred = model.forward(X)
                    
                    pred_last = pred[:, -1]
                    y_ce = y[:, 0].long()
                    
                    accuracy = compute_accuracy(y_ce, pred_last)
                
            print("Batch: ", batch)
            print(f"Accuracy: {accuracy}")
    
    return accuracy, y_ce, pred_last  
                    
                    