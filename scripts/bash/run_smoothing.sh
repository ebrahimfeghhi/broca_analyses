#!/bin/bash

# Define the directory
DIR="/home2/ebrahim/neural_seq_decoder/scripts/"

# Change to the specified directory
cd "$DIR" || { echo "Directory $DIR not found"; exit 1; }

# Define the list of alpha values
alphas=(0.02 0.04)

# Loop through each alpha value
for alpha in "${alphas[@]}"; do

    echo "Running with alpha=$alpha"
    python run_reg.py --exist_ok --load_X_by_sess --val_sess --X "ba44_em_$alpha" --fname "ba44_em_$alpha"

done
