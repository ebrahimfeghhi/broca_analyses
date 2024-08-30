#!/bin/bash


# Define the directory
DIR="/home2/ebrahim/neural_seq_decoder/scripts/"

# Change to the specified directory
cd "$DIR" || { echo "Directory $DIR not found"; exit 1; }

alphas=(0.1 0.2 0.4 0.6)
delays=(10 25 50 75 100 150 200 300)

# Loop through each alpha value first
for alpha in "${alphas[@]}"; do
    # Loop through each delay value
    for delay in "${delays[@]}"; do

        echo "Running with alpha=$alpha and delay=$delay"
        python run_reg.py --exist_ok --load_X_by_sess --val_sess --X "ba44_alpha_${alpha}_delay_$delay" --fname "ba44_alpha_${alpha}_delay_$delay" --device 2

    done
done