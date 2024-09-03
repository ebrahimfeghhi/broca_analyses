#!/bin/bash

# Define the directory
DIR="/home2/ebrahim/neural_seq_decoder/scripts/"

# Change to the specified directory
cd "$DIR" || { echo "Directory $DIR not found"; exit 1; }

alphas=(0.03)
delay=(25 50 100 200)

# Loop through each alpha value first
for alpha in "${alphas[@]}"; do
    # Loop through each delay value
    for delay in "${delay[@]}"; do
        echo "Running with alpha=$alpha and delay=$delay"
        python run_reg.py --exist_ok --load_X_by_sess --val_sess --X "goaudio_ba44_alpha_${alpha}_delay_${delay}_150" --fname "ba44_alpha_${alpha}_delay_${delay}_150" --device 2
    done
done

goaudio_ba44_alpha_{selected_alpha}_delay_{delay_str}{sess}
