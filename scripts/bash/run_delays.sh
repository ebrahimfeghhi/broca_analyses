#!/bin/bash

# Define the directory
DIR="/home2/ebrahim/neural_seq_decoder/scripts/"

# Change to the specified directory
cd "$DIR" || { echo "Directory $DIR not found"; exit 1; }

#delays=(5 10 15 20 25 30 35 40 45 50 55 60 65 70 74)
delays=(150 200 300)

# Loop through each alpha value
for delay in "${delays[@]}"; do

    echo "Running with delay=$delay"
    python run_reg.py --exist_ok --load_X_by_sess --val_sess --X "ba44_delay_$delay" --fname "ba44_delay_$delay"

done
