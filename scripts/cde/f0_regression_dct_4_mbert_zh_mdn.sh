#!/bin/bash

# Get a timestamp for unique log file names
timestamp=$(date +"%Y%m%d_%H%M%S")

# Define log files with the timestamp
output_log="/home/user/ding/Projects/Prosody/logs/mdn_dct_4_mbert_zh_${timestamp}.out"
error_log="/home/user/ding/Projects/Prosody/logs/mdn_dct_4_mbert_zh_${timestamp}.err"


# Send some noteworthy information to the output log
echo "Running on node: $(hostname)" > "$output_log"
echo "In directory:    $(pwd)" >> "$output_log"
echo "Starting on:     $(date)" >> "$output_log"

# Set the GPU to use (GPU 1 in this case)
export CUDA_VISIBLE_DEVICES=4

echo $CUDA_VISIBLE_DEVICES

# Binary or script to execute
nohup python src/train.py experiment=cde/f0_regression_dct_4_mbert_zh_mdn >> "$output_log" 2>> "$error_log" &

# Send more noteworthy information to the output log
echo "Finished at:     $(date)" >> "$output_log"

# End the script with exit code 0
exit 0