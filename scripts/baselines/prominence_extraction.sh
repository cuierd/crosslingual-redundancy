#!/bin/bash

# Get a timestamp for unique log file names
timestamp=$(date +"%Y%m%d_%H%M%S")

# Define log files with the timestamp
output_log="/Users/cui/Documents/uzh/PhD/Projects/Prosody/crosslingual-redundancy/logs/extract_prominence_en_${timestamp}.out"
error_log="/Users/cui/Documents/uzh/PhD/Projects/Prosody/crosslingual-redundancy/logs/extract_prominence_en_${timestamp}.err"


# Send some noteworthy information to the output log
echo "Running on node: $(hostname)" > "$output_log"
echo "In directory:    $(pwd)" >> "$output_log"
echo "Starting on:     $(date)" >> "$output_log"

# Binary or script to execute
nohup python src/extraction_prominence.py >> "$output_log" 2>> "$error_log" &

# Send more noteworthy information to the output log
echo "Finished at:     $(date)" >> "$output_log"

# End the script with exit code 0
exit 0
