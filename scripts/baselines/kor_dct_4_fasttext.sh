#!/bin/bash

#SBATCH --output=/swdata/yin/Cui/prosody/fasttext_logs/cv-hyperparam-%j-dct_4_ft_kor10.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/swdata/yin/Cui/prosody/fasttext_logs/cv-hyperparam-%j-dct_4_ft_kor10.err  # where to store error messages

#SBATCH -p gpu
#SBATCH -t 10:00:00 
#SBATCH -N 1                  
#SBATCH --gres=gpu:1          
#SBATCH --mem=12G             # Increase memory (adjust as needed)

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Command to execute Python script
python3 src/models/baselines/train_vector_fasttext_cv.py \
                        --NUM_RUNS 12 \
                        --INPUT_SIZE 300 \
                        --DEVICE cuda \
                        --NUM_MIX 10 \
                        --OUTPUT_SIZE 8 \
                        --EPOCHS 50 \
                        --BATCH_SIZE 256 \
                        --DATA_DIR "/swdata/yin/Cui/prosody/languages/kor/cache/" \
                        --FASTTEXT_PATH "/swdata/yin/Cui/prosody/data/models/fastText/cc.kor.300.bin" \
                        --EMB_MODEL "fasttext" \

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
