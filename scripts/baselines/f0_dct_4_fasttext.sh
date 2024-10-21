#!/bin/bash

#SBATCH --output=/cluster/work/cotterell/cui/prosody/logs/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/cluster/work/cotterell/cui/prosody/logs/%j.err  # where to store error messages

#SBATCH -t 04:00:00 
#SBATCH -n 1   
#SBATCH --mem-per-cpu=64000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g 

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Command to execute Python script
python3 src/models/baselines/train_vector.py \
                        --NUM_RUNS 15 \
                        --INPUT_SIZE 300 \
                        --DEVICE cuda \
                        --OUTPUT_SIZE 8 \
                        --EPOCHS 50 \
                        --BATCH_SIZE 256 \
                        --DATA_DIR "/cluster/work/cotterell/gacampa/en/cache" \
                        --GLOVE_PATH "/om/user/luwo/projects/data/models/glove/glove.6B.300d.txt" \
                        --FASTTEXT_PATH "/cluster/work/cotterell/gacampa/MIT_prosody/models/fasttext_en.bin" \
                        --EMB_MODEL "fasttext" \

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
