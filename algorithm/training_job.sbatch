#!/bin/bash -l

#SBATCH --job-name=training_job                     # Job name
#SBATCH --output=logs.txt  # Output log file
#SBATCH --error=error.txt   # Error log file
#SBATCH --nodes=1                                      # Use a single node
#SBATCH --ntasks=1                                     # Number of tasks (1 for a single process)
#SBATCH --gres=gpu:v100:2                              # Request 1 V100 GPU
#SBATCH --partition=v100	                            # Use the V100 partition
#SBATCH --time=12:00:00                                # Set a time limit of 12 hours
#SBATCH --export=NONE                                  # Do not export environment variables
	
unset SLURM_EXPORT_ENV                                         # Unset SLURM_EXPORT_ENV to avoid conflicts

# Load the required Python module
module purge
module load python/3.12-conda

# Activate your virtual environment
source ../../insightface/recognition/arcface_torch/.pal/bin/activate

# Run your Python script with all the required parameters
python train_cls.py   configs/transfer.yaml --run_dir runs-128/face/mcunet-5fps/sparse_100kb/sgd_qas_nomom \
    --net_name mcunet-5fps  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom \
    --enable_backward_config 1 --n_bias_update 22 --manual_weight_idx 21-24-27-30-36-39 \
    --weight_update_ratio 1-1-1-1-0.125-0.25 > train_sparse_update_face.txt 2>&1