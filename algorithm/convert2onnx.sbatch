#!/bin/bash -l

#SBATCH --job-name=training_job                     # Job name
#SBATCH --output=logs.txt  # Output log file
#SBATCH --error=error.txt   # Error log file
#SBATCH --nodes=1                                      # Use a single node
#SBATCH --ntasks=1                                     # Number of tasks (1 for a single process)
#SBATCH --gres=gpu:v100:1                              # Request 1 V100 GPU
#SBATCH --partition=v100	                            # Use the V100 partition
#SBATCH --time=6:00:00                                # Set a time limit of 12 hours
#SBATCH --export=NONE                                  # Do not export environment variables
	
unset SLURM_EXPORT_ENV                                         # Unset SLURM_EXPORT_ENV to avoid conflicts

# Load the required Python module
module purge
module load python/3.12-conda

# Activate your virtual environment
source ../../insightface/recognition/arcface_torch/.pal/bin/activate

# Run your Python script with all the required parameters
python torch2onnx.py \
    --config configs/transfer.yaml \
    --input runs/syn/mbv2-w0.35/6b+6w/sgd_qas/checkpoint/ckpt.best.pth \
    --output face_model.onnx \
    --simplify > convert.txt 2>&1