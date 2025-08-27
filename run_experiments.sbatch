#!/bin/bash -l
#
# This is an example Slurm submission script to run the experiments.
#
# USAGE:
#
# 1. To run a sweep of all optimizers for a given model (e.g., 'mlp'):
#    sbatch --export=ALL,MODEL='mlp' run_experiments.sbatch
#
# 2. To run a single experiment (e.g., 'adam_hd' on 'mlp'):
#    sbatch --export=ALL,MODEL='mlp',OPTIMIZERS='adam_hd' --array=0 run_experiments.sbatch
#

# --- Slurm Directives ---
#SBATCH --job-name=hypergrad-exp      # Job name
#SBATCH --output=logs/hypergrad_%A_%a.out # Standard output and error log (%A: job ID, %a: array task ID)
#SBATCH --nodes=1                     # Run on a single node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=16G                     # Job memory request
#SBATCH --time=00:20:00               # Time limit hrs:min:sec
#SBATCH --gres=gpu:p100:1             # Request 1 GPU for the job


# --- Environment Setup ---
# This section might need changes depending on your cluster's configuration.
echo "Setting up environment..."
module purge
source /ibex/user/liut0c/opt/miniforge3/bin/activate # Activate your Python virtual environment
mamba activate hypergrad
module load cuda

# --- Experiment Logic ---
# Define optimizer configurations to sweep through. This can be overridden from the command line.
# OPTIMIZERS=${OPTIMIZERS:-"sgd sgdn adam sgd_hd sgdn_hd adam_hd"}
# read -r -a OPTIMIZERS_ARRAY <<< "$OPTIMIZERS"

# CURRENT_OPTIMIZER=${OPTIMIZERS_ARRAY[$SLURM_ARRAY_TASK_ID]}

# echo "Starting job $SLURM_JOB_ID, array task $SLURM_ARRAY_TASK_ID"
# echo "Model: $MODEL, Optimizer: $CURRENT_OPTIMIZER"

# # NOTE: The training script 'train.py' is not included in the repository.
# # You will need to create one that accepts arguments for model, optimizer, etc.,
# # and saves results to CSV files as expected by 'plot.py'.
# python train.py --model "$MODEL" --method "$CURRENT_OPTIMIZER" --epochs 50 --dir "results" --save --cuda --workers 3
python train.py --cuda --workers 3 --model logreg --method sgd --save --epochs 10 --alpha_0 0.001 --beta 0.001
