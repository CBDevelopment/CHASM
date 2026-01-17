#!/bin/bash
#SBATCH --chdir=scratch/CoronalHoles/src/polished_pipeline
#SBATCH -o ../../logs/chasm967_predict.out
#SBATCH -e ../../logs/chasm967_predict.err
#SBATCH -t 12:00:00           # time limit 
#SBATCH -N 1                  # number of nodes
#SBATCH -n 1                  # number of tasks (CPU cores)
#SBATCH --gres=gpu:1          # request 1 GPU
#SBATCH --mem=8G           # memory 
#SBATCH -J chasm967_predict       # job name

# Activate your venv
source ../../.venv/bin/activate


# Run your inference command
python3 -m predict ../../models/final_model_CHASM-967_moved2.pt PREDICTIONS/CHASM967