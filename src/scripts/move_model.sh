#!/bin/bash
#SBATCH --chdir=scratch/CoronalHoles/src/polished_pipeline
#SBATCH -o ../../logs/move.out
#SBATCH -e ../../logs/move.err
#SBATCH -t 0:15:00           # time limit 
#SBATCH -N 1                  # number of nodes
#SBATCH -n 1                  # number of tasks (CPU cores)
#SBATCH --mem=4G           # memory 
#SBATCH -J move       # job name

# Activate your venv
source ../../.venv/bin/activate


# Run your inference command
python3 -m move_model_location