#!/bin/bash
#SBATCH --chdir=scratch/CoronalHoles/src/polished_pipeline
#SBATCH -o ../../logs/resizemag.out
#SBATCH -e ../../logs/resizemag.err
#SBATCH -t 1:00:00           # time limit 
#SBATCH -N 1                  # number of nodes
#SBATCH -n 1                  # number of tasks (CPU cores)
#SBATCH --mem=4G           # memory 
#SBATCH -J resizemag       # job name

# Activate your venv
source ../../.venv/bin/activate

# Run your inference command
python3 -m scratch.resize_mag