#!/bin/bash
#SBATCH --chdir=scratch/CoronalHoles/src/polished_pipeline
#SBATCH -o ../../logs/gentables.out
#SBATCH -e ../../logs/gentables.err
#SBATCH -t 6:00:00           # time limit 
#SBATCH -N 1                  # number of nodes
#SBATCH -n 1                  # number of tasks (CPU cores)
#SBATCH --mem=16G          # memory 
#SBATCH -J gentables       # job name

# Activate your venv
source ../../.venv/bin/activate


# Run your inference command
python3 -m stats_updated.tables_from_raw_data