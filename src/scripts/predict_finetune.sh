#!/bin/bash
#SBATCH --chdir=scratch/CoronalHoles/src/polished_pipeline
#SBATCH -o ../../logs/chasm1111_ft_1e-6_predict.out
#SBATCH -e ../../logs/chasm1111_ft_1e-6_predict.err
#SBATCH -t 6:00:00           # time limit 
#SBATCH -N 1                  # number of nodes
#SBATCH -n 1                  # number of tasks (CPU cores)
#SBATCH --gres=gpu:1          # request 1 GPU
#SBATCH --mem=8G           # memory 
#SBATCH -J chasm1111_ft_1e-6_predict       # job name

# Activate your venv
source ../../.venv/bin/activate


# Run your inference command
python3 -m predict ../../models/moved/no_bad_days_CHASM1111_finetuned_model1e-6_moved.pt PREDICTIONS/Finetuned/CHASM1111_finetuned_1e-6