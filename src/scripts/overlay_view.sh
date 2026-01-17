#!/bin/bash
#SBATCH --chdir=scratch/CoronalHoles/src/polished_pipeline
#SBATCH -o ../../logs/overlay.out
#SBATCH -e ../../logs/overlay.err
#SBATCH -t 0:15:00           # time limit 
#SBATCH -N 1                  # number of nodes
#SBATCH -n 1                  # number of tasks (CPU cores)
#SBATCH --mem=4G           # memory 
#SBATCH -J overlay       # job name

# Activate your venv
source ../../.venv/bin/activate


# Run your inference command
python3 -m stats_updated.overlay_view_scratch \
--aia "CoronalHoles/src/polished_pipeline/PREDICTIONS/CHASM1111/aia/2022/193/2022-07-10.fits" \
--blue "CoronalHoles/src/polished_pipeline/PREDICTIONS/CHASM1111/chronnos/inputs/map/512/2022-07-10.npy" \
--red "CoronalHoles/src/polished_pipeline/PREDICTIONS/CHASM1111/chronnos/inputs/map/512/2022-07-10.npy"