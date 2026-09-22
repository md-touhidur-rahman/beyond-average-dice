#!/bin/bash -l
#SBATCH --job-name=fs_sam
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/rlvl/rlvl178v/prl_medclipsam/MedCLIP-SAM/part2/logs/fs_sam_%j.out
#SBATCH --error=/home/hpc/rlvl/rlvl178v/prl_medclipsam/MedCLIP-SAM/part2/logs/fs_sam_%j.err

cd /home/hpc/rlvl/rlvl178v/prl_medclipsam

/home/hpc/rlvl/rlvl178v/.conda/envs/medclipsam/bin/python \
beyond-average-dice/scripts/brain_p20/run_failure_success_sam.py
