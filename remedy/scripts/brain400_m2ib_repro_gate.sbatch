#!/bin/bash -l
#SBATCH --job-name=m2ib_gate
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/home/hpc/rlvl/rlvl178v/prl_medclipsam/beyond-average-dice/remedy/logs/m2ib_gate_%j.out
#SBATCH --error=/home/hpc/rlvl/rlvl178v/prl_medclipsam/beyond-average-dice/remedy/logs/m2ib_gate_%j.err

set -euo pipefail

cd /home/hpc/rlvl/rlvl178v/prl_medclipsam

export HF_HOME=/home/hpc/rlvl/rlvl178v/prl_medclipsam/medclipsam_hf_home
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/home/hpc/rlvl/rlvl178v/.conda/envs/medclipsam/bin/python \
    beyond-average-dice/remedy/scripts/brain400_m2ib_repro_gate.py
