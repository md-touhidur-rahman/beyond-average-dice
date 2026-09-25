#!/bin/bash -l
#SBATCH --job-name=stoch_p1
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/home/hpc/rlvl/rlvl178v/prl_medclipsam/beyond-average-dice/remedy/logs/stoch_p1_%j.out
#SBATCH --error=/home/hpc/rlvl/rlvl178v/prl_medclipsam/beyond-average-dice/remedy/logs/stoch_p1_%j.err

set -euo pipefail

cd /home/hpc/rlvl/rlvl178v/prl_medclipsam

export HF_HOME=/home/hpc/rlvl/rlvl178v/prl_medclipsam/medclipsam_hf_home
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/home/hpc/rlvl/rlvl178v/.conda/envs/medclipsam/bin/python \
  beyond-average-dice/remedy/scripts/run_stochastic_pilot_stage1.py
