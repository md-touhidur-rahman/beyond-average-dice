#!/bin/bash -l
#SBATCH --job-name=text_mech
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/home/hpc/rlvl/rlvl178v/prl_medclipsam/beyond-average-dice/remedy/logs/text_mech_%j.out
#SBATCH --error=/home/hpc/rlvl/rlvl178v/prl_medclipsam/beyond-average-dice/remedy/logs/text_mech_%j.err

set -euo pipefail

cd /home/hpc/rlvl/rlvl178v/prl_medclipsam

export HF_HOME=/home/hpc/rlvl/rlvl178v/prl_medclipsam/medclipsam_hf_home
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/home/hpc/rlvl/rlvl178v/.conda/envs/medclipsam/bin/python \
  beyond-average-dice/remedy/scripts/analyze_text_embedding_mechanism.py
