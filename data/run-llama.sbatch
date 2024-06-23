#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH -t 24:00:00
#SBATCH -A m2404
#SBATCH -C gpu&hbm80g
#SBATCH -q regular

source ~/dbrx/.env/bin/activate

python -u  llama_script.py
