#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH -t 20:00:00
#SBATCH -A m2404
#SBATCH -C gpu&hbm80g
#SBATCH -q regular

source ~/dbrx/.env/bin/activate

python -u dbrx_script.py
