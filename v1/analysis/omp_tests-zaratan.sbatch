#!/bin/bash
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -J train-causal
#SBATCH -A bhatele-lab-cmsc
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --mem=16384
#SBATCH --mail-type=FAIL

MODEL="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/gpt-neo-omp-for-loops-ckpt"
TOKENIZER="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/gpt-neo-omp-for-loops-ckpt"
DEVICE="$CUDA_VISIBLE_DEVICES"
CACHE_DIR="/scratch/zt1/project/bhatele-lab/user/dnicho/.cache/huggingface"
TEMPERATURE="0.2"

module load python/3.8.12/zen2 git-lfs/zen2/3.1.2 openmpi/4.1.1/gcc/9.4.0/zen2 cuda/11.6.2/gcc
source .env/bin/activate
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
export HF_HOME="${CACHE_DIR}"

echo "device(s): $CUDA_VISIBLE_DEVICES"


python omp_tests.py \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --cache-dir $CACHE_DIR \
    --temperature $TEMPERATURE \
    --num-samples 10 \
    --device $DEVICE 
