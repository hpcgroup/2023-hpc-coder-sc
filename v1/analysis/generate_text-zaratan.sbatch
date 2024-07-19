#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -J model-inference
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --mem=16384

module load python/3.8.12/zen2 git-lfs/zen2/3.1.2 openmpi/4.1.1/gcc/9.4.0/zen2 cuda/11.6.2/gcc
source .env/bin/activate
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"

echo "device(s): $CUDA_VISIBLE_DEVICES"

#MODEL="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/polycoder-hpc-source-ckpt/checkpoint-200"
MODEL="hpcgroup/gpt2-medium-hpc-source"
TOKENIZER="./hpc-tok"
DEVICE="$CUDA_VISIBLE_DEVICES"
PROMPT="/* saxpy -- multiply scalar float a by vector x and add to y */ void saxpy(float *x, float *y, float a, int N) { for (int i = 0; i < N; i++) {"
#PROMPT="cuda-prompt.txt"

echo "Prompt: \"${PROMPT}\""

python generate_text.py \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --device $DEVICE \
    --min-len 25 \
    --max-len 350 \
    --num-samples 15 \
    --text "${PROMPT}"
