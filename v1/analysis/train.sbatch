#!/bin/bash
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -J hpc-llm-train
#SBATCH -A bhatele-lab-aac

#####STATCH -p gpu
#####STATCH --gres=gpu:a100

# config params
DATASET="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/dataset.jsonl"
TOKENIZER="./hpc-tok"
MODEL="gpt2"
SEQ_LENGTH="1024"
LM_TASK="causal"

if [ -d "/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/dataset-tokens" ]; then
    TOKENS="--load-tokens /scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/dataset-tokens"
else
    TOKENS="--save-tokens /scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/dataset-tokens"
fi

# setup environment
module load python/3.8.12/zen2
source .env/bin/activate
export HF_DATASETS_CACHE="/scratch/zt1/project/bhatele-lab/user/dnicho/.cache/huggingface/datasets"

# run job
python train.py --input $DATASET --log debug \
    --tokenizer $TOKENIZER  \
    ${TOKENS} \
    --model $MODEL \
    --lm-task $LM_TASK \
    --max-seq-length $SEQ_LENGTH 
