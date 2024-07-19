#!/bin/bash
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -J evaluate-causal
#SBATCH -A bhatele-lab-cmsc
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=131072

#DATASET="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/dataset.json"
DATASET="daniellnichols/hpc-source"
MODEL="daniellnichols/gpt-neo-hpc-source"
CACHE_DIR="/scratch/zt1/project/bhatele-lab/user/dnicho/.cache/huggingface"
OUTPUT_DIR="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/gpt-neo-hpc-ckpt"
LOG_DIR="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/logging"
DSCONFIG="./ds_config_zero3.json"

MAX_STEPS="10000"

module load python/3.8.12/zen2 git-lfs/zen2/3.1.2 openmpi/4.1.1/gcc/9.4.0/zen2 cuda/11.6.2/gcc
source .env/bin/activate

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"

echo "device(s): $CUDA_VISIBLE_DEVICES"

deepspeed run_clm.py \
    --model_name_or_path ${OUTPUT_DIR}/checkpoint-3500 \
    --tokenizer_name ./hpc-tok \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATASET \
    --validation_split_percentage 5 \
    --cache_dir $CACHE_DIR \
    --optim adamw_torch \
    --fp16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --seed 42 \
    --do_eval \
    --deepspeed $DSCONFIG \
    --max_eval_samples 250 \
    --logging_steps 50 \
    --log_level passive 
