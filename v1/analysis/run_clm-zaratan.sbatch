#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -J train-causal
#SBATCH -A bhatele-lab-cmsc
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=196608
#SBATCH --mail-type=FAIL

#DATASET="hpcgroup/hpc-source"
DATASET="hpcgroup/omp-for-loops"
#MODEL="hpcgroup/polycoder-hpc-source"
CACHE_DIR="/scratch/zt1/project/bhatele-lab/user/dnicho/.cache/huggingface"
OUTPUT_DIR="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/gpt-neo-omp-for-loops-ckpt"
DSCONFIG="./ds_config_zero2.json"
HUB_TOKEN="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"

MAX_STEPS="50000"
MAX_CHECKPOINTS="5"

module load python/3.8.12/zen2 git-lfs/zen2/3.1.2 openmpi/4.1.1/gcc/9.4.0/zen2 cuda/11.6.2/gcc
source .env/bin/activate

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
#export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${CACHE_DIR}"

echo "device(s): $CUDA_VISIBLE_DEVICES"

deepspeed run_clm.py \
    --model_name_or_path hpcgroup/gpt-neo-hpc-source \
    --tokenizer_name ./hpc-tok \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATASET \
    --validation_split_percentage 5 \
    --cache_dir $CACHE_DIR \
    --optim adamw_torch \
    --fp16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --seed 42 \
    --do_eval \
    --do_train \
    --deepspeed $DSCONFIG \
    --num_train_epochs 3 \
    --save_steps 500 \
    --save_total_limit $MAX_CHECKPOINTS \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --max_eval_samples 1000 \
    --logging_steps 50 \
    --log_level passive 


    #--use_auth_token \
    #--push_to_hub \
    #--hub_model_id $MODEL \
    #--hub_private_repo \
    #--hub_token $HUB_TOKEN
