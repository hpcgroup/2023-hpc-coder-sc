#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 04:00:00
#SBATCH -J train-causal
#SBATCH -A bhatele-lab-cmsc
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --mem=16384
#SBATCH --mail-type=FAIL

# run params
MODEL="hpcgroup/gpt-neo-hpc-source"
TOKENIZER="hpcgroup/gpt-neo-hpc-source"
PROMPTS="codegen_tests.json"
OUTPUT="gpt-neo-results.jsonl"
NUM_SAMPLES="50"
MIN_LEN="25"
MAX_LEN="512"
TOP_K="50"
TOP_P="0.93"
TEMPERATURES="0.2 0.4 0.6 0.8"
MAX_SEQ_LENGTH="1024"
BATCH_SIZE="1"

# environment setup
DEVICE="$CUDA_VISIBLE_DEVICES"
CACHE_DIR="/scratch/zt1/project/bhatele-lab/user/dnicho/.cache/huggingface"
module load python/3.8.12/zen2 git-lfs/zen2/3.1.2 openmpi/4.1.1/gcc/9.4.0/zen2 cuda/11.6.2/gcc
source /home/dnicho/code-ml/analysis/.env/bin/activate
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
export HF_HOME="${CACHE_DIR}"

export OMP_NUM_THREADS="2"
export TOKENIZERS_PARALLELISM="true"

echo "device(s): $CUDA_VISIBLE_DEVICES"

if [ -f $OUTPUT ]; then
    rm $OUTPUT
fi

# run script
python codegen_tests.py \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --input $PROMPTS \
    --output $OUTPUT \
    --cache-dir $CACHE_DIR \
    --num-samples $NUM_SAMPLES \
    --min-len $MIN_LEN \
    --max-len $MAX_LEN \
    --top-k $TOP_K \
    --top-p $TOP_P \
    --temperatures $TEMPERATURES \
    --max-sequence-length $MAX_SEQ_LENGTH \
    --batch-size $BATCH_SIZE \
    --device $DEVICE 