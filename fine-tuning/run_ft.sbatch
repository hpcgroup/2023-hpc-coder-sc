#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH -t 11:00:00
#SBATCH -A m2404
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --qos=regular
#SBATCH --ntasks-per-node=4
#SBATCH --output=run_ft.out

# Getting number of nodes and GPUs
NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))

# ENV variables for torch.distributed
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=16

# ENV variables for fast NCCL on perlmutter
# remove for other clusters
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1

# hf env variables, remove/change if needed
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"


module load pytorch
source ${SCRATCH}/axonn_venv/bin/activate

SCRIPT="python -u ft.py --dtype bf16 --global-batch-size 128 --gradient-acc-steps 2 --log-interval 1 --sequence-length 8192 --use-flash-attention"

#Uncomment if you want to check max memory usage
#SCRIPT="$SCRIPT --check-max-mem-usage"

run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh $SCRIPT"

echo $run_cmd
eval $run_cmd
