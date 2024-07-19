#!/bin/bash
# Initial setup for training requires some work to be done on login node. Namely the dataset has to be copied
# to scratch space for the job to access it.
# THIS SCRIPT IS SPECIFIC TO ZARATAN.
# author: Daniel Nichols
# date: October 2022

SRC="/afs/shell.umd.edu/project/bhatele-lab/shared/hpc-llms/data/dataset.jsonl"
DST="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/dataset.json"

# copy dataset
echo "Copying dataset..."
cp ${SRC} ${DST}
if [ $? -ne 0 ]; then
      echo "Error copying dataset!"
fi

# create huggingface cache
echo "Copying huggingface cache..."
cp -r /afs/shell.umd.edu/project/bhatele-lab/user/dnicho/.cache/huggingface \
      /scratch/zt1/project/bhatele-lab/user/dnicho/.cache/
if [ $? -ne 0 ]; then
      echo "Error copying huggingface cache!"
fi

# move save dirs
echo "Copying saved models..."
cp -r /afs/shell.umd.edu/project/bhatele-lab/user/dnicho/code-ml/models \
    /scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/
if [ $? -ne 0 ]; then
      echo "Error copying saved models!"
fi
