#!/bin/bash
# Data cannot persist in scratch space after training job. Clear these out from login node.
# THIS SCRIPT IS SPECIFIC TO ZARATAN.
# author: Daniel Nichols
# date: October 2022

# remove dataset
echo "Removing dataset..."
rm /scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/dataset.json
if [ $? -ne 0 ]; then
    echo "Error deleting dataset!"
fi

# remove cache
echo "Saving and removing cache..."
cp -r /scratch/zt1/project/bhatele-lab/user/dnicho/.cache/huggingface \
    /afs/shell.umd.edu/project/bhatele-lab/user/dnicho/.cache
if [ $? -eq 0 ]; then
    rm -r /scratch/zt1/project/bhatele-lab/user/dnicho/.cache/huggingface
else
    echo "Error copying .cache directory!"
fi

# move model checkpoints
echo "Saving and removing saved models..."
cp -r /scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/models \
    /afs/shell.umd.edu/project/bhatele-lab/user/dnicho/code-ml/
if [ $? -eq 0 ]; then
    rm -r /scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/models
else
    echo "Error copying saved models!"
fi
