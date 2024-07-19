# HPC LLM Training

> [!WARNING]
> This directory contains the old hpc-coder-v1 scripts. See the [main repo](/)
> for more up-to-date scripts, data, and models.

This repo contains scripts for training LLMs on HPC source code data.
It is organized as follows:

- ***data:*** scripts and utilities for collecting and preprocessing the dataset [[README]](data/README.md)
- ***analysis:*** scripts related to analyzing dataset and training LLM [[README]](analysis/README.md)

To overview of the workflow from start to finish is as follows.
Use `data/collect-repo-metadata.py` and `data/edit-metadata.py` to create dataset of GitHub repositories, if desired,
otherwise use the existing `repos-gt3.csv` dataset.
Run `data/clone-repos.py` to clone the repositories to a desired location.
`data/collect-dataset.py` script can then be used to create a json lines dataset with all the textual data.
The `analysis/run_clm-*.sbatch` scripts can then be used to train the models on the data.

## Notes and Misc.

*Weird bug fix #1:*
I had to change the default value of `max_workers` from 64 to 32 in the parameter list of 
`_get_origin_metadata_locally_or_by_urls` in `datasets/data_files.py#L708`. 
Zaratan's CPUs have 64 cores, but tqdm errors trying to start 64 threads for some reason.
Similar to this I also generally have to `export TOKENIZERS_PARALLELISM=false` to prevent 
huggingface from spinning up threads in forked processes.

*Weird bug fix #2:*
I had to change the following lines in `torch/distributed/distributed_c10d.py`, line 2068.
The PolyCoder model seems to output non-contiguous logits
and there is a current bug in PyTorch where the NCCL backend errors if passed
non-contiguous tensors to `all_gather`. This bug is documented [here](https://github.com/pytorch/pytorch/issues/73515)
and [here](https://github.com/pytorch/pytorch/pull/75276).
With a future release of torch this likely won't be necessary.

```python
work = default_pg.allgather([tensor_list], [tensor])
# to -->
work = default_pg.allgather([[t.contiguous() for t in tensor_list]], [tensor.contiguous()])
```