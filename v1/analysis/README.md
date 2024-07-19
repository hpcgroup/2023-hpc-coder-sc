# analysis

This directory contains scripts for training LLMs on the dataset.

## run_clm scripts
`run_clm.py` is an extension of the default huggingface script for training
causal language models.
We use it here to train the different models on the dataset.
They are called by the sbatch scripts (i.e. `run_clm-zaratan.sbatch`), which 
also take care of the environment setup.
These are fairly specific to my Zaratan environment.

## generate_text.py
A script for using the model to generate code based on a prompt.
Run `python generate_text.py -h` to see how to pass args.

## parse_losses.py
If not using tensorboard or other monitoring software, then you can use 
this to parse the training output text for loss/accuracy values.
Simply give it a list of files to parse and where to output the training and
validation csv files.
Run `python parse_losses.py -h` for more options.

## plot_training_data.py
Uses the CSVs output by `parse_losses.py` to create training and validation
loss+perplexity curves.
By default this will save them in the `figs/` directory.

## train-tokenizer.py
Trains a tokenizer on the dataset.
Taken from HuggingFace repo (see script for URL and how to run).

