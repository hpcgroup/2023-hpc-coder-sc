#!/bin/bash
# Wrapper script for repo-plots.py. This will generate all of the plots describing the repos in the dataset.
# author: Daniel Nichols
# date: October 2022

FIGS=figs
if [ $# -gt 1 ]; then
    DATASET_PATH="$2"
else
    DATASET_PATH="repos-gt3.csv"
fi

mkdir -p $FIGS

python repo-plots.py -d $DATASET_PATH \
    --languages $FIGS/languages-hist.png \
    --sizes $FIGS/sizes-hist.png \
    --stars $FIGS/stars-hist.png \
    --watchers $FIGS/watchers-hist.png \
    --forks $FIGS/forks-hist.png \
    --tags $FIGS/tags-wordcloud.png \
    --extensions $FIGS/extensions-hist.png \
    --loc $FIGS/loc-hist.png
