#!/bin/bash
# Helper script for calling collect-repo-metadata.py
# author: Daniel Nichols
# date: October 2022

MIN_STARS=3
TAGS="hpc mpi openmp proxy-application miniapp mini-app parallel-computing scientific-computing high-performance-computing computational-science"
LANGUAGES="c c++"

python3 collect-repo-metadata.py --min-stars $MIN_STARS --tags $TAGS \
	--languages $LANGUAGES --output repos.csv
