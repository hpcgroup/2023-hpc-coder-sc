#!/bin/bash
# Copy repos recursively from one directory to another
# author: Daniel Nichols
# date: October 2022

if [ $# -ne 2 ]; then
    printf "usage: %s <source-root> <dest-root>\n" "$0"
    exit 1
fi

SRCROOT="$1"
DSTROOT="$2"

echo "cp -r $SRCROOT $DSTROOT"
cp -r $SRCROOT $DSTROOT
