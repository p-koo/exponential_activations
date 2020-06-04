#!/bin/bash

dirpath="../results/task4/conv_filters"
tomtom -evalue -thresh 0.1 -o $dirpathbasset_relu $dirpathbasset_relu.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o $dirpathbasset_exponential $dirpathbasset_exponential.meme ../data/JASPAR_CORE_2016_vertebrates.meme

