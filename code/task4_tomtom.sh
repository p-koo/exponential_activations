#!/bin/bash

dirpath="../results/task4/conv_filters"
tomtom -evalue -thresh 0.1 -o $dirpath/basset_relu $dirpath/basset_relu.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o $dirpath/basset_exponential $dirpath/basset_exponential.meme ../data/JASPAR_CORE_2016_vertebrates.meme

