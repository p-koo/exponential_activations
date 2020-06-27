#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python3 task3_train_model.py
CUDA_VISIBLE_DEVICES=0 python3 task3_attribution_scores.py
