#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python3 task1_train_log.py
bash task1_tomtom_log.py
python3 task1_filter_match_log.py

CUDA_VISIBLE_DEVICES=1 python3 task3_train_log.py
CUDA_VISIBLE_DEVICES=1 python3 task3_attribution_scores_log.py

CUDA_VISIBLE_DEVICES=1 python3 task5_train_model.py
CUDA_VISIBLE_DEVICES=1 python3 task6_train_model.py