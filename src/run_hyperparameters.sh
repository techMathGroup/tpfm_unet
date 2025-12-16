#!/usr/bin/env bash

# Define hyperparameter values
#BASE_FILTERS_LIST=(16 32 48)
#DEPTH_LIST=(3 4 5)
#
#for bf in "${BASE_FILTERS_LIST[@]}"; do
#    for d in "${DEPTH_LIST[@]}"; do
#        echo "Running with base_filters=$bf depth=$d"
#        python src/train.py model.base_filters="$bf" model.depth="$d"
#    done
#done

LR_LIST=(0.002 0.001 0.0008 0.0006 0.0004 0.0002 0.0001)

for lr in "${LR_LIST[@]}"; do
    echo "Running with learning_rate=$lr"
    python src/train.py model.learning_rate="$lr" experiment_name="learning_rate"
done