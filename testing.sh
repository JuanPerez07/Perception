#!/bin/bash

# Check different batches to check the fluctuation 
# of the Mean Reprojection Error
MIN_BATCH=5
for i in {0..10}  # Loops 10 times
do
    batch=$((i + MIN_BATCH))
    python3 main.py $batch >> mpe_results.txt
done
