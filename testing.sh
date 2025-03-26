#!/bin/bash

# Check different batches to check the fluctuation 
# of the Mean Reprojection Error

for i in {2..15}  # Loops from 2 to 15
do
    python3 main.py $i >> mpe_results.txt
done
