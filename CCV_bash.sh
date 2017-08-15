#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -t 00:05:00
#SBATCH -o sum.out

# Load CUDA module
module load cuda

# Compile CUDA program and run
nvcc -arch sm_20 vecadd.cu -o vecadd
./vecadd
