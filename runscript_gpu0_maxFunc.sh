#!/bin/bash

for i in 0 1
do 
     echo Current Num "$i"
     echo Run Code "100k"
     CUDA_VISIBLE_DEVICES=3
     python tf_HSM_main_MaxFunc_100k_upgraded.py REGION=1 LR=1e-2 RESTART_TRIAL=$i SEED=$i ITERATIONS=100000
done
