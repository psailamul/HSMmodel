#!/bin/bash

for i in `seq 13 24`;
do
    echo Current = "$i"
    echo RESTART_TRIAL=$i SEED=$i GPU DEVICES = 2
    CUDA_VISIBLE_DEVICES=2 python tf_HSM_main_MaxFunc_100k_upgraded.py REGION=1 LR=0.1 RESTART_TRIAL=$i SEED=$i ITERATIONS=100000
done
