#!/bin/bash

for i in 2 3
do
     for k in 2 3
     do
        ITR=$(($k * 100000))
        echo Current Seed "$i"
        echo Current ITR "$ITR"
        echo Run Code "lr=0.1"
        echo RESTART_TRIAL=$i SEED=$i ITERATIONS=$ITR
        CUDA_VISIBLE_DEVICES=1 python tf_HSM_main_MaxFunc_100k_upgraded.py REGION=1 LR=0.1 RESTART_TRIAL=$i SEED=$i ITERATIONS=$ITR
     done
done
