#!/bin/bash

for i in 0
do
    for k in  1 10 100
     do
        lr=$(($k * (1000)))
        echo Current Seed "$i"
        echo Current LR "$LR"
        #echo Run Code "lr=0.1"
        echo RESTART_TRIAL=$i SEED=$i LR=$lr
        CUDA_VISIBLE_DEVICES=0 
        python tf_HSM_main_MaxFunc_100k_upgraded.py REGION=1 LR=$lr RESTART_TRIAL=$i SEED=$i ITERATIONS=10000
     done
done


#CUDA_VISIBLE_DEVICES=2 python tf_HSM_main_MaxFunc_100k_upgraded.py LR=0.1 ITERATIONS=1000