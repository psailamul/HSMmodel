#!/bin/bash

for ((i=25; i <= 50 ; i=i+4)); 
do
    echo Current = "$i"
    echo REGION=1 SEED=$i RESTART_TRIAL=0
    CUDA_VISIBLE_DEVICES=3 python tf_HSM_main_Scipy_jac_np_multrials.py REGION=1 SEED=$i RESTART_TRIAL=0 |& tee -a "Seed_LOG/HSM_Tensorflow_region1_seed${i}_trial0.txt"
done
