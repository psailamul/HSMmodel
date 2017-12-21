#!/bin/bash

for i in `seq 1 12`;
do
    echo Current = "$i"
    echo REGION=1 SEED=$i RESTART_TRIAL=0
    CUDA_VISIBLE_DEVICES=3 python tf_HSM_main_Scipy_jac_np_multrials.py REGION=1 SEED=$i RESTART_TRIAL=0 | tee Seed_LOG/HSM_Tensorflow_region1_seed$i_trial0.txt
done
