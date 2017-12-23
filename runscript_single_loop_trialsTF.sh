#!/bin/bash

for i in `seq 0 4`;
do
    echo Current = "$i"
    echo REGION=1 SEED=13 RESTART_TRIAL=$i
    CUDA_VISIBLE_DEVICES=2 python tf_HSM_main_Scipy_jac_np_multrials.py REGION=1 SEED=13 RESTART_TRIAL=$i |& tee -a "Seed_LOG/HSM_Tensorflow_region1_seed13_trial${i}.txt"
done
