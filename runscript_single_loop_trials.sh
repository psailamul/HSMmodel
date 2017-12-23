#!/bin/bash

for i in `seq 6 10`;
do
    echo Current = "$i"
    echo REGION=1 SEED=$i RESTART_TRIAL=0
    python HSM_theano_SciPy_minimize_multirun.py REGION=1 SEED=$i RESTART_TRIAL=0 |& tee -a "Seed_LOG/HSM_Theano_region1_seed${i}_trial0.txt"
done
