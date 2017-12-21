#!/bin/bash

for i in `seq 1 5`;
do
    echo Current = "$i"
    echo REGION=1 SEED=13 RESTART_TRIAL=$i
    python HSM_theano_SciPy_minimize_multirun.py REGION=1 SEED=13 RESTART_TRIAL=$i | tee Seed_LOG/HSM_Theano_region1_seed13_trial$i.txt
done
