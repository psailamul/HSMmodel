#!/bin/bash

for i in 0 1
do 
     echo Current Num "$i"
     echo Run Code "100k"
     ipython tf_HSM_main_MaxFunc_100k.py GPU_ID=3 RESTART_TRIAL=$i SEED=$i ITERATIONS=100000
done
python tf_HSM_main_MaxFunc_100k.py GPU_ID=3 RESTART_TRIAL=0 SEED=0 ITERATIONS=100000