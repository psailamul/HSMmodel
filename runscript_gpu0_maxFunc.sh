#!/bin/bash

for i in 0 1
do 
     echo Current Num "$i"
     echo Run Code "100k"
     ipython tf_HSM_main_MaxFunc_100k.py GPU_ID=0 RESTART_TRIAL=$i SEED=$i ITERATIONS=100000
done
