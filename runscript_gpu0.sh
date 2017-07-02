#!/bin/bash

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12
do 
     echo Current Num "$i"
     ipython tf_HSM_main_restart.py GPU_ID=0 RESTART_TRIAL=$i SEED=$i
done
