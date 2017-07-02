#!/bin/bash

for i in 13 14 15 16 17 18 19 20 21 22 23 24
do 
     echo Current Num "$i"
     ipython tf_HSM_main_restart.py GPU_ID=1 RESTART_TRIAL=$i SEED=$i
done
