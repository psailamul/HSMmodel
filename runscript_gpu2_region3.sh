#!/bin/bash

for i in 25 26 27 28 29 30 31 32 33 34 35 36 37
do 
     echo Current Num "$i"
     ipython tf_HSM_main_restart_region3.py GPU_ID=2 RESTART_TRIAL=$i SEED=$i
done
