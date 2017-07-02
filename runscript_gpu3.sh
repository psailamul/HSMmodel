#!/bin/bash

for i in 38 39 40 41 42 43 44 45 46 47 48 49 50
do 
     echo Current Num "$i"
     ipython tf_HSM_main_restart.py GPU_ID=3 RESTART_TRIAL=$i SEED=$i
done
