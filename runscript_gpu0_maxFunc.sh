#!/bin/bash

for i in 1
do 
     echo Current Num "$i"
     echo Run Code "100k"
     python tf_HSM_main_MaxFunc_100k_upgraded.py RESTART_TRIAL=$i SEED=$i ITERATIONS=100000
done
