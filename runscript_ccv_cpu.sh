#!/bin/bash

# set name of job
#SBATCH --job-name=HSMtheano_seed

# Request 1 CPU core  === cpu per task
#SBATCH -n 8

# set the number of nodes
#SBATCH --nodes=1

# Request a GPU partition node and access to 1 GPU
##SBATCH -p gpu --gres=gpu:1

# Request memory
#SBATCH --mem=10G

# set max wallclock time
#SBATCH -t 24:00:00

# Provide index values (TASK IDs)
#SBATCH --array=21-50

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e Seed_LOG/attempt2/HSM_Theano_region1_seed%a_trial0_%A.txt
#SBATCH -o Seed_LOG/attempt2/HSM_Theano_region1_seed%a_trial0_%A.out 

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=pachaya_sailamul@brown.edu

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job
module load theano/0.9.0 gcc/5.2.0

# run the application
echo "Running job array number: "$SLURM_ARRAY_TASK_ID
echo REGION=1 SEED=$SLURM_ARRAY_TASK_ID RESTART_TRIAL=0 MAXITER=100000
python HSM_theano_SciPy_minimize_ccv.py REGION=1 SEED=$SLURM_ARRAY_TASK_ID MAXITER=100000




