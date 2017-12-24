#!/bin/bash

# Job Name
#SBATCH -J HSMtheano_seed

# Walltime requested
#SBATCH -t 25:00:00

# Provide index values (TASK IDs)
#SBATCH --array=16-20

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e arrayjob-%A.err
#SBATCH -o "Seed_LOG/HSM_Theano_region1_seed${SLURM_ARRAY_TASK_ID}_trial0.txt"

# single core
#SBATCH -c 1

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job

module load theano/0.9.0.nov2017


echo "Running job array number: "$SLURM_ARRAY_TASK_ID
echo REGION=1 SEED=$SLURM_ARRAY_TASK_ID RESTART_TRIAL=0
python HSM_theano_SciPy_minimize_multirun_ccv.py REGION=1 SEED=$SLURM_ARRAY_TASK_ID RESTART_TRIAL=0