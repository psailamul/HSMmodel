#!/bin/bash

# set name of job
#SBATCH --job-name=Allen_HSM
# set the number of nodes
#SBATCH --nodes=1

# Request 1 CPU core  === cpu per task
#SBATCH -n 1

# Request a GPU partition node and access to 1 GPU
##SBATCH -p bibs-gpu --gres=gpu:1

# Request memory
#SBATCH --mem=10G

# set max wallclock time
#SBATCH -t 25:00:00

# Provide index values (TASK IDs)
#SBATCH --array=1-5

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e Seed_LOG/subsampling/tfHSM_region1subsampling_seed%a_trial0_%A.txt
#SBATCH -o Seed_LOG/subsampling/tfHSM_region1subsampling_seed%a_trial0_%A.out

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=pachaya_sailamul@brown.edu

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job
# Set up the environment by loading modules

module load cuda/8.0.61 cudnn/5.1 tensorflow/1.1.0_cpu 

echo "Running job array number: "$SLURM_ARRAY_TASK_ID
echo REGION=$1 SEED=$SLURM_ARRAY_TASK_ID ITERATIONS=1 NUM_LGN=9 HLSR=0.2  STOP EARLY 
python tf_HSM_main_Scipy_jac_np_multrials_ccv_subsampling.py SEED=$SLURM_ARRAY_TASK_ID NUM_LGN=9 HLSR=0.2 ITERATIONS=1 REGION=$1



## [90, 180, 450, 900, 1350, 1800] 
## sbatch runsbatch_cpu_subsampling.sh 90
