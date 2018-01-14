#!/bin/bash

# set name of job
#SBATCH --job-name=tfHSM

# set the number of nodes
#SBATCH --nodes=1

# Request 1 CPU core  === cpu per task
#SBATCH -n 1

# Request memory
#SBATCH --mem=10G

# set max wallclock time
#SBATCH -t 24:00:00

# Provide index values (TASK IDs)
#SBATCH --array=1-20

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e Seed_LOG/tfHSM/region1_seed13_LGN%a_trial0_hyper_%A.txt
#SBATCH -o Seed_LOG/tfHSM/region1_seed13_LGN%a_trial0_hyper_%A.out

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=pachaya_sailamul@brown.edu

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job
# Set up the environment by loading modules

module load cuda/8.0.44 cudnn/5.1 tensorflow/1.0.0

echo "Running job array number: "$SLURM_ARRAY_TASK_ID
echo REGION=1 SEED=13 ITERATIONS=1 NUM_LGN=$SLURM_ARRAY_TASK_ID HLSR=0.2  STOP EARLY 
python tf_HSM_main_Scipy_jac_np_multrials_ccv.py SEED=13 NUM_LGN=$SLURM_ARRAY_TASK_ID HLSR=0.2 ITERATIONS=1 REGION=1 |& tee -a "Seed_LOG/tfHSM/check_param_seed13_trial0_numLGN${SLURM_ARRAY_TASK_ID}_hlsr0.2.txt"

##echo "Running job array number: "$SLURM_ARRAY_TASK_ID
##echo REGION=1 SEED=$SLURM_ARRAY_TASK_ID ITERATIONS=100000 NUM_LGN=$1 HLSR=$2  STOP EARLY 
##python tf_HSM_main_CONV5_ccv.py SEED=$SLURM_ARRAY_TASK_ID NUM_LGN=$1 HLSR=$2 ITERATIONS=100000 REGION=1 |& tee -a "Seed_LOG/AntolikConv5/AntolikConv_region1_seed${SLURM_ARRAY_TASK_ID}_trial0_LGN${1}_hlsr${2}.txt"
