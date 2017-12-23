#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -t 00:05:00
#SBATCH -o sum.out
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -t 24:00:00
##SBATCH --qos=bibs-jfallonp-condo
#SBATCH --qos=bibs-tserre-condo
#SBATCH --mem=123G
#SBATCH -J babas
#SBATCH -p bibs-gpu
#SBATCH -o ./tmp/babas_o_%j.txt
#SBATCH -e ./tmp/babas_e_%j.txt
# Load CUDA module
module load cuda

# Compile CUDA program and run
nvcc -arch sm_20 vecadd.cu -o vecadd
./vecadd



for i in `seq 6 10`;
do
    echo Current = "$i"
    echo REGION=1 SEED=$i RESTART_TRIAL=0
    python HSM_theano_SciPy_minimize_multirun.py REGION=1 SEED=$i RESTART_TRIAL=0 |& tee -a "Seed_LOG/HSM_Theano_region1_seed${i}_trial0.txt"
done



#!/bin/bash

# Job Name
#SBATCH -J theanoseed

# Walltime requested
#SBATCH -t 25:00:00

# Provide index values (TASK IDs)
#SBATCH --array=15-16

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e arrayjob-%a.err
#SBATCH -o arrayjob-%a.out

# single core
#SBATCH -c 1

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job

module load theano/0.9.0.nov2017


echo "Running job array number: "$SLURM_ARRAY_TASK_ID
echo REGION=1 SEED=$SLURM_ARRAY_TASK_ID RESTART_TRIAL=0
python HSM_theano_SciPy_minimize_multirun_ccv.py REGION=1 SEED=$SLURM_ARRAY_TASK_ID RESTART_TRIAL=0 |& tee -a "Seed_LOG/HSM_Theano_region1_seed${i}_trial0.txt"