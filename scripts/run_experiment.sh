#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=romae-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=./logs/run%j.out
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --exclusive

# --------------------------------------------------------------------
# A script for running RoMAE experiments on the Leonardo compute cluster.
#
# Environment variables that must be set:
# VIRTUALENV_LOC : The location of the virtual environment with all dependencies
# EXPERIMENT_NAME : Name of the experiment python package being run
#
# Any arguments passed to this script will be forwarded to the experiment
# --------------------------------------------------------------------
# Load .env file
set -a; source .env; set +a

if [[ -z "${VIRTUALENV_LOC}" ]]; then
  echo "Please set the VIRTUALENV_LOC environment variable in the .env file"
  exit
fi
if [[ -z "${EXPERIMENT_NAME}" ]]; then
  echo "Please set the EXPERIMENT_NAME environment variable in the .env file"
  exit
fi

module load cuda/12.3
module load python/3.11.6--gcc--8.5.0

# Load the virtual environment
# shellcheck source=.env
source "$VIRTUALENV_LOC"

# All command line arguments passed to the script
ARGS="$@"

# Number of GPUS on each booster node, change depending on the actual hardware
GPUS_PER_NODE=4
# Splitting 32 CPU's between 4 gpus gives 8 cpus per process
CPUS_PER_PROCESS=8

# Number of nodes and processes in the current job
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# Tell the RoMA how many CPU's each dataloader should spawn
export ROMA_TRAINER_NUM_DATASET_WORKERS=$CPUS_PER_PROCESS

# Use the first node's hostname as the master node address
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=6000

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Machine rank: $SLURM_JOBID"
echo "Num processes: $NUM_PROCESSES"
echo "Num machines: $NNODES"

export LAUNCHER="accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --multi_gpu \
    --enable_cpu_affinity \
    --num_cpu_threads_per_process $CPUS_PER_PROCESS \
    --mixed_precision bf16 \
    --module \
    --rdzv_backend c10d \
    --dynamo_mode default \
    --dynamo_backend inductor \
    --dynamo_use_dynamic \
    "

export CMD="$LAUNCHER $EXPERIMENT_NAME $ARGS"

echo "Running command: $CMD"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

# Exit the virtualenv for posterity
deactivate
