#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=romae-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=./logs/run%j.out
#SBATCH --exclusive

# --------------------------------------------------------------------
# A script for running RoMA experiments on the MareNostrum 5 compute cluster.
#
# Environment variables that must be set:
# VIRTUALENV_LOC : The location of the virtual environment with all dependencies
# EXPERIMENT_NAME : Name of the experiment python package being run
#
# Any arguments passed to this script will be forwarded to the experiment
# --------------------------------------------------------------------
# Load .env file
set -a; source .env; set +a

module load openMPI/5.0.5 cuda/12.8

if [[ -z "${VIRTUALENV_LOC}" ]]; then
  echo "Please set the VIRTUALENV_LOC environment variable in the .env file"
  exit
fi
if [[ -z "${EXPERIMENT_NAME}" ]]; then
  echo "Please set the EXPERIMENT_NAME environment variable in the .env file"
  exit
fi

# Load the virtual environment
# shellcheck source=.env
source "$VIRTUALENV_LOC"

# All command line arguments passed to the script
ARGS="$@"

# Number of GPUS on each GPU node, change depending on the actual hardware
GPUS_PER_NODE=2
# Splitting 24 CPU's between 4 gpus gives 8 cpus per process
CPUS_PER_PROCESS=12

# Number of nodes and processes in the current job
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# Tell the RoMA how many CPU's each dataloader should spawn
export ROMA_TRAINER_NUM_DATASET_WORKERS=$CPUS_PER_PROCESS
# Tell srun how many cpus there are as well
#export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=none

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
    --enable_cpu_affinity \
    --multi_gpu \
    --num_cpu_threads_per_process $CPUS_PER_PROCESS \
    --module \
    --rdzv_backend c10d \
    --mixed_precision no \
    "

export CMD="$LAUNCHER $EXPERIMENT_NAME $ARGS"

echo "Running command: $CMD"
export TORCHDYNAMO_VERBOSE=1
mpirun bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

# Exit the virtualenv for posterity
deactivate
