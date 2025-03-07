#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=robite-plasticc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=./logs/run%j.out
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --exclusive

# --------------------------------------------------------------------
# A script for running RoBiTE pretraining on the Leonardo compute cluster.
#
# Environment variables that should be set:
# VIRTUALENV_LOC : The location of the virtual environment with all dependencies
# ROBITE_PACKAGE_ROOT : Location of the RoBiTE package
# --------------------------------------------------------------------

if [ $# -ne 1 ]; then
  echo "Please specify the RoBiTE subcommand to run as an argument"
fi

# Load .env file
set -a; source .env; set +a


module load cuda/12.3
module load python/3.11.6--gcc--8.5.0


# Load the virtual environment
source "$VIRTUALENV_LOC"

# Use the first node's hostname as the master node address
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=6000

echo "Master address: $MASTER_ADDR"

export LAUNCHER="accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --multi_gpu \
    --dynamo_mode max-autotune \
    "

export PROGRAM="$ROBITE_PACKAGE_ROOT/robite $1"
export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

# Exit the virtualenv for posterity
deactivate
