#!/bin/bash
#SBATCH --job-name=mace_md
#SBATCH --partition=skylake-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:p100:1
#SBATCH --time=00:10:00
#SBATCH --output=mace_%j.out
#SBATCH --error=mace_%j.err

# --------------------------------------------------
# Environment
# --------------------------------------------------

module load python-scientific/3.11.5-foss-2023b

source ~/venvs/mace/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MACE_MODEL_PATH="$HOME/models/mace/MACE-OFF23_small.model"

cd "$SLURM_SUBMIT_DIR"

# --------------------------------------------------
# Input arguments
# --------------------------------------------------

INPUT=${1:?Please provide an input structure file}
RHO0=${2:?Please provide the target density}
NSTEPS=${3:-24000}

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Input: $INPUT"
echo "Target density: $RHO0"
echo "Steps: $NSTEPS"


# --------------------------------------------------
# Run simulation
# --------------------------------------------------

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
    --format=csv -l 5 > gpu_${SLURM_JOB_ID}.csv &

GPU_MON_PID=$!

srun python -u short_NPT_run.py "$INPUT" "$RHO0" "$NSTEPS"

kill $GPU_MON_PID

