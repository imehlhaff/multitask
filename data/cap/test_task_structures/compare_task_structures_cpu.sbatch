#!/bin/bash
# CPU version of compare_task_structures.sbatch: trains the same 3 models, but with compare_task_structures_cpu.py
# (BERT-mini instead of roberta-base) on CPU cores, no GPU. `python compare_task_structures_cpu.py tasks` lists them:
#   index 0  single: major topic only (22 classes)
#   index 1  single: subtopic only (226 classes)
#   index 2  joint: one model with a major topic head and a subtopic head
#
# Lines below an EDIT comment are cluster-specific. Run everything from this directory:
#
#   # The by-document splits must exist: python ../02_split.py (writes ../clean/cap_{train,val,test}.csv)
#   mkdir -p results/logs results/errors
#   # Download the encoder once, so 3 jobs don't race to fill the Hugging Face cache (and in case compute nodes
#   # are offline)
#   python -c "from transformers import AutoTokenizer, AutoModel; m = 'google/bert_uncased_L-4_H-256_A-4'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"
#   # Time one job before submitting everything: its log shows seconds per step, and an epoch is about 3,600 steps
#   sbatch --array=2 --export=ALL,ARGS="--seed 1" compare_task_structures_cpu.sbatch
#   for s in 1 2 3; do sbatch --export=ALL,ARGS="--seed $s" compare_task_structures_cpu.sbatch; done
#
#   # Score on val and test (seconds; the login node is fine). It scores every condition whose runs have all
#   # finished, so it can also be run while jobs are still going:
#   for s in 1 2 3; do python compare_task_structures_cpu.py aggregate --seed $s; done
#
# Other settings: pass script arguments through ARGS, e.g. ARGS="--seed 1 --encoder google/bert_uncased_L-4_H-512_A-8"
# (and the same --encoder to aggregate). Results go to results/<encoder>_seed<seed>/. Rerunning skips runs whose
# results already exist unless ARGS includes --overwrite, so after a failure resubmit only the failed indices:
#   sbatch --array=1 --export=ALL,ARGS="--seed 1" compare_task_structures_cpu.sbatch

#SBATCH --job-name=cap_task_structures_cpu
# EDIT: a CPU partition, with its QOS and account
#SBATCH --partition=ssd
#SBATCH --qos ssd
#SBATCH -A ssd
#SBATCH --mail-user=jgint@uchicago.edu
#SBATCH --mail-type=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
# About 114k training sentences of 64 tokens per epoch, for up to 10 epochs. A guess: check the test job's seconds
# per step
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/errors/%x_%A_%a.out

set -eo pipefail

module load python
source activate multitask
# After activation: conda's activate scripts read unset variables
set -u

cd "$SLURM_SUBMIT_DIR"
# The repo root, so `import multitask` works even if the package isn't pip-installed in the env
export PYTHONPATH="$(cd ../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=1
# Write output as it happens. Otherwise Python buffers it when writing to a file, so the log stays empty for a long
# time and the last lines can be missing if the job crashes.
export PYTHONUNBUFFERED=1

# Hide any GPU, so the job runs the same on any node of the partition
export CUDA_VISIBLE_DEVICES=""
# TensorFlow sizes its thread pools by the node's cores, not the job's. With several jobs on one node, each would start
# a thread per core and they'd slow each other down, so use only the cores Slurm allotted.
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

STEP=${STEP:-train}
ARGS=${ARGS:-}

echo "Job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} CPUs, step ${STEP}, args: ${ARGS}"

if [ "$STEP" = aggregate ]; then
    python compare_task_structures_cpu.py aggregate $ARGS
else
    python compare_task_structures_cpu.py train --task "$SLURM_ARRAY_TASK_ID" $ARGS
fi
