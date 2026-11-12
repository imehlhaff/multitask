#!/bin/bash
# Train the 23 models compared by compare_task_structures.py, one per array task on one GPU
# (`python compare_task_structures.py tasks` lists them):
#   indices 0-18  single: one binary model per technique
#   index 19      multilabel: one multilabel head over all 19 techniques
#   index 20      category_heads: one multilabel head per task category
#   index 21      category_trunks: a shared branch per task category, then one binary head per technique
#   index 22      core_trunk: a shared trunk for the four core techniques, then one binary head per technique
#
# Lines marked EDIT are cluster-specific. Run everything from this directory:
#
#   mkdir -p results/logs
#   # Download the encoder once, so 23 jobs don't race to fill the Hugging Face cache (and in case compute nodes
#   # are offline)
#   python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('roberta-base'); AutoModel.from_pretrained('roberta-base')"
#   jid=$(sbatch --parsable compare_task_structures.sbatch)
#
#   # Score on dev. This takes seconds on CPU, so the login node is fine. It scores every condition whose runs
#   # have all finished, so it can also be run while jobs are still going:
#   python compare_task_structures.py aggregate
#   # or queue it to run after the array succeeds:
#   sbatch --dependency=afterok:$jid --array=0 --export=ALL,STEP=aggregate compare_task_structures.sbatch
#
# Other settings: pass script arguments through ARGS. For example
#   sbatch --export=ALL,ARGS="--encoder roberta-large --seed 1 --learning-rate 1e-5" compare_task_structures.sbatch
#   python compare_task_structures.py aggregate --encoder roberta-large --seed 1
# aggregate only takes --encoder, --seed and --run-name, which pick the results directory.
# Results go to results/<encoder>_seed<seed>/, so runs with different encoders or seeds never overwrite each other.
# Rerunning skips runs whose results already exist unless ARGS includes --overwrite (train only), so after a
# failure, resubmit the whole array or only the failed indices: sbatch --array=3,22 compare_task_structures.sbatch

#SBATCH --job-name=semeval_task_structures
#SBATCH --partition=ssd-gpu
#SBATCH --qos ssd
#SBATCH -A ssd
#SBATCH --mail-user=jgint@uchicago.edu
#SBATCH --mail-type=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-22
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/errors/%x_%A_%a.out

set -eo pipefail

module load python
source activate multitask
# After activation: conda's activate scripts read unset variables
set -u

# TensorFlow compiles some GPU kernels at runtime and needs CUDA's libdevice and ptxas for it, from pip's
# nvidia-cuda-nvcc-cu12 (pip install "nvidia-cuda-nvcc-cu12==12.5.82"). Put its ptxas ahead of older ones on PATH,
# such as the Anaconda module's. Without this, the first GPU op fails with "JIT compilation failed".
NVCC_DIR="$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cuda_nvcc"
if [ -d "$NVCC_DIR" ]; then
    export PATH="$NVCC_DIR/bin:$PATH"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVCC_DIR"
else
    echo "Warning: $NVCC_DIR not found; GPU kernels may fail to compile" >&2
fi

cd "$SLURM_SUBMIT_DIR"
# The repo root, so `import multitask` works even if the package isn't pip-installed in the env
export PYTHONPATH="$(cd ../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=1
# Write output as it happens. Otherwise Python buffers it when writing to a file, so the log stays empty for a long
# time and the last lines can be missing if the job crashes.
export PYTHONUNBUFFERED=1

STEP=${STEP:-train}
ARGS=${ARGS:-}

echo "Job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} on $(hostname), step ${STEP}, args: ${ARGS}"

if [ "$STEP" = aggregate ]; then
    python compare_task_structures.py aggregate $ARGS
else
    # Training silently falls back to CPU if TensorFlow can't see the GPU, so print what it sees. Then run one op on
    # it, so a broken CUDA setup fails here in seconds rather than after the data and encoder load.
    python -c "import tensorflow as tf; print('GPUs visible to TensorFlow:', tf.config.list_physical_devices('GPU')); print('GPU op check:', tf.math.rsqrt(tf.constant([4.0])).numpy())"
    python compare_task_structures.py train --task "$SLURM_ARRAY_TASK_ID" $ARGS
fi
