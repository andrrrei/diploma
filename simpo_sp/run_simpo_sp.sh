#!/bin/bash
#SBATCH --job-name=simpo_sp_1
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/s02210446/simpo_sp_1_%j.out
#SBATCH --error=/scratch/s02210446/simpo_sp_1_%j.err

cd /scratch/s02210446

export CPO_ALPHA=0.2
export BASE_MODEL_NAME="models/qwen2.5-3b-instruct-100-gm-60-saiga-5-0.05-rag_merged"
export OUTPUT_DIR="models/qwen2.5-3b-instruct-100-gm-60-saiga-5-0.05-rag-selfplay_1-sp"
export SAMPLE_FRACTION=1.0

export NNODES=2
export GPUS_PER_NODE=8
head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
head_node_ip=$(srun --nodes=1 --ntasks=1 --gpus=0 -w "$head_node" hostname --ip-address)

export head_node
export head_node_ip

srun --container-image=$PWD/ngc_cuda_pytorch_24_04_v1+latest.sqsh \
     --container-mounts=$PWD:/workspace \
     --container-workdir=/workspace \
     bash simpo_sp/run_python.sh "$CPO_ALPHA" "$BASE_MODEL_NAME" "$OUTPUT_DIR" "$SAMPLE_FRACTION"
