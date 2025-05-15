#!/bin/bash
#SBATCH --job-name=sft_ss
#SBATCH --nodes=2      
#SBATCH --gres=gpu:8                    
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00                 
#SBATCH --output=/scratch/s02210446/sft_ss_%j.out     
#SBATCH --error=/scratch/s02210446/sft_ss_%j.err        

cd /scratch/s02210446

export SAMPLE_FRACTION=0.4
export MAX_ANSWER_TOKENS=100000000
export BASE_MODEL_NAME="models2/qwen2.5-3b-instruct-100-gm_merged"
export OUTPUT_DIR="models2/qwen2.5-3b-instruct-100-gm-40-saiga"
export OPUS_SCORE_THRESHOLD=8

export NNODES=2
export GPUS_PER_NODE=8
head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
head_node_ip=$(srun --nodes=1 --ntasks=1 --gpus=0 -w "$head_node" hostname --ip-address)

export head_node
export head_node_ip

srun --container-image=$PWD/ngc_cuda_pytorch_24_04_v1+latest.sqsh \
     --container-mounts=$PWD:/workspace \
     --container-workdir=/workspace \
     bash sft_ss/run_python.sh "$SAMPLE_FRACTION" "$MAX_ANSWER_TOKENS" "$BASE_MODEL_NAME" "$OUTPUT_DIR" "$OPUS_SCORE_THRESHOLD"
