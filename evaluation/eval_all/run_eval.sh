#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1      
#SBATCH --gres=gpu:1                    
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00                 
#SBATCH --output=/scratch/s02210446/eval_%j.out     
#SBATCH --error=/scratch/s02210446/eval_%j.err      

cd /scratch/s02210446

head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
head_node_ip=$(srun --nodes=1 --ntasks=1 --gpus=0 -w "$head_node" hostname --ip-address)

export head_node
export head_node_ip

MODEL_PATH="models/qwen2.5-3b-instruct-100-gm-60-saiga-5-0.05-rag-selfplay_1-sp_merged"
TASKS="--tasks libra ru_mmlu en_mmlu flores llmaaj" # libra ru_mmlu en_mmlu flores llmaaj

srun --container-image=$PWD/ngc_cuda_pytorch_vllm_20_10_24_v8.sqsh \
     --container-mounts=$PWD:/workspace \
     --container-workdir=/workspace \
     bash EVAL/eval/run_python.sh "$MODEL_PATH" $TASKS