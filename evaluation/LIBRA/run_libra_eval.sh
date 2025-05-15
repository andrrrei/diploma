#!/bin/bash
#SBATCH --job-name=libra_eval   
#SBATCH --nodes=1      
#SBATCH --gres=gpu:1                    
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00                 
#SBATCH --output=/scratch/s02210446/libra_eval_%j.out     
#SBATCH --error=/scratch/s02210446/libra_eval_%j.err      

cd /scratch/s02210446

export NNODES=2
export GPUS_PER_NODE=8
head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
head_node_ip=$(srun --nodes=1 --ntasks=1 --gpus=0 -w "$head_node" hostname --ip-address)

export head_node
export head_node_ip

srun --container-image=$PWD/ngc_cuda_pytorch_24_04_v1+latest.sqsh \
     --container-mounts=$PWD:/workspace \
     --container-workdir=/workspace \
     bash evaluation/LIBRA/run_python.sh
