#!/bin/bash
#SBATCH --job-name=check_dataset
#SBATCH --output=/scratch/s02210446/check_dataset_%j.out
#SBATCH --error=/scratch/s02210446/check_dataset_%j.err

srun --container-image=$PWD/ngc_cuda_pytorch_24_04_v1+latest.sqsh \
     --container-mounts=$PWD:/workspace \
     --container-workdir=/workspace \
     python datasets/check_dataset.py
