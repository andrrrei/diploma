#!/bin/bash

current_node=$(hostname)
echo Current Node: $current_node
echo Head Node Name: $head_node
echo Head Node IP: $head_node_ip

rdzv_id="sft-job-${SLURM_JOB_ID}"
rdzv_port="29500"
rdzv_endpoint="${head_node_ip}:${rdzv_port}"
echo Rendezvous ID: $rdzv_id
echo Rendezvous Port: $rdzv_port

pip install -U "transformers==4.45.2" "peft==0.13.0" "datasets==2.18.0" "trl==0.11.1" "accelerate==0.34.2" "mmh3==4.1.0"

torchrun --nnodes=${NNODES} \
         --nproc-per-node=${GPUS_PER_NODE} \
         --rdzv-id=${rdzv_id} \
         --rdzv-backend=c10d \
         --rdzv-endpoint=${rdzv_endpoint} \
         simpo_sp/simpo_sp.py "$@"
