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

pip install -U "transformers==4.45.2" "peft==0.13.0" "datasets==2.18.0" "accelerate==0.34.2" "pymorphy2==0.9.1" "rouge-score==0.1.2" "evalica==0.3.2" "fire==0.7.0" "shortuuid==1.0.13"

python evaluation/LLMAAJ/llmaaj_eval.py