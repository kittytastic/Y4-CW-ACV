#!/bin/bash
#SBATCH -N 1		# Nodes
#SBATCH -c 1		# Cores
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=debug
#SBATCH --job-name=dbg

source /etc/profile
source ~/anaconda3/bin/activate
conda init
conda activate DL

python --version &
pids[0]=$!

sleep 10 &
pids[1]=$!

for pid in ${pids[*]}; do
    wait $pid
done
