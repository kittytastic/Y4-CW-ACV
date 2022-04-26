#!/bin/bash
#SBATCH -N 1		# Nodes
#SBATCH -c 2		# Cores
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=HP
#SBATCH --time=0-03:00:00

source /etc/profile
source ~/anaconda3/bin/activate
conda init
conda activate ACV


(cd ../Solution/; python Q1.1_apply_human_patch_extract.py -b 16 -w 2) &
pids[0]=$!


for pid in ${pids[*]}; do
    wait $pid
done
