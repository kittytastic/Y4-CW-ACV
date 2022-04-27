#!/bin/bash
#SBATCH -N 1		# Nodes
#SBATCH -c 4		# Cores
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=strech
#SBATCH --time=0-12:00:00

source /etc/profile
source ~/anaconda3/bin/activate
conda init
conda activate ACV


(cd ../Solution/; python ./Q2.3_train.py --dataroot stretch --batch_size 4 --name better_patch_strech --use_wandb --init_gain 12) &
pids[0]=$!


for pid in ${pids[*]}; do
    wait $pid
done
