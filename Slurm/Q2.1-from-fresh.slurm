#!/bin/bash
#SBATCH -N 1		# Nodes
#SBATCH -c 4		# Cores
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=from-fresh
#SBATCH --time=1-00:00:00

source /etc/profile
source ~/anaconda3/bin/activate
conda init
conda activate ACV


(cd ../Solution/; python ./Q2.1_train_frame_style_transfer.py --dataroot ../Dataset/Train_Raw/ --name game2movie --use_wandb --batch_size 4) &
pids[0]=$!


for pid in ${pids[*]}; do
    wait $pid
done
