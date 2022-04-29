#!/bin/bash
#SBATCH -N 1		# Nodes
#SBATCH -c 4		# Cores
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=from-fresh
#SBATCH --time=2-00:00:00

source /etc/profile
source ~/anaconda3/bin/activate
conda init
conda activate ACV


(cd ../Solution/; python ./Q2.1_train_background.py --dataroot ../Dataset/Train_Raw/ --init_gain 48 --name game2movie_new --use_wandb --batch_size 4) &
pids[0]=$!


for pid in ${pids[*]}; do
    wait $pid
done
