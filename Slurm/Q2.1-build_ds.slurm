#!/bin/bash
#SBATCH -N 1		# Nodes
#SBATCH -c 1		# Cores
#SBATCH -p cpu
#SBATCH --qos=short
#SBATCH --job-name=from-pretrained
#SBATCH --time=0-01:w:00:00

source /etc/profile
source ~/anaconda3/bin/activate
conda init
conda activate ACV


(cd ../Solution/; python ./Q2.1_make_background_dataset.py) &
pids[0]=$!


for pid in ${pids[*]}; do
    wait $pid
done
