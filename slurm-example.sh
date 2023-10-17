#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-00:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB
#SBATCH --account=eecs595f23_class

# set up job
module load python cuda
pushd /home/tianlix/EECS595
source venv/bin/activate

# run job
for bs in 16 32
do
    for lr in 1e-4 1e-3
    do
        for num_ep in 5 10
        do
            python EECS592_hw3/finetune.py --batch_size $bs --lr $lr --num_epochs $num_ep
        done
    done
done

