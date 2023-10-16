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
python sentiment-analysis.py

