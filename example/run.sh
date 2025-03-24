#!/bin/bash

#SBATCH --job-name=tp430
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sbatch_logs/log_%j.log
#SBATCH --partition=short,long
#SBATCH --time=01-23:00:00
#SBATCH --account=mi2lab-hi

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate deepwater

date

python example_tp430_new.py

date