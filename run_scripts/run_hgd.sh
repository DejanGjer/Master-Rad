#!/bin/bash
#SBATCH --job-name=denoising
#SBATCH --ntasks=1
#SBATCH --nodelist=n17
#SBATCH --partition=cuda
#SBATCH --output slurm.%J.out
#SBATCH --error slurm.%J.err
#SBATCH --time=24:00:00

cd ..
CUDA_VISIBLE_DEVICES=0 python main.py