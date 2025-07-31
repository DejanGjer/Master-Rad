#!/bin/bash
#SBATCH --job-name=master_h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=cuda
#SBATCH --time=24:00:00

source ../env_gpu/bin/activate
which python

train_py_path=$1
train_dir=$(dirname "$train_py_path")

cd "$train_dir"
python "$(basename "$train_py_path")"
