#!/bin/bash

#SBATCH --job-name=master_d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=cuda
#SBATCH --time=24:00:00

# Default values
OUTDIR="/home/dgjer/master/Master-Rad/denoised_smoothing_results/stability_obj/cifar10_dncnn_0_20"
NOISE=0.2
EPOCHS=10
OBJECTIVE="stability"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --outdir=*)
            OUTDIR="${1#*=}"
            shift
            ;;
        --noise=*)
            NOISE="${1#*=}"
            shift
            ;;
        --epochs=*)
            EPOCHS="${1#*=}"
            shift
            ;;
        --objective=*)
            OBJECTIVE="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

source ../env_gpu/bin/activate
which python

cd ../denoised_smoothing

# Run the appropriate command based on objective
if [ "$OBJECTIVE" = "stability" ]; then
    python code/train_denoiser.py \
        --dataset cifar10 \
        --arch cifar_dncnn \
        --outdir "$OUTDIR" \
        --classifier "/home/dgjer/master/Master-Rad/base_training_checkpoints/sweep_2025-07-06_01-56-46/2025-07-06_01-56-46/checkpoints/model_normal.pth" \
        --noise "$NOISE" \
        --epochs "$EPOCHS" \
        --objective "$OBJECTIVE" \
        --synergy
else
    python code/train_denoiser.py \
        --dataset cifar10 \
        --arch cifar_dncnn \
        --outdir "$OUTDIR" \
        --noise "$NOISE" \
        --epochs "$EPOCHS" \
        --objective "$OBJECTIVE"
fi
